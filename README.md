# llm-key-pool

**Provider-agnostic, family-aware API key pooling for LLM proxies — rotation, cooldowns, and cross-provider isolation in one small stdlib-only library.**

<!--
  TODO: replace these placeholder badges with real ones once the package
  is published to PyPI and CI is set up. Suggested:
  - PyPI:    https://img.shields.io/pypi/v/llm-key-pool
  - License: https://img.shields.io/badge/license-MIT-blue
  - Tests:   GitHub Actions workflow badge (pytest)
-->
![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-lightgrey)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

---

## The Problem

If you run any kind of LLM proxy or gateway — even a small internal one — you
eventually end up holding **multiple API keys for multiple providers**:
a few Anthropic keys, a couple of Google AI Studio keys, an OpenAI key or two.
Maybe they belong to different accounts, different billing tiers, or different
rate-limit buckets.

The moment you have more than one key, you're forced to hand-roll the same
plumbing everyone else hand-rolls:

- **Rotation.** Spreading requests across keys so no single key eats all the
  traffic (and all the rate limits).
- **Cooldowns.** When a key gets a `429`, it needs to sit out briefly. When it
  gets a `401`/`403`, it's probably revoked and should sit out much longer —
  but not be discarded forever, because auth errors are sometimes transient.
- **Family isolation.** The quiet, embarrassing bug: an Anthropic key being
  sent to the Gemini API because a lookup table got shared or a config got
  copy-pasted. The provider rejects it, you burn a retry, and your logs fill
  with misleading auth errors.
- **State tracking.** Which key was used last? How many times has this one
  failed? Is it worth trying again?

None of this is hard, but all of it is fiddly, and getting the concurrency
right (two async tasks grabbing keys at the same time) is where the subtle
bugs live. Everyone writes this code inside their proxy, badly, once per
project.

## Why this exists

The existing options sit at two extremes:

- **Full gateways** like LiteLLM are excellent, but they're an entire proxy —
  routing, spend tracking, a config format, a server. If you already *have* a
  proxy and just need key management, adopting a second gateway is the wrong
  shape of dependency.
- **Nothing.** Below the gateway tier, there's no small, standalone library
  that just does key pooling. So it gets reimplemented inline, project after
  project.

`llm-key-pool` is the missing middle: a single-purpose library that manages
the keys and nothing else. It doesn't know how to call any provider, doesn't
parse responses, and doesn't care what HTTP client you use. You tell it what
happened (`mark_success` / `mark_failure`); it tells you which key to use next.

Design constraints it commits to:

- **Zero dependencies.** Core logic is pure Python stdlib (`asyncio`,
  `dataclasses`, `time`). It will never bring a transitive dependency into
  your proxy.
- **Async-safe.** All state mutations go through an `asyncio.Lock`, so it can
  be shared freely across FastAPI handlers or any asyncio tasks.
- **Hard family isolation.** A key belongs to exactly one family. Registering
  the same key under two families raises `DuplicateKeyError`, and lookups only
  ever search the requested family. A `claude` key structurally *cannot* be
  returned for a `gemini` request.
- **Testable by construction.** The clock is injectable, so cooldown behavior
  is tested deterministically — no `sleep()` in the test suite.

## Installation

```bash
pip install llm-key-pool
```

<!-- Placeholder: package not yet published to PyPI. Until then: -->
Until the PyPI release, install from source:

```bash
git clone https://github.com/alibarkat/llm-key-pool
cd llm-key-pool
pip install -e .
```

Requires Python 3.9+.

## Quick start

Set your keys in the environment, one variable per provider family:

```bash
export KEYPOOL_CLAUDE_KEYS=sk-ant-alpha,sk-ant-beta
export KEYPOOL_GEMINI_KEYS=AIza-one,AIza-two
```

Then wire the pool into your request path:

```python
import asyncio
from keypool import KeyPool, NoKeysAvailableError

pool = KeyPool.from_env()  # discovers KEYPOOL_*_KEYS automatically

async def complete(family: str, prompt: str) -> str:
    for _ in range(3):  # try up to 3 different keys
        try:
            key = await pool.get_key(family)
        except NoKeysAvailableError as exc:
            # Every key is cooling down; exc.retry_after says when to retry.
            raise RuntimeError(f"Pool exhausted, retry in {exc.retry_after:.0f}s")

        try:
            response = await call_your_provider(family, key, prompt)
        except ProviderHTTPError as exc:
            await pool.mark_failure(key, exc.status_code)  # benches the key
            continue                                        # next key
        else:
            await pool.mark_success(key)                    # clears flags
            return response

    raise RuntimeError("All attempts failed")
```

That's the whole integration surface: `get_key`, `mark_failure`,
`mark_success`. A runnable end-to-end demo (with a mock flaky provider) lives
in [`examples/basic_usage.py`](examples/basic_usage.py).

Optionally, validate all keys at startup:

```python
async def checker(family: str, key: str) -> bool:
    ...  # make a cheap authenticated call, return True if it works

report = await pool.check_health(checker)
# {'claude': {'sk-ant-alpha': True, 'sk-ant-beta': False}, ...}
# Unhealthy keys are automatically benched as if they returned 401.
```

## Configuration

Keys are configured through environment variables:

| Variable pattern | Example | Meaning |
|---|---|---|
| `KEYPOOL_<FAMILY>_KEYS` | `KEYPOOL_CLAUDE_KEYS=k1,k2,k3` | Comma-separated keys for family `claude` |
| `KEYPOOL_GEMINI_KEYS` | `KEYPOOL_GEMINI_KEYS=AIza-1` | Keys for family `gemini` |
| `KEYPOOL_GPT_KEYS` | `KEYPOOL_GPT_KEYS=sk-1,sk-2` | Keys for family `gpt` |

Notes:

- Family names come from the variable name and are case-insensitive at lookup
  time (`get_key("CLAUDE")` and `get_key("claude")` are equivalent).
- Whitespace around keys is stripped; empty entries are ignored.
- The `KEYPOOL` prefix is configurable: `KeyPool.from_env(prefix="MYAPP")`
  reads `MYAPP_<FAMILY>_KEYS` instead.
- You can skip env vars entirely and register programmatically:

  ```python
  pool = KeyPool()
  pool.register_family("claude", ["sk-ant-1", "sk-ant-2"])
  pool.add_key("claude", "sk-ant-3")
  ```

Cooldown durations are configurable through `CooldownPolicy`:

```python
from keypool import CooldownPolicy, KeyPool

policy = CooldownPolicy(
    rate_limit_cooldown=60.0,   # 429
    auth_cooldown_base=300.0,   # 401/403, doubles per consecutive failure
    auth_cooldown_max=3600.0,   # backoff cap
    generic_cooldown=30.0,      # any other error status
)
pool = KeyPool.from_env(cooldown_policy=policy)
```

## How cooldowns work

When you call `mark_failure(key, status_code)`, the key is benched based on
what kind of failure it was:

| Status | Behavior |
|---|---|
| `429` | Short cooldown (**60s** default). Rate limits clear quickly; the key rejoins rotation soon. |
| `401` / `403` | Key is flagged **likely-invalid** and benched with exponential backoff: **300s**, then 600s, 1200s… capped at **3600s**. Each consecutive auth failure doubles the wait. |
| anything else | Generic short cooldown (**30s** default) — covers 500s and transient provider errors. |

Two details worth knowing:

- **Likely-invalid keys are deprioritized, not deleted.** Once its cooldown
  expires, a flagged key is only handed out if *no healthy key is available*.
  This means a revoked key stops burning your retries, but a key that hit a
  transient auth blip (or gets un-revoked) can still recover.
- **Success fully rehabilitates a key.** `mark_success(key)` clears the
  cooldown, resets the auth-failure streak, and removes the likely-invalid
  flag — a key that just worked is evidently valid. The lifetime
  `failure_count` is kept for observability.

Selection among available keys is least-recently-used, which behaves as
round-robin under steady load.

When every key in a family is benched, `get_key` raises
`NoKeysAvailableError` with a `retry_after` value — handy for propagating a
`Retry-After` header from your proxy.

## Contributing

Contributions are welcome. The bar for the core module is deliberately
strict: **no new runtime dependencies** in `keypool/` — extensions (e.g. a
Redis-backed state store) belong in optional modules.

To get set up:

```bash
git clone https://github.com/alibarkat/llm-key-pool
cd llm-key-pool
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Guidelines:

- Add tests for any behavior change — the suite in
  [`tests/test_pool.py`](tests/test_pool.py) uses an injectable fake clock,
  so cooldown tests never need to sleep.
- Keep type hints and docstrings complete; the package ships as typed.
- Open an issue first for anything larger than a bug fix, so design can be
  discussed before code.

Ideas on the roadmap: Redis backend for multi-process pools, pluggable
selection strategies (weighted, cost-aware), metrics hooks.

## License

[MIT](LICENSE) © Ali Barkat
