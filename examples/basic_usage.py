"""Basic usage example for llm-key-pool.

Simulates calling a flaky LLM provider: keys get rate-limited (429),
one key is revoked (401), and the pool rotates and benches keys
accordingly.

Run with::

    python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import random

from keypool import CooldownPolicy, KeyPool, NoKeysAvailableError

# Pretend these came from the environment:
#   KEYPOOL_CLAUDE_KEYS=sk-ant-alpha,sk-ant-beta,sk-ant-revoked
#   KEYPOOL_GEMINI_KEYS=AIza-one,AIza-two
FAKE_ENV = {
    "KEYPOOL_CLAUDE_KEYS": "sk-ant-alpha,sk-ant-beta,sk-ant-revoked",
    "KEYPOOL_GEMINI_KEYS": "AIza-one,AIza-two",
}


async def mock_llm_call(family: str, key: str) -> str:
    """Pretend to call an LLM API and return a status.

    ``sk-ant-revoked`` always fails auth; other keys are occasionally
    rate limited.
    """
    await asyncio.sleep(0.05)  # simulated network latency
    if key == "sk-ant-revoked":
        raise MockHTTPError(401)
    if random.random() < 0.3:
        raise MockHTTPError(429)
    return f"[{family}] completion generated with {key}"


class MockHTTPError(Exception):
    """Stand-in for an HTTP error from a provider SDK."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


async def complete(pool: KeyPool, family: str, attempts: int = 3) -> str:
    """Try up to *attempts* keys from *family* for a single completion."""
    for _ in range(attempts):
        try:
            key = await pool.get_key(family)
        except NoKeysAvailableError as exc:
            return f"[{family}] pool exhausted (retry in {exc.retry_after:.0f}s)"
        try:
            result = await mock_llm_call(family, key)
        except MockHTTPError as exc:
            print(f"    {key} failed with {exc.status_code}, cooling down")
            await pool.mark_failure(key, exc.status_code)
            continue
        await pool.mark_success(key)
        return result
    return f"[{family}] all attempts failed"


async def main() -> None:
    # Short cooldowns so the demo stays snappy.
    policy = CooldownPolicy(rate_limit_cooldown=2.0, auth_cooldown_base=30.0)
    pool = KeyPool.from_env(environ=FAKE_ENV, cooldown_policy=policy)
    print(f"Registered families: {pool.families}\n")

    for i in range(6):
        family = "claude" if i % 2 == 0 else "gemini"
        print(f"Request {i + 1} ({family}):")
        print(f"    -> {await complete(pool, family)}")

    print("\nFinal pool state:")
    for family, states in pool.snapshot().items():
        for s in states:
            flags = " LIKELY-INVALID" if s.likely_invalid else ""
            print(f"    {family}: {s.key} failures={s.failure_count}{flags}")


if __name__ == "__main__":
    asyncio.run(main())
