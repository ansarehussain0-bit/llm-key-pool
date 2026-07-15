"""Tests for keypool.pool.

Uses a fake clock injected into the pool so cooldown expiry can be
tested without sleeping.
"""

from __future__ import annotations

import asyncio

import pytest

from keypool import (
    CooldownPolicy,
    DuplicateKeyError,
    KeyPool,
    NoKeysAvailableError,
    UnknownFamilyError,
    UnknownKeyError,
)


class FakeClock:
    """Controllable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def pool(clock: FakeClock) -> KeyPool:
    p = KeyPool(clock=clock)
    p.register_family("claude", ["ck-1", "ck-2", "ck-3"])
    p.register_family("gemini", ["gk-1"])
    return p


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# Family isolation
# ----------------------------------------------------------------------

def test_key_never_crosses_families(pool: KeyPool) -> None:
    """A gemini request must never receive a claude key, and vice versa."""
    async def scenario():
        for _ in range(10):
            assert (await pool.get_key("claude")).startswith("ck-")
            assert (await pool.get_key("gemini")).startswith("gk-")

    run(scenario())


def test_duplicate_key_across_families_rejected(pool: KeyPool) -> None:
    with pytest.raises(DuplicateKeyError):
        pool.add_key("gemini", "ck-1")


def test_duplicate_key_same_family_is_idempotent(pool: KeyPool) -> None:
    pool.add_key("claude", "ck-1")
    assert len(pool.snapshot()["claude"]) == 3


def test_unknown_family_raises(pool: KeyPool) -> None:
    with pytest.raises(UnknownFamilyError):
        run(pool.get_key("mistral"))


def test_family_name_is_case_insensitive(pool: KeyPool) -> None:
    key = run(pool.get_key("CLAUDE"))
    assert key.startswith("ck-")


def test_empty_family_raises_no_keys(clock: FakeClock) -> None:
    p = KeyPool(clock=clock)
    p.register_family("empty")
    with pytest.raises(NoKeysAvailableError) as excinfo:
        run(p.get_key("empty"))
    assert excinfo.value.retry_after is None


# ----------------------------------------------------------------------
# Round-robin / LRU selection
# ----------------------------------------------------------------------

def test_round_robin_within_family(pool: KeyPool, clock: FakeClock) -> None:
    """With no failures, keys rotate: each key used once per cycle."""
    async def scenario():
        seen = []
        for _ in range(6):
            clock.advance(1.0)  # distinct last_used_at timestamps
            seen.append(await pool.get_key("claude"))
        return seen

    seen = run(scenario())
    first_cycle, second_cycle = seen[:3], seen[3:]
    assert sorted(first_cycle) == ["ck-1", "ck-2", "ck-3"]
    assert first_cycle == second_cycle  # same LRU order repeats


def test_unused_keys_preferred(pool: KeyPool, clock: FakeClock) -> None:
    async def scenario():
        first = await pool.get_key("claude")
        clock.advance(1.0)
        second = await pool.get_key("claude")
        return first, second

    first, second = run(scenario())
    assert first != second


# ----------------------------------------------------------------------
# Cooldowns
# ----------------------------------------------------------------------

def test_429_applies_short_cooldown(pool: KeyPool, clock: FakeClock) -> None:
    async def scenario():
        key = await pool.get_key("gemini")
        await pool.mark_failure(key, 429)
        with pytest.raises(NoKeysAvailableError) as excinfo:
            await pool.get_key("gemini")
        assert excinfo.value.retry_after == pytest.approx(60.0)

        clock.advance(61.0)
        assert await pool.get_key("gemini") == key  # recovered

    run(scenario())


def test_401_applies_long_cooldown_and_flags_invalid(
    pool: KeyPool, clock: FakeClock
) -> None:
    async def scenario():
        key = await pool.get_key("gemini")
        await pool.mark_failure(key, 401)

        state = pool.snapshot()["gemini"][0]
        assert state.likely_invalid is True

        # Still benched after the 429 window (60s) — auth cooldown is longer.
        clock.advance(61.0)
        with pytest.raises(NoKeysAvailableError):
            await pool.get_key("gemini")

        clock.advance(300.0)  # past the 300s auth base
        assert await pool.get_key("gemini") == key

    run(scenario())


def test_auth_cooldown_backs_off_exponentially(clock: FakeClock) -> None:
    policy = CooldownPolicy(auth_cooldown_base=100.0, auth_cooldown_max=350.0)
    p = KeyPool(cooldown_policy=policy, clock=clock)
    p.register_family("f", ["k"])

    async def cooldown_after_failure(status: int) -> float:
        await p.mark_failure("k", status)
        state = p.snapshot()["f"][0]
        return state.cooldown_until - clock.now

    async def scenario():
        assert await cooldown_after_failure(401) == pytest.approx(100.0)
        assert await cooldown_after_failure(403) == pytest.approx(200.0)
        # 400 would exceed the cap of 350.
        assert await cooldown_after_failure(401) == pytest.approx(350.0)

    run(scenario())


def test_non_auth_error_uses_generic_cooldown(pool: KeyPool, clock: FakeClock) -> None:
    async def scenario():
        key = await pool.get_key("gemini")
        await pool.mark_failure(key, 500)
        with pytest.raises(NoKeysAvailableError) as excinfo:
            await pool.get_key("gemini")
        assert excinfo.value.retry_after == pytest.approx(30.0)

    run(scenario())


def test_likely_invalid_keys_deprioritized(pool: KeyPool, clock: FakeClock) -> None:
    """After its auth cooldown expires, a flagged key is only used last."""
    async def scenario():
        await pool.mark_failure("ck-1", 401)
        clock.advance(10_000.0)  # everything is off cooldown now
        picks = set()
        for _ in range(2):
            clock.advance(1.0)
            picks.add(await pool.get_key("claude"))
        return picks

    picks = run(scenario())
    assert picks == {"ck-2", "ck-3"}  # ck-1 skipped while healthy keys exist


# ----------------------------------------------------------------------
# Failure tracking and recovery
# ----------------------------------------------------------------------

def test_mark_success_clears_cooldown_and_invalid_flag(
    pool: KeyPool, clock: FakeClock
) -> None:
    async def scenario():
        await pool.mark_failure("gk-1", 401)
        await pool.mark_success("gk-1")
        state = pool.snapshot()["gemini"][0]
        assert state.likely_invalid is False
        assert state.auth_failure_streak == 0
        assert state.failure_count == 1  # history is kept
        assert await pool.get_key("gemini") == "gk-1"

    run(scenario())


def test_failure_count_accumulates(pool: KeyPool) -> None:
    async def scenario():
        await pool.mark_failure("gk-1", 429)
        await pool.mark_failure("gk-1", 500)
        await pool.mark_failure("gk-1", 401)

    run(scenario())
    assert pool.snapshot()["gemini"][0].failure_count == 3


def test_unknown_key_raises(pool: KeyPool) -> None:
    with pytest.raises(UnknownKeyError):
        run(pool.mark_failure("nope", 429))
    with pytest.raises(UnknownKeyError):
        run(pool.mark_success("nope"))


def test_family_of(pool: KeyPool) -> None:
    assert pool.family_of("ck-1") == "claude"
    assert pool.family_of("gk-1") == "gemini"
    assert pool.family_of("nope") is None


# ----------------------------------------------------------------------
# Env-var loading
# ----------------------------------------------------------------------

def test_from_env_parses_families_and_keys() -> None:
    env = {
        "KEYPOOL_CLAUDE_KEYS": "a, b ,c",
        "KEYPOOL_GPT_KEYS": "d",
        "KEYPOOL_EMPTYISH_KEYS": " , ,",
        "OTHERPREFIX_CLAUDE_KEYS": "should-be-ignored",
        "KEYPOOL_UNRELATED": "also-ignored",
    }
    pool = KeyPool.from_env(environ=env)
    snap = pool.snapshot()
    assert sorted(pool.families) == ["claude", "emptyish", "gpt"]
    assert [s.key for s in snap["claude"]] == ["a", "b", "c"]
    assert [s.key for s in snap["gpt"]] == ["d"]
    assert snap["emptyish"] == []


def test_from_env_custom_prefix() -> None:
    env = {"MYAPP_CLAUDE_KEYS": "x"}
    pool = KeyPool.from_env(prefix="MYAPP", environ=env)
    assert pool.families == ["claude"]


# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------

def test_health_check_benches_unhealthy_keys(pool: KeyPool, clock: FakeClock) -> None:
    async def checker(family: str, key: str) -> bool:
        return key != "ck-2"

    async def scenario():
        results = await pool.check_health(checker)
        assert results["claude"] == {"ck-1": True, "ck-2": False, "ck-3": True}
        assert results["gemini"] == {"gk-1": True}
        picks = set()
        for _ in range(3):
            clock.advance(1.0)
            picks.add(await pool.get_key("claude"))
        return picks

    picks = run(scenario())
    assert picks == {"ck-1", "ck-3"}


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------

def test_concurrent_get_key_is_safe(pool: KeyPool, clock: FakeClock) -> None:
    """Concurrent tasks all receive keys from the right family."""
    async def scenario():
        async def grab():
            clock.advance(0.001)
            return await pool.get_key("claude")

        results = await asyncio.gather(*[grab() for _ in range(30)])
        assert all(k.startswith("ck-") for k in results)
        # All three keys participate in rotation.
        assert set(results) == {"ck-1", "ck-2", "ck-3"}

    run(scenario())
