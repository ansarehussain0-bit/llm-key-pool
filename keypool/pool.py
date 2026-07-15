"""Core key-pool implementation.

This module is intentionally dependency-free: it only uses the Python
standard library so it can be dropped into any project.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from .exceptions import (
    DuplicateKeyError,
    NoKeysAvailableError,
    UnknownFamilyError,
    UnknownKeyError,
)

__all__ = ["CooldownPolicy", "KeyState", "KeyPool"]

# Signature of an optional health-check callback:
# ``async def check(family: str, key: str) -> bool``
HealthChecker = Callable[[str, str], Awaitable[bool]]

_ENV_KEY_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<family>[A-Z0-9]+)_KEYS$")


@dataclass(frozen=True)
class CooldownPolicy:
    """How long a key is benched after different kinds of failures.

    All durations are in seconds.

    Attributes:
        rate_limit_cooldown: Applied on HTTP 429 (rate limited). Short,
            because rate limits clear quickly.
        auth_cooldown_base: Base cooldown applied on HTTP 401/403
            (invalid / revoked / unauthorized key). Doubles on every
            consecutive auth failure of the same key.
        auth_cooldown_max: Upper bound for the auth-failure backoff.
        generic_cooldown: Applied on any other error status (e.g. 500s).
    """

    rate_limit_cooldown: float = 60.0
    auth_cooldown_base: float = 300.0
    auth_cooldown_max: float = 3600.0
    generic_cooldown: float = 30.0


@dataclass
class KeyState:
    """Mutable bookkeeping for a single API key.

    Attributes:
        key: The API key string itself.
        family: The family the key belongs to (e.g. ``"claude"``).
        last_used_at: Monotonic timestamp of the last time the key was
            handed out, or ``None`` if never used.
        cooldown_until: Monotonic timestamp until which the key is benched,
            or ``None`` if the key is available.
        failure_count: Total number of failures recorded for this key.
        auth_failure_streak: Consecutive 401/403 failures. Drives the
            exponential backoff and the ``likely_invalid`` flag. Reset on
            success.
        likely_invalid: ``True`` once the key has failed with 401/403.
            Such keys are only handed out when no healthy key is available.
    """

    key: str
    family: str
    last_used_at: Optional[float] = None
    cooldown_until: Optional[float] = None
    failure_count: int = 0
    auth_failure_streak: int = 0
    likely_invalid: bool = False

    def is_available(self, now: float) -> bool:
        """Return ``True`` if the key is not currently cooling down."""
        return self.cooldown_until is None or self.cooldown_until <= now


class KeyPool:
    """Family-aware pool of API keys with cooldown and LRU rotation.

    Keys are grouped into *families* (e.g. ``"claude"``, ``"gemini"``,
    ``"gpt"``). A key registered under one family can never be returned
    for another family — this guard is enforced at registration time by
    rejecting duplicate keys across families, and at lookup time by only
    searching the requested family.

    Selection strategy: among available keys of a family, the
    least-recently-used key is returned, which yields round-robin
    behaviour under steady load. Keys flagged as likely invalid
    (previous 401/403) are deprioritised and only used as a last resort.

    All public methods are coroutine functions guarded by an
    :class:`asyncio.Lock`, making the pool safe to share across
    concurrent tasks.

    Example:
        >>> pool = KeyPool()
        >>> pool_setup = pool.register_family("claude", ["sk-a", "sk-b"])
        >>> # inside async code:
        >>> # key = await pool.get_key("claude")
        >>> # await pool.mark_failure(key, status_code=429)
    """

    def __init__(
        self,
        cooldown_policy: Optional[CooldownPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialise an empty pool.

        Args:
            cooldown_policy: Cooldown durations to apply on failures.
                Defaults to :class:`CooldownPolicy` defaults.
            clock: Monotonic time source. Injectable for testing.
        """
        self._policy = cooldown_policy or CooldownPolicy()
        self._clock = clock
        self._lock = asyncio.Lock()
        # family name -> ordered list of key states
        self._families: Dict[str, List[KeyState]] = {}
        # key string -> its state (also enforces one-family-per-key)
        self._by_key: Dict[str, KeyState] = {}

    # ------------------------------------------------------------------
    # Registration (synchronous: done once at startup, before serving)
    # ------------------------------------------------------------------

    def register_family(self, family: str, keys: Optional[List[str]] = None) -> None:
        """Register a family, optionally with an initial list of keys.

        Args:
            family: Family name; normalised to lowercase.
            keys: Optional initial keys. Blank entries are ignored.

        Raises:
            DuplicateKeyError: If any key is already registered under a
                different family.
        """
        family = family.strip().lower()
        self._families.setdefault(family, [])
        for key in keys or []:
            self.add_key(family, key)

    def add_key(self, family: str, key: str) -> None:
        """Add a single key to an already-registered family.

        Args:
            family: Family name (must be registered first, or will be
                created implicitly).
            key: The API key. Whitespace is stripped; empty keys and
                exact duplicates within the same family are ignored.

        Raises:
            DuplicateKeyError: If the key already belongs to another family.
        """
        family = family.strip().lower()
        key = key.strip()
        if not key:
            return
        existing = self._by_key.get(key)
        if existing is not None:
            if existing.family != family:
                raise DuplicateKeyError(existing.family, family)
            return  # same key, same family: idempotent
        state = KeyState(key=key, family=family)
        self._families.setdefault(family, []).append(state)
        self._by_key[key] = state

    @classmethod
    def from_env(
        cls,
        prefix: str = "KEYPOOL",
        environ: Optional[Dict[str, str]] = None,
        cooldown_policy: Optional[CooldownPolicy] = None,
    ) -> "KeyPool":
        """Build a pool from environment variables.

        Scans the environment for variables matching
        ``<PREFIX>_<FAMILY>_KEYS`` whose value is a comma-separated list
        of keys, e.g.::

            KEYPOOL_CLAUDE_KEYS=sk-ant-1,sk-ant-2
            KEYPOOL_GEMINI_KEYS=AIza-1
            KEYPOOL_GPT_KEYS=sk-1,sk-2,sk-3

        Args:
            prefix: Env-var prefix, without the trailing underscore.
            environ: Mapping to read from; defaults to ``os.environ``.
                Injectable for testing.
            cooldown_policy: Optional cooldown configuration.

        Returns:
            A pool with one family per matching variable.
        """
        env = os.environ if environ is None else environ
        pool = cls(cooldown_policy=cooldown_policy)
        for name, value in env.items():
            match = _ENV_KEY_PATTERN.match(name)
            if match is None or match.group("prefix") != prefix:
                continue
            family = match.group("family").lower()
            keys = [k.strip() for k in value.split(",") if k.strip()]
            pool.register_family(family, keys)
        return pool

    # ------------------------------------------------------------------
    # Runtime API (async-safe)
    # ------------------------------------------------------------------

    async def get_key(self, family: str) -> str:
        """Return the best available key for *family*.

        Preference order:
            1. Available keys never flagged as likely-invalid, LRU first.
            2. Available likely-invalid keys (last resort), LRU first.

        The returned key's ``last_used_at`` is updated, so repeated calls
        rotate through the family round-robin style.

        Args:
            family: The family to draw a key from.

        Returns:
            The API key string.

        Raises:
            UnknownFamilyError: If the family was never registered.
            NoKeysAvailableError: If the family is empty or every key is
                cooling down. ``retry_after`` on the exception says when
                to try again.
        """
        family = family.strip().lower()
        async with self._lock:
            states = self._families.get(family)
            if states is None:
                raise UnknownFamilyError(family)
            if not states:
                raise NoKeysAvailableError(family, retry_after=None)

            now = self._clock()
            available = [s for s in states if s.is_available(now)]
            if not available:
                soonest = min(s.cooldown_until or now for s in states)
                raise NoKeysAvailableError(
                    family, retry_after=max(0.0, soonest - now)
                )

            def lru_order(state: KeyState) -> float:
                # Never-used keys sort first.
                return state.last_used_at if state.last_used_at is not None else float("-inf")

            healthy = [s for s in available if not s.likely_invalid]
            candidates = healthy or available
            chosen = min(candidates, key=lru_order)
            chosen.last_used_at = now
            return chosen.key

    async def mark_failure(self, key: str, status_code: int) -> None:
        """Record a failed request for *key* and apply a cooldown.

        Cooldowns by status code:
            * 429 — short rate-limit cooldown.
            * 401 / 403 — key flagged ``likely_invalid`` and benched with
              exponential backoff (base doubles per consecutive auth
              failure, capped at the policy maximum).
            * anything else — short generic cooldown.

        Args:
            key: The key that failed.
            status_code: HTTP status code returned by the provider.

        Raises:
            UnknownKeyError: If the key is not in the pool.
        """
        async with self._lock:
            state = self._by_key.get(key)
            if state is None:
                raise UnknownKeyError()

            now = self._clock()
            state.failure_count += 1

            if status_code == 429:
                cooldown = self._policy.rate_limit_cooldown
            elif status_code in (401, 403):
                state.auth_failure_streak += 1
                state.likely_invalid = True
                cooldown = min(
                    self._policy.auth_cooldown_base
                    * (2 ** (state.auth_failure_streak - 1)),
                    self._policy.auth_cooldown_max,
                )
            else:
                cooldown = self._policy.generic_cooldown

            state.cooldown_until = now + cooldown

    async def mark_success(self, key: str) -> None:
        """Record a successful request for *key*, clearing failure state.

        Resets the cooldown, the auth-failure streak and the
        ``likely_invalid`` flag (a key that just worked is evidently
        valid). The total ``failure_count`` is kept for observability.

        Args:
            key: The key that succeeded.

        Raises:
            UnknownKeyError: If the key is not in the pool.
        """
        async with self._lock:
            state = self._by_key.get(key)
            if state is None:
                raise UnknownKeyError()
            state.cooldown_until = None
            state.auth_failure_streak = 0
            state.likely_invalid = False

    async def check_health(self, checker: HealthChecker) -> Dict[str, Dict[str, bool]]:
        """Run *checker* against every key in the pool.

        Intended for optional startup validation. Keys for which the
        checker returns ``False`` (or raises) are flagged
        ``likely_invalid`` and benched with the auth cooldown, exactly as
        if they had returned 401.

        Args:
            checker: ``async def checker(family, key) -> bool``. Should
                return ``True`` if the key is usable.

        Returns:
            ``{family: {key: healthy}}`` for all registered keys.
        """
        # Snapshot outside the lock so the checker's network calls do not
        # block the pool.
        async with self._lock:
            snapshot = [(s.family, s.key) for states in self._families.values() for s in states]

        results: Dict[str, Dict[str, bool]] = {}
        for family, key in snapshot:
            try:
                healthy = await checker(family, key)
            except Exception:
                healthy = False
            results.setdefault(family, {})[key] = healthy
            if not healthy:
                await self.mark_failure(key, status_code=401)
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def families(self) -> List[str]:
        """Names of all registered families."""
        return list(self._families)

    def family_of(self, key: str) -> Optional[str]:
        """Return the family a key belongs to, or ``None`` if unknown."""
        state = self._by_key.get(key)
        return state.family if state else None

    def snapshot(self) -> Dict[str, List[KeyState]]:
        """Return a shallow copy of the pool state, for debugging/metrics.

        Note: the :class:`KeyState` objects are the live ones; treat them
        as read-only.
        """
        return {family: list(states) for family, states in self._families.items()}
