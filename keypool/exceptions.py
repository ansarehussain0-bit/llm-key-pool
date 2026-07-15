"""Exceptions raised by the key pool."""

from __future__ import annotations


class KeyPoolError(Exception):
    """Base class for all key-pool errors."""


class UnknownFamilyError(KeyPoolError):
    """Raised when a family has not been registered with the pool."""

    def __init__(self, family: str) -> None:
        self.family = family
        super().__init__(f"Unknown key family: {family!r}")


class UnknownKeyError(KeyPoolError):
    """Raised when a key is not present in the pool."""

    def __init__(self) -> None:
        super().__init__("Key is not registered in the pool")


class DuplicateKeyError(KeyPoolError):
    """Raised when the same key is added to more than one family.

    A key belongs to exactly one family. This is the hard guard that
    prevents cross-family key usage (e.g. a ``claude`` key being handed
    out for a ``gemini`` request).
    """

    def __init__(self, existing_family: str, new_family: str) -> None:
        self.existing_family = existing_family
        self.new_family = new_family
        super().__init__(
            f"Key already registered under family {existing_family!r}; "
            f"cannot also register it under {new_family!r}"
        )


class NoKeysAvailableError(KeyPoolError):
    """Raised when every key in a family is cooling down (or the family is empty).

    Attributes:
        family: The family that was requested.
        retry_after: Seconds until the earliest key becomes available again,
            or ``None`` if the family has no keys at all.
    """

    def __init__(self, family: str, retry_after: "float | None" = None) -> None:
        self.family = family
        self.retry_after = retry_after
        if retry_after is None:
            msg = f"Family {family!r} has no keys configured"
        else:
            msg = (
                f"All keys in family {family!r} are cooling down; "
                f"retry in {retry_after:.1f}s"
            )
        super().__init__(msg)
