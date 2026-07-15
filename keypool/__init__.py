"""llm-key-pool: provider-agnostic, family-aware API key pool for LLM proxies.

Public API::

    from keypool import KeyPool, CooldownPolicy
    from keypool import NoKeysAvailableError, UnknownFamilyError
"""

from .exceptions import (
    DuplicateKeyError,
    KeyPoolError,
    NoKeysAvailableError,
    UnknownFamilyError,
    UnknownKeyError,
)
from .pool import CooldownPolicy, KeyPool, KeyState

__version__ = "0.1.0"

__all__ = [
    "KeyPool",
    "KeyState",
    "CooldownPolicy",
    "KeyPoolError",
    "UnknownFamilyError",
    "UnknownKeyError",
    "DuplicateKeyError",
    "NoKeysAvailableError",
    "__version__",
]
