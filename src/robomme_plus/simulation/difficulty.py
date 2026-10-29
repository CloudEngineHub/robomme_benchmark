"""Utility helpers for validating and normalizing Robomme difficulty hints."""

from __future__ import annotations

from typing import Optional

# The whitelist is shared by all tasks; each task's own configs decide which tiers it supports.
NATIVE_DIFFICULTIES = ("easy", "medium", "hard")
# Internal scene presets used by the current public difficulty mapping.
NEWVALUE_DIFFICULTIES = ("xhard1", "xhard2", "xhard3", "xhard4")
ALL_NEWVALUE_TIERS = NEWVALUE_DIFFICULTIES
VALID_DIFFICULTIES = set(NATIVE_DIFFICULTIES) | set(ALL_NEWVALUE_TIERS)
_NEWVALUE_TIER = {name: i + 1 for i, name in enumerate(ALL_NEWVALUE_TIERS)}
NO_TIER_ENVS = ("MoveCube", "InsertPeg")


def is_newvalue_difficulty(value: Optional[str]) -> bool:
    """Family check: whether this episode's difficulty is a new-value tier; None and the three native tiers return False."""
    return isinstance(value, str) and value.strip().lower() in _NEWVALUE_TIER


def require_xhard4_only(difficulty: Optional[str], env_name: str) -> None:
    """Environments without a tier gradient (only InsertPeg and MoveCube call this) reject xhard1/2/3 and accept only xhard4, with no silent mapping."""
    if is_newvalue_difficulty(difficulty) and difficulty.strip().lower() != "xhard4":
        raise ValueError(
            f"{env_name} has no difficulty gradient and adds no tiers; only the new-value tier 'xhard4' is supported, got {difficulty!r}"
        )


def normalize_robomme_difficulty(value: Optional[str]) -> Optional[str]:
    """Return a canonical difficulty string or ``None`` if no override was provided."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(f"difficulty must be a string (got {type(value).__name__!r}).")

    normalized = value.strip().lower()
    if normalized not in VALID_DIFFICULTIES:
        raise ValueError(
            "Unsupported difficulty level. Available options: "
            f"{sorted(VALID_DIFFICULTIES)}."
        )

    return normalized
