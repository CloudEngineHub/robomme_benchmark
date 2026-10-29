"""Bound private calibration effort per layout, never policy execution."""
from dataclasses import dataclass, field
import math
import time


MAX_LAYOUT_ROBOT_TRIALS = 6
MAX_LAYOUT_SEARCH_SECONDS = 180.0


class LayoutSearchExhausted(RuntimeError):
    """The caller should sample another layout, preserving the phase quota."""


@dataclass
class ReferenceSearchBudget:
    max_trials: int = MAX_LAYOUT_ROBOT_TRIALS
    max_seconds: float = MAX_LAYOUT_SEARCH_SECONDS
    trials: int = field(default=0, init=False)
    started: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self):
        if not isinstance(self.max_trials, int) or isinstance(self.max_trials, bool) or self.max_trials < 1:
            raise ValueError("Layout search trial limit must be a positive integer")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Layout search time limit must be positive and finite")

    def check_time(self):
        elapsed = time.monotonic() - self.started
        if elapsed >= self.max_seconds:
            raise LayoutSearchExhausted(
                f"Layout search budget exhausted after {elapsed:.1f}s "
                f"(limit {self.max_seconds:g}s, {self.trials} robot trials); resample layout"
            )

    def start_trial(self):
        self.check_time()
        if self.trials >= self.max_trials:
            raise LayoutSearchExhausted(
                f"Layout search budget exhausted after {self.trials} robot trials "
                f"(limit {self.max_trials}); resample layout"
            )
        self.trials += 1
