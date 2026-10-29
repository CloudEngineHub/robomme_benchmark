"""Seeded simultaneous or staggered cues with an inclusive final visible step."""

import numpy as np


HIGHLIGHT_MODES = ("simultaneous", "staggered")


def resolve_highlight_mode(seed, mode="random"):
    """Use an independent stream so the mode cannot change objects or goals."""
    if mode == "random":
        return HIGHLIGHT_MODES[int(np.random.default_rng([int(seed), 104731]).integers(2))]
    if mode not in HIGHLIGHT_MODES:
        raise ValueError(f"highlight_mode must be 'random' or one of {HIGHLIGHT_MODES}")
    return mode


def highlight_schedule(
    seed, count, *, start_step=10, end_step=119, duration_steps=45, mode="staggered"
):
    """Simultaneous cues share a window; staggered cues have distinct starts/ends."""
    mode = resolve_highlight_mode(seed, mode)
    low, high = (duration_steps, duration_steps) if np.isscalar(duration_steps) else duration_steps
    if not all(isinstance(value, (int, np.integer)) for value in (low, high)):
        raise ValueError("Highlight durations must be integer frame counts")
    if count < 1 or low < 1 or high < low or start_step < 0:
        raise ValueError("Positive target count/duration and a nonnegative start are required")
    last_start = end_step - high + 1
    if last_start < start_step:
        raise ValueError("The highlight duration exceeds the available window")
    if mode == "simultaneous":
        # Like the original cue, all targets turn on at the start together and
        # turn off together. Keep the current level's duration range.
        duration = int(np.random.default_rng([int(seed), 104730]).integers(low, high + 1))
        return [
            {"start_step": start_step, "end_step": start_step + duration - 1}
            for _ in range(count)
        ]
    if last_start - start_step + 1 < count:
        raise ValueError("The highlight window cannot fit distinct starts for every target")
    rng = np.random.default_rng([int(seed), 104729])
    # Draw each target's duration once on a separate stream. Resampling only
    # starts below avoids favoring durations that happen to have fewer ties.
    durations = (
        np.full(count, low, dtype=int) if low == high
        else np.random.default_rng([int(seed), 104730]).integers(low, high + 1, count)
    )
    span = max(high, count - 1)
    if count > 1 and last_start - start_step < span:
        raise ValueError("The highlight window cannot avoid all-target overlap")
    for _ in range(1000):
        if count == 1:
            starts = [int(rng.integers(start_step, last_start + 1))]
        else:
            # Separate two windows by the maximum duration. Shortening any
            # individual window cannot create common all-target overlap.
            first = int(rng.integers(start_step, last_start - span + 1))
            last = int(rng.integers(first + span, last_start + 1))
            middle = rng.choice(np.arange(first + 1, last), count - 2, replace=False)
            starts = [first, last, *(int(x) for x in middle)]
            rng.shuffle(starts)
        ends = [start + int(duration) - 1 for start, duration in zip(starts, durations)]
        if len(set(ends)) == count:
            return [
                {"start_step": start, "end_step": end}
                for start, end in zip(starts, ends)
            ]
    raise ValueError("Could not sample distinct highlight end frames")
