"""Deterministic goal and motion settings for supplied episode seeds."""

from functools import lru_cache
from itertools import product

import numpy as np

TASKS = ("BinFill", "PickXtimes", "SwingXtimes", "StopCube")
RGB = ("red", "green", "blue")
# Goal counts of dynamic BinFill and easy StopCube.
COUNTS = {"BinFill": tuple(range(3, 7)), "StopCube": tuple(range(3, 9))}
STOP_COUNTS = {
    "easy": tuple(range(3, 9)),
    "medium": tuple(range(5, 11)),
    "hard": tuple(range(5, 11)),
}
# Easy BinFill/PickXtimes/SwingXtimes: upstream RoboMME's static hard scene with
# counts beyond upstream's maximum (bin 5, 5 repetitions, 3 swings).
EASY_COUNTS = {
    "BinFill": (6, 7, 8),
    "PickXtimes": (6, 7, 8),
    "SwingXtimes": (4, 5, 6, 7),
}
SHAPES = ("figure8", "ellipse", "circle", "line")
STOP_SHAPES = {
    "easy": SHAPES,
    "medium": SHAPES,
    "hard": (*SHAPES, "double_ellipse"),
}
SPEEDS = ("medium", "high", "ultra_high", "mixed_loops", "mixed_halves")
# BinFill conveyor speed (m/s) by dynamic difficulty level.
BELT_SPEEDS = {"medium": 0.050, "hard": 0.065}
# Dynamic BinFill levels: how many of red/green/blue the goal asks for (at least
# one cube of each); medium draws 2 or 3 with equal chance.
BIN_COLOR_COUNTS = {"medium": (2, 3), "hard": (3,)}
# PickXtimes levels: repetitions, target speed (multiple of a 12 s lap), and the
# range of ellipse sizes (multiple of the 13 x 7 cm semi-axes).
PICK_LEVELS = {
    "medium": {"counts": (5, 6), "speed": 1.5, "scale": (0.8, 0.95)},
    "hard": {"counts": (7, 8), "speed": 2.0, "scale": (1.0, 1.2)},
}
# SwingXtimes dynamic levels: cycles and the most dodges per target visit.
SWING_LEVELS = {
    "medium": {"counts": (3, 4), "max_evades": 2},
    "hard": {"counts": (4, 5, 6), "max_evades": 4},
}
STOP_DURATIONS = {
    "medium": [64, 52, 68, 56, 60, 48],
    "high": [44, 36, 46, 38, 42, 34],
    "ultra_high": [30, 24, 32, 26, 28, 22],
    # Passes alternate a high-speed lap and an ultra-high lap.
    "mixed_loops": [44, 24, 46, 26, 42, 22],
    # Each pass starts at the point farthest from the target: [steps to the
    # target at ultra-high speed, steps back to the farthest point at high speed].
    "mixed_halves": [[15, 22], [12, 18], [16, 23], [13, 19], [14, 21], [11, 17]],
}
# StopCube: each episode scales its path by a seeded factor in this range (the
# thin ellipse keeps its 3 cm width); pass durations scale with it, so each
# level's cube speed is unchanged.
STOP_SIZE_SCALE = (0.7, 1.3)
# Medium/hard vary the complete route layout, independently of goal/speed draws.
STOP_SIZE_SCALES = {"easy": STOP_SIZE_SCALE, "medium": (0.65, 1.45), "hard": (0.6, 1.6)}
# Medium/hard line length, measured end to end in meters. Scaling the pass
# duration by length / 0.30 preserves the existing high/ultra-high speeds.
STOP_LINE_LENGTH_RANGE = (0.45, 0.52)
# One touching-ellipse pass is 0.62019 m at base size, versus 0.48927 m for
# the original figure-eight. Match its mean high/ultra-high travel speeds.
STOP_DOUBLE_ELLIPSE_TIME_SCALE = 1.26758
STOP_TARGET_SCALES = {"easy": 1.0, "medium": 1.0, "hard": 2 / 3}
# StopCube levels: the cube speeds a seed may draw.
STOP_LEVELS = {
    "easy": ("medium",),
    "medium": ("high", "ultra_high"),
    "hard": ("mixed_loops", "mixed_halves"),
}
# Speeds whose press must reach the button quickly.
FAST_STOP_SPEEDS = ("ultra_high", "mixed_loops", "mixed_halves")
PROFILES_BY_TASK = {
    "BinFill": "binfill_conveyor",
    "PickXtimes": "pickxtimes_moving",
    "SwingXtimes": "swingxtimes_reactive",
    "StopCube": "stopcube_random",
}


def static_level(task, level):
    """True for the static easy level (upstream scene, larger counts)."""
    return level == "easy" and task in EASY_COUNTS


def swing_start_side(seed):
    """Choose the first target with equal odds, independently of other draws."""
    rng = np.random.default_rng([int(seed), 67867967])
    return str(rng.choice(("right", "left")))


def goal_counts(task, level="hard"):
    """Goal counts of a task at a difficulty level."""
    if static_level(task, level):
        return EASY_COUNTS[task]
    if task == "PickXtimes":
        return PICK_LEVELS[level]["counts"]
    if task == "SwingXtimes":
        return SWING_LEVELS[level]["counts"]
    if task == "StopCube":
        return STOP_COUNTS[level]
    return COUNTS[task]


@lru_cache(maxsize=32)
def bin_combinations(total, colors=None):
    """Red/green/blue goal counts (0-4 each) summing to ``total``; with ``colors``,
    only those asking for exactly that many colors."""
    values = [v for v in product(range(5), repeat=3) if sum(v) == total]
    return (
        values
        if colors is None
        else [v for v in values if sum(n > 0 for n in v) == colors]
    )


def sample_variation(task, seed, level="hard"):
    """Reconstruct a goal and motion selection from the episode seed (and level)."""
    if task not in TASKS or int(seed) != seed or seed < 0:
        raise ValueError("A counting task and nonnegative integer seed are required")
    rng = np.random.default_rng([int(seed), 982451653, TASKS.index(task)])
    count = int(rng.choice(goal_counts(task, level)))
    result = {"count": count}
    if task == "SwingXtimes":
        result["start_side"] = swing_start_side(seed)
    if static_level(task, level):
        return result  # the native scene samples colors and layout from the same seed
    if task == "BinFill":
        colors = int(rng.choice(BIN_COLOR_COUNTS[level]))
        combinations = bin_combinations(count, colors)
        values = combinations[int(rng.integers(len(combinations)))]
        result["rgb_counts"] = dict(zip(RGB, values))
    elif task in ("PickXtimes", "SwingXtimes"):
        result["target_color"] = str(rng.choice(RGB))
    else:
        result.update(
            shape=str(rng.choice(STOP_SHAPES[level])),
            speed=str(rng.choice(STOP_LEVELS[level])),
        )
    return result

