"""VideoPlace difficulty levels (CPU-only)."""

import numpy as np

LEVELS = {
    "easy": {"demo_cube_count": 1, "swap_start_interval_steps": 80, "targets": 4},
    "medium": {"demo_cube_count": 1, "swap_start_interval_steps": 40, "targets": 6},
    "hard": {"demo_cube_count": 2, "swap_start_interval_steps": 40, "targets": (7, 8)},
}


def level_parameters(level, seed):
    """Resolve the target count without consuming layout or cube-history draws."""
    config = dict(LEVELS[level])
    if isinstance(config["targets"], tuple):
        low, high = config["targets"]
        config["targets"] = int(np.random.default_rng([int(seed), 872434]).integers(low, high + 1))
    return config


# Native generator tiers used for goal and cube-history sampling; internal only.
# VideoPlaceButton's xhard3 needs five targets, so easy (four rings) uses xhard1.
NATIVE_TIERS = {
    "VideoPlaceOrder": {"easy": "xhard1", "medium": "xhard1", "hard": "xhard1"},
    "VideoPlaceButton": {"easy": "xhard1", "medium": "xhard3", "hard": "xhard3"},
}
