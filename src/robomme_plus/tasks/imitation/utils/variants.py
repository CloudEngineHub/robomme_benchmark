"""Seeded geometry variants and levels of the imitation tasks (CPU-only).

MoveCube has geometry variants, not difficulty levels: an omitted variant is
sampled from an independent seed stream. InsertPeg has no variants: difficulty
sets its hole-width range and the seed samples the scene. PatternLock and
RouteStick levels control path length, the execution stick, and the
demonstration board.
"""

VARIANTS = {
    "MoveCube": {
        "hook": "move_hook",
        "poke": "move_poke",
        "scoop": "move_scoop",
        "ball": "move_ball",
        "cue": "move_cue",
        "scoop_left": "move_scoop_left",
        "scoop_right": "move_scoop_right",
    },
}
# Seed-only tasks: one geometry profile, no variants and no levels.
SEED_ONLY_PROFILES = {"InsertPeg": "insert_peg"}
# Tracing tasks: one geometry profile; the level sets path length, the execution
# stick and the board. Upstream RoboMME hard traces 4-8 PatternLock nodes and
# 4-7 RouteStick segments with the full stick and no board.
# Every level uses a full demonstration stick. Easy keeps it full length;
# medium/hard shorten it for execution. PatternLock v6 keeps 50-70% in
# medium and 35-60% in hard; v5 keeps 40-60% in both. Older versions and
# RouteStick keep 25-50%.
# RouteStick keeps its demonstration visible at every level.
TRACE_PROFILES = {"PatternLock": "pattern_trace", "RouteStick": "route_trace"}
_LONG = {
    "PatternLock": {"path_node_range": [9, 12]},
    "RouteStick": {"segment_count_range": [6, 8]},
}
_UPSTREAM_HARD = {
    "PatternLock": {"path_node_range": [4, 8]},
    "RouteStick": {"segment_count_range": [4, 7]},
}
TRACE_LEVELS = {
    task: {
        "easy": dict(_LONG[task], board=False),
        "medium": dict(_UPSTREAM_HARD[task], board=False),
        "hard": dict(_LONG[task], board=True),
    }
    for task in TRACE_PROFILES
}
TRACE_LEVELS["PatternLock"]["easy"].update(
    shorten_stick=False, rotate_grid=False, route_mode="adjacent",
)
TRACE_LEVELS["RouteStick"]["easy"].update(shorten_stick=False, segment_count_range=[8, 13])
for _level in ("medium", "hard"):
    TRACE_LEVELS["PatternLock"][_level].update(
        shorten_stick=True, rotate_grid=True, route_mode="any_grid",
    )
    TRACE_LEVELS["RouteStick"][_level].update(shorten_stick=True)
TRACE_LEVELS["RouteStick"]["hard"].update(board=False)
VARIANT_SEED_OFFSET = 610136
TRACE_STICK_FRACTION_RANGE = (0.25, 0.50)
PATTERN_STICK_FRACTION_RANGE_V5 = (0.40, 0.60)
PATTERN_STICK_FRACTION_RANGES_V6 = {'medium': (0.50, 0.70), 'hard': (0.35, 0.60)}

# Version 4 keeps earlier recorded scenes replayable while introducing the
# visible-board, progressively longer pattern and level-specific tool change.
PATTERN_LEVELS_V4 = {
    "easy": dict(path_node_range=[4, 5], board=False, shorten_stick=False,
                 rotate_grid=False, shift_grid=False, route_mode="adjacent"),
    "medium": dict(path_node_range=[6, 8], board=False, shorten_stick=True,
                   rotate_grid=True, shift_grid=True, route_mode="any_grid"),
    "hard": dict(path_node_range=[9, 12], board=False, shorten_stick=True,
                 rotate_grid=True, shift_grid=True, route_mode="any_grid",
                 grid_shift_m=0.15, grid_rotation_range_deg=[10.0, 15.0]),
}
PATTERN_TILT_RANGES_DEG = {"easy": (0.0, 0.0), "medium": (10.0, 20.0), "hard": (20.0, 30.0)}


def validate_level_stick_angle(value, level):
    import math
    from numbers import Real

    low, high = PATTERN_TILT_RANGES_DEG[level]
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f'{level} execution_stick_angle_deg must be within {low}–{high}')
    return float(value)


def pattern_level_tilt(seed, level, angle_deg=None):
    """Uniform continuous polar angle and full-circle, independent mount azimuth."""
    import numpy as np

    angle = (float(np.random.default_rng([int(seed), 610143]).uniform(*PATTERN_TILT_RANGES_DEG[level]))
             if angle_deg is None else validate_level_stick_angle(angle_deg, level))
    return [angle, tracing_stick_tilt(seed)[1]]


def tracing_stick_fraction(task, seed, *, stick_change_version=1, level=None):
    """Sample the shortened asset; easy keeps using its full demonstration stick."""
    import numpy as np

    task_index = tuple(TRACE_PROFILES).index(task)
    bounds = (PATTERN_STICK_FRACTION_RANGE_V5
              if task == 'PatternLock' and stick_change_version == 5
              else TRACE_STICK_FRACTION_RANGE)
    if task == 'PatternLock' and stick_change_version == 6:
        if level not in ('easy', 'medium', 'hard'):
            raise ValueError('PatternLock v6 stick length requires an easy, medium or hard level')
        # Easy never activates this asset; retain its v5 geometry for registration.
        bounds = PATTERN_STICK_FRACTION_RANGES_V6.get(level, PATTERN_STICK_FRACTION_RANGE_V5)
    return float(np.random.default_rng([int(seed), 610139, task_index]).uniform(
        *bounds
    ))


def tracing_stick_tilt(seed):
    """Sample PatternLock's legacy mount within a 30-degree cone."""
    import numpy as np

    rng = np.random.default_rng([int(seed), 610140])
    cosine = rng.uniform(np.nextafter(np.cos(np.radians(30)), 1), 1)
    angle = float(np.degrees(np.arccos(cosine)))
    return [angle, float(rng.uniform(-180, 180))]


PATTERN_STICK_ANGLES_DEG = (10.0, 20.0, 30.0)


def validate_pattern_stick_angle(value):
    from numbers import Real

    if isinstance(value, bool) or not isinstance(value, Real) or value not in PATTERN_STICK_ANGLES_DEG:
        raise ValueError('execution_stick_angle_deg must be 10, 20 or 30')
    return float(value)


def pattern_stick_tilt(seed, angle_deg=None):
    """Discrete execution tilt with an independent seeded mounting direction.

    Version 3 uses 10, 20 or 30 degrees; an explicit angle pins a review case.
    The demonstration uses a separate straight cylinder regardless of this pose.
    """
    import numpy as np

    angle = (float(np.random.default_rng([int(seed), 610141]).choice(PATTERN_STICK_ANGLES_DEG))
             if angle_deg is None else validate_pattern_stick_angle(angle_deg))
    return [angle, tracing_stick_tilt(seed)[1]]


def imitation_variant(task, seed, variant=None):
    """Return ``(variant, profile)``; sample the variant from the seed when omitted.

    Tasks without variants return ``(None, profile)`` and reject any ``variant``.
    """
    import numpy as np

    if task in SEED_ONLY_PROFILES:
        if variant is not None:
            raise ValueError(
                f"{task} has no variants (got {variant!r}); use seed and difficulty"
            )
        return None, SEED_ONLY_PROFILES[task]
    choices = VARIANTS[task]
    if variant is None:
        # Left/right remain explicit compatibility aliases; the default uses
        # one equally weighted choice per demonstrated tool method.
        from robomme_plus.tasks.imitation.move_cube_support.config import method_selection
        _, variant = method_selection(seed)
    if variant not in choices:
        raise ValueError(
            f"Unknown variant {variant!r} for {task}; choose from {tuple(choices)}"
        )
    return variant, choices[variant]
