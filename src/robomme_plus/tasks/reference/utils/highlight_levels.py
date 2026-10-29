"""Three public levels backed exclusively by the native extra tiers."""

import hashlib

NATIVE_TIERS = {
    "easy": ("xhard1",),
    "medium": ("xhard2",),
    "hard": ("xhard3", "xhard4"),
}
PUBLIC_LEVELS = {tier: level for level, tiers in NATIVE_TIERS.items() for tier in tiers}
DURATION_STEPS = {"easy": 45, "medium": (30, 45), "hard": (16, 45)}
CUBE_COUNTS = {"easy": (9, 10), "medium": (11, 13), "hard": (14, 17)}
PREVIOUS_CUBE_COUNTS = {"easy": (7, 8), "medium": (9, 10), "hard": (11, 13)}
COMPACT_CUBE_BOUNDS = ((-0.38, 0.12), (-0.32, 0.32))
DENSE_CUBE_BOUNDS = ((-0.35, 0.09), (-0.28, 0.28))
DENSE_CUBE_GAP_FACTOR = 1


def cube_count_range(level, layout_version=4):
    """Preserve old recorded layouts while new episodes use the larger pools."""
    if type(layout_version) is not int or layout_version not in (1, 2, 3, 4):
        raise ValueError("PickHighlight layout_version must be 1, 2, 3 or 4")
    counts = PREVIOUS_CUBE_COUNTS if layout_version == 1 else CUBE_COUNTS
    return counts[level]


def sample_highlight_ids(seed, total, count):
    """Uniform subset of the finished pool, independent of layout/color draws."""
    import numpy as np

    if not 1 <= count <= total:
        raise ValueError("Highlighted count must be between 1 and the total cube count")
    rng = np.random.default_rng([int(seed), 104732])
    return rng.choice(total, count, replace=False).tolist()


def validate_color_mode(mode):
    if mode not in ("independent", "similar"):
        raise ValueError("PickHighlight color_mode must be 'independent' or 'similar'")
    return mode


def sample_similar_colors(seed, total):
    """Balanced, shuffled shade families, independent of placement and goals."""
    import colorsys
    import numpy as np

    rng = np.random.default_rng([int(seed), 104735])
    count = int(rng.integers(2, 4))
    center = float(rng.random())
    hues = (center + np.linspace(-.07, .07, count)) % 1
    saturation, value = rng.uniform(.60, .82), rng.uniform(.62, .85)
    families = np.arange(total) % count
    rng.shuffle(families)
    hsv = np.column_stack((
        (hues[families] + rng.uniform(-.012, .012, total)) % 1,
        saturation + rng.uniform(-.04, .04, total),
        value + rng.uniform(-.035, .035, total),
    ))
    rgba = [list(colorsys.hsv_to_rgb(*color)) + [1.] for color in hsv]
    return rgba, dict(mode="similar", family_count=count, family_hues=hues.tolist(),
                      family_ids=families.tolist(), hue_jitter_degrees=4.32,
                      saturation_jitter=.04, value_jitter=.035)


def native_difficulty(level, seed, recorded_tier=None):
    """Resolve a seeded tier, or preserve the tier of a verified test episode."""
    if level not in NATIVE_TIERS:
        raise ValueError(
            f"PickHighlight difficulty must be one of {tuple(NATIVE_TIERS)}"
        )
    tiers = NATIVE_TIERS[level]
    if recorded_tier is not None:
        if recorded_tier not in tiers:
            raise ValueError(
                f"Recorded PickHighlight tier {recorded_tier!r} does not belong to {level!r}"
            )
        return recorded_tier
    digest = hashlib.sha256(f"pick-highlight-tier-v1/{int(seed)}".encode()).digest()
    return tiers[int.from_bytes(digest[:8], "little") % len(tiers)]
