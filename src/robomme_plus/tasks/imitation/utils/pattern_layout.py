"""Seeded grid translation, rotation and optional spacing change (CPU only)."""

import math
from numbers import Real
import numpy as np


LAYOUT_VERSION = "one_cell_shift_rotation_v1"
SCALED_LAYOUT_VERSION = "one_cell_shift_rotation_scale_v2"
LEVEL_LAYOUT_VERSION = "level_shift_rotation_scale_v3"
GRID_SPACING_SCALES = (0.8, 1.0, 1.2, 1.4)
LEVEL_GRID_SPACING_SCALES = {
    "easy": (1.0,), "medium": (0.9, 1.1), "hard": (0.9, 1.1, 1.2, 1.3),
}


def validate_grid_scale(value, level=None):
    choices = (LEVEL_GRID_SPACING_SCALES[level] if level is not None else
               tuple(sorted(set(GRID_SPACING_SCALES + (0.9, 1.1, 1.3)))))
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value not in choices):
        raise ValueError(f"execution_grid_scale must be one of {choices}")
    return float(value)


def grid_spacing_scale(seed, level=None):
    """Sample execution/demo spacing independently of the route and grid pose."""
    choices = GRID_SPACING_SCALES if level is None else LEVEL_GRID_SPACING_SCALES[level]
    return float(np.random.default_rng([int(seed), 618005]).choice(choices))


def global_path(seed, grid_size, count):
    """Sample distinct endpoints anywhere on the grid, with a non-neighbor move."""
    if not 2 <= count <= grid_size ** 2 or grid_size < 3:
        raise ValueError("Global paths need a grid of at least 3×3 and 2–N² nodes")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 618004]))
    while True:
        nodes = rng.permutation(grid_size ** 2)[:count].tolist()
        if any(max(abs(a // grid_size - b // grid_size),
                   abs(a % grid_size - b % grid_size)) > 1
               for a, b in zip(nodes, nodes[1:])):
            return nodes


def grid_subgoal(start_node, end_node, grid_size):
    """Describe node identities with 1-based coordinates, independent of pose."""
    def point(node):
        row, column = divmod(node, grid_size)
        return f"grid ({row + 1}, {column + 1})"

    movement = (f"move to {point(end_node)}" if start_node is None else
                f"move from {point(start_node)} to {point(end_node)}")
    return f"{movement} on the {grid_size}×{grid_size} grid"


def grid_layouts(seed, grid_size, spacing, midpoint, rotate=True, *, execution_scale=1.0, shift_grid=True,
                 shift_distance_m=None, rotation_range_deg=(5.0, 10.0)):
    """Translate, rotate, and optionally resize the execution grid.

    The independent seed stream does not consume the route sampler's draws.
    Directions use the robot frame: forward is +x and left is +y.
    A larger execution shift preserves the original demonstration positions.
    """
    execution_scale = validate_grid_scale(execution_scale)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 618003]))
    directions = (("left", (0, 1)), ("right", (0, -1)),
                  ("forward", (1, 0)), ("backward", (-1, 0)))
    direction, axis = directions[int(rng.integers(4))]
    demo_shift = np.asarray(axis, dtype=float) * spacing
    shift = np.asarray(axis, dtype=float) * (spacing if shift_distance_m is None else shift_distance_m)
    if not shift_grid:
        direction, shift, demo_shift = "none", np.zeros(2), np.zeros(2)
    yaw_deg = float(rng.choice([-1, 1]) * rng.uniform(*rotation_range_deg))
    if not rotate:
        yaw_deg = 0.0
    local = np.array([[(row - (grid_size - 1) / 2) * spacing,
                       (col - (grid_size - 1) / 2) * spacing]
                      for row in range(grid_size) for col in range(grid_size)])
    layouts = {}
    for phase in ("demonstration", "execution"):
        execution = phase == "execution"
        scale = execution_scale if execution else 1.0
        center = np.asarray(midpoint, dtype=float) + (0.5 if execution else -0.5) * demo_shift
        if execution:
            center += shift - demo_shift
        angle = np.radians(yaw_deg if execution else 0.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        layouts[phase] = {
            "version": (LEVEL_LAYOUT_VERSION if shift_distance_m is not None or tuple(rotation_range_deg) != (5.0, 10.0)
                        else SCALED_LAYOUT_VERSION if execution_scale != 1 else LAYOUT_VERSION),
            "phase": phase,
            "direction": direction if execution else "none",
            "spacing_m": float(spacing * scale), "grid_size": int(grid_size),
            "translation_xy_m": shift.tolist() if execution else [0.0, 0.0],
            "yaw_deg": yaw_deg if execution else 0.0,
            "center_xy_m": center.tolist(),
            "positions_xy_m": (local * scale @ rotation.T + center).tolist(),
        }
    return layouts


def execution_point(xy, layouts):
    """Carry a demonstrated drawing point into the moved and scaled grid frame."""
    demo, execution = layouts["demonstration"], layouts["execution"]
    angle = np.radians(execution["yaw_deg"])
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    scale = execution["spacing_m"] / demo["spacing_m"]
    return ((np.asarray(xy) - demo["center_xy_m"]) * scale @ rotation.T
            + execution["center_xy_m"])
