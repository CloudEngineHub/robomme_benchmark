"""Distinct-cover scoring with interchangeable duplicates of the same color."""

import itertools

import numpy as np
from transforms3d.quaternions import quat2mat


def cover_exposure(cover_xyz, cover_q, cube_xyz, cube_q, half_size):
    """Return an opening reason when the cube no longer fits under its cover.

    Dimensions match build_hollow_cover. Small contact/settling tolerances
    allow 3 mm at the sides and 6 mm below the rim, whose nominal gap is 2 mm.
    This checks geometry, including tilt, rather than a grasp flag or a large
    pickup-height threshold.
    """
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * (half_size / 1.2)
    world = corners @ quat2mat(cube_q).T + np.asarray(cube_xyz)
    local = (world - np.asarray(cover_xyz)) @ quat2mat(cover_q)
    inner = half_size * 1.25
    if np.max(np.abs(local[:, :2])) > inner + .003:
        return "cube_outside_container_sides"
    if np.max(local[:, 2]) > .002 + 2 * inner + .006:
        return "container_lifted_or_tipped_open"
    if np.min(local[:, 2]) < .004 - .003:
        return "cube_outside_container_roof"
    return None


class ColorPickupProgress:
    """Accept each cover once, with interchangeable covers of the same color."""

    def __init__(self, color_order, counts, colors_by_actor):
        self.sequence = [color for color in color_order for _ in range(counts[color])]
        self.colors_by_actor = dict(colors_by_actor)
        self.picked = []

    def allowed(self, stage):
        return {
            name
            for name, color in self.colors_by_actor.items()
            if color == self.sequence[stage] and name not in self.picked
        }

    def invalid(self, stage, raised):
        return self.invalid_reason(stage, raised) is not None

    def invalid_reason(self, stage, raised):
        lifted = set(raised)
        if len(lifted) > 1:
            return "multiple_containers"
        if lifted & set(self.picked):
            return "repeated_container"
        if lifted - self.allowed(stage):
            return "wrong_container"
        return None

    def observe(self, stage, raised):
        if len(self.picked) > stage:
            return True
        assert len(self.picked) == stage
        if self.invalid(stage, raised) or not raised:
            return False
        self.picked.append(next(iter(raised)))
        return True
