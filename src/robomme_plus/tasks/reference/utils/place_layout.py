"""Seeded target sampling over the reachable table region."""

import math

import numpy as np

from robomme_plus.tasks.reference.utils import place_swaps as swaps


def sample_scene_targets(seed, layout, cube_half_size, count=6, radius=0.04):
    obstacles = [(np.asarray(layout["button_xy"]), 0.055)]
    obstacles += [
        (np.asarray(p[:2]), math.sqrt(2) * cube_half_size)
        for p in layout["cubes"].values()
    ]
    layout_rng = np.random.SeedSequence(int(seed) + 872431).spawn(3)[0]
    return swaps.sample_random_layout(
        count, radius, np.random.default_rng(layout_rng), obstacles
    )


def build_ring(env, point, name, radius=0.04, thickness=0.005):
    import sapien

    from robomme_plus.simulation.actors import (
        build_purple_white_target,
    )

    target = build_purple_white_target(
        env.scene,
        radius=radius,
        thickness=thickness,
        name=name,
        body_type="kinematic",
        add_collision=False,
        initial_pose=sapien.Pose(
            p=[*point, thickness], q=[math.sqrt(0.5), 0, math.sqrt(0.5), 0]
        ),
    )
    target._target_radius = radius
    if not hasattr(env, "_spawned_targets"):
        env._spawned_targets, env._spawned_target_count = [], 0
    env._spawned_targets.append(target)
    env._spawned_target_count += 1
    return target
