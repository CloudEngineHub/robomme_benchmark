"""Physical covers, route execution, and ordered subgoal scoring."""

import math

import numpy as np

from robomme_plus.common.sim import position
from robomme_plus.common.sim import scalar as scalar
from robomme_plus.common.sim import set_position as set_position
from robomme_plus.common.sim import step_number as step_number
from robomme_plus.scoring.subgoals import evaluate_tasks as evaluate_tasks
from robomme_plus.scoring.subgoals import hold as hold
from robomme_plus.scoring.subgoals import task as task
from robomme_plus.tasks.permanence.utils.moving import swept_gap


def build_hollow_cover(
    env, *, callsign=None, position=None, z_rotation_deg=0, **kwargs
):
    """Keep native outer geometry but move its solid grasp block above the roof."""
    import sapien
    from transforms3d.euler import euler2quat

    half = env.cube_half_size
    inner, t, h, tf = half * 1.25, 0.0025, half * 1.25, 0.002
    xy = [0, 0] if position is None else list(position)[:2]
    builder = env.scene.create_actor_builder()
    builder.initial_pose = sapien.Pose(
        [*xy, tf + 2 * h],
        euler2quat(math.pi, 0, math.radians(z_rotation_deg), axes="rxyz"),
    )
    parts = [
        ([0, 0, -half / 2], [half, half, half / 2]),
        ([0, 0, tf], [inner + t, inner + t, tf]),
        ([-inner - t, 0, tf + h], [t, inner + t, h]),
        ([inner + t, 0, tf + h], [t, inner + t, h]),
        ([0, -inner - t, tf + h], [inner + t, t, h]),
        ([0, inner + t, tf + h], [inner + t, t, h]),
    ]
    for center, size in parts:
        builder.add_box_collision(sapien.Pose(center), size)
        builder.add_box_visual(sapien.Pose(center), size)
    return builder.build_dynamic(name=callsign)


class Routes:
    def __init__(
        self,
        env,
        actors,
        radius,
        speed,
        steps=720,
        obstacles=(),
        initial=None,
        initial_reject=None,
        planner=None,
        seed=None,
    ):
        self.env, self.actors = env, actors
        self.paths, self.metadata = planner(
            len(actors),
            radius,
            int(env.seed) + 742931 if seed is None else seed,
            steps=steps,
            speed=speed,
            obstacles=obstacles,
            initial=initial,
            initial_reject=initial_reject,
        )
        self.z = [float(position(a, initial=False)[2]) for a in actors]
        self.q = [
            a.pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy() for a in actors
        ]
        self.cursor = 0
        self.cursors = np.zeros(len(actors), dtype=int)
        self.radius = radius
        self.obstacles = obstacles
        self.active = True
        self.execution_start = None
        self.pinned = set()
        self.carried = {}
        self.metadata["actor_names"] = [a.name for a in actors]
        self.metadata["movement_steps"] = 0
        self.metadata["simultaneously_moving_max"] = 0
        self.apply(0)
        self.history = [self.paths[0].copy()]

    def location(self, i):
        return self.paths[self.cursors[i], i]

    def apply(self, index):
        index = min(index, len(self.paths) - 1)
        for i, actor in enumerate(self.actors):
            if i in self.pinned:
                continue
            p = [*self.paths[index, i], self.z[i]]
            old = position(actor, initial=False)
            set_position(actor, p, self.q[i])
            cube = self.carried.get(i)
            if cube is not None and position(cube, initial=False)[2] < 0.06:
                cp = position(cube, initial=False).copy()
                cp[:2] += np.asarray(p[:2]) - old[:2]
                set_position(cube, cp)

    def update(self):
        if not self.active:
            return
        before = np.array([position(a, initial=False)[:2] for a in self.actors])
        # A held object above the table can pass over another object. Only
        # tabletop footprints constrain the planar scripted trajectories.
        tabletop = np.array(
            [
                position(a, initial=False)[2] <= self.z[i] + 0.02
                for i, a in enumerate(self.actors)
            ]
        )
        radii = np.full(int(tabletop.sum()), self.radius)
        after = before.copy()
        next_cursors = self.cursors.copy()
        for i, actor in enumerate(self.actors):
            if (
                i in self.pinned
                or position(actor, initial=False)[2] > self.z[i] + 0.02
                or scalar(self.env.agent.is_grasping(actor))
            ):
                continue
            candidate_index = min(self.cursors[i] + 1, len(self.paths) - 1)
            candidate = self.paths[candidate_index, i]
            if (
                np.linalg.norm(candidate - before[i])
                > self.metadata["max_speed_m_per_control_step"] + 0.003
            ):
                continue
            trial = after.copy()
            trial[i] = candidate
            if swept_gap(before[tabletop], trial[tabletop], radii) < 0.004:
                continue
            after[i] = candidate
            next_cursors[i] = candidate_index
        minimum = swept_gap(before[tabletop], after[tabletop], radii)
        prior_gap = swept_gap(before[tabletop], before[tabletop], radii)
        if minimum < min(0, prior_gap) - 1e-6:
            raise RuntimeError(f"Moving-object collision: {minimum}")
        self.metadata["minimum_actual_continuous_pair_gap_m"] = min(
            self.metadata.get("minimum_actual_continuous_pair_gap_m", float("inf")),
            minimum,
        )
        moving = int(np.sum(np.linalg.norm(after - before, axis=1) > 1e-5))
        self.metadata["movement_steps"] += 1
        if self.execution_start is not None:
            self.metadata["motion_steps_after_reset"] = (
                self.metadata.get("motion_steps_after_reset", 0) + 1
            )
        self.metadata["simultaneously_moving_max"] = max(
            self.metadata["simultaneously_moving_max"], moving
        )
        self.cursors = next_cursors
        self.cursor += 1
        for i, actor in enumerate(self.actors):
            if np.linalg.norm(after[i] - before[i]) > 1e-8:
                set_position(actor, [*after[i], self.z[i]], self.q[i])
        self.history.append(after.copy())

    def predict(self, index, steps):
        before = np.array([position(a, initial=False)[:2] for a in self.actors])
        cursors = self.cursors.copy()
        for _ in range(max(0, int(steps))):
            after = before.copy()
            for i in range(len(before)):
                if i in self.pinned:
                    continue
                k = min(cursors[i] + 1, len(self.paths) - 1)
                trial = after.copy()
                trial[i] = self.paths[k, i]
                if swept_gap(before, trial, np.full(len(before), self.radius)) >= 0.004:
                    after[i] = trial[i]
                    cursors[i] = k
            before = after
        return before[index]


