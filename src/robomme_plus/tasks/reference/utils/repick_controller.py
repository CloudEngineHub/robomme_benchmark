"""Instance-local swarm controller and retained memory-demo helpers."""

import numpy as np

from robomme_plus.common.sim import (
    position,
    scalar,
    set_position,
    step_number,
)
from robomme_plus.tasks.reference.utils.repick_motion import (
    plan_rotations,
    plan_routes,
    swept_gap,
    yaw_quaternion,
)
from robomme_plus.scoring.subgoals import evaluate_tasks as evaluate_tasks


def wait_task(env, duration, demo=False, name="static"):
    timer = {"start": None}

    def ready():
        if timer["start"] is None:
            timer["start"] = step_number(env)
        return step_number(env) >= timer["start"] + duration

    return task(name, ready, lambda e, p: hold(e, p, duration + 1), demo=demo)


def initial_cube_occluded(env, xy, half_size):
    """Conservative camera-to-cube ray test against the expanded fixed board."""
    board = env._limit_diagnostics["occluder"]
    center = np.asarray(board["center_xyz_m"])
    extent = np.asarray(board["size_xyz_m"]) / 2 + half_size + 0.006
    eye = env._default_sensor_configs[0].pose.p
    if hasattr(eye, "detach"):
        eye = eye.detach().cpu().numpy()
    eye = np.asarray(eye).reshape(3)
    direction = np.asarray([*xy, half_size]) - eye
    with np.errstate(divide="ignore", invalid="ignore"):
        near, far = (
            (center - extent - eye) / direction,
            (center + extent - eye) / direction,
        )
    entry = max(0, float(np.max(np.minimum(near, far))))
    exit = min(1, float(np.min(np.maximum(near, far))))
    return entry <= exit


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
        workspace_bounds=None,
        max_group_diameter=None,
    ):
        self.env, self.actors = env, actors
        self.paths, self.metadata = plan_routes(
            len(actors),
            radius,
            int(env.seed) + 742931,
            steps=steps,
            speed=speed,
            obstacles=obstacles,
            initial=initial,
            initial_reject=initial_reject,
            workspace_bounds=workspace_bounds,
            max_group_diameter=max_group_diameter,
            exchange_destinations=env._limit_config["exchange_destinations"],
            goal_refresh_range=env._limit_config["goal_refresh_range"],
        )
        self.z = [float(position(a, initial=False)[2]) for a in actors]
        self.q = [
            a.pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy() for a in actors
        ]
        self.angles, self.metadata["rotation"] = plan_rotations(
            len(actors), int(env.seed) + 742933, steps,
            control_freq=env.control_freq,
        )
        self.rotation_cursors = np.zeros(len(actors), dtype=int)
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
            set_position(actor, p, yaw_quaternion(self.q[i], self.angles[index, i]))
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
        next_rotation_cursors = self.rotation_cursors.copy()
        for i, actor in enumerate(self.actors):
            if (
                i in self.pinned
                or position(actor, initial=False)[2] > self.z[i] + 0.02
                or scalar(self.env.agent.is_grasping(actor))
            ):
                continue
            # Turn independently of XY waits. Held, raised and pinned objects
            # are excluded above, so scripted rotation never fights a grasp.
            next_rotation_cursors[i] = min(
                self.rotation_cursors[i] + 1, len(self.angles) - 1
            )
            candidate_index = min(self.cursors[i] + 1, len(self.paths) - 1)
            candidate = self.paths[candidate_index, i]
            if (
                np.linalg.norm(candidate - before[i])
                > self.metadata["max_speed_m_per_control_step"] + 0.003
            ):
                continue
            trial = after.copy()
            trial[i] = candidate
            diameter_limit = self.metadata.get("max_group_diameter_m")
            if diameter_limit is not None and np.any(
                np.linalg.norm(trial[tabletop] - candidate, axis=1) > diameter_limit
            ):
                continue
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
        if self.metadata.get("max_group_diameter_m") is not None and tabletop.any():
            current = after[tabletop]
            diameter = float(
                np.linalg.norm(current[:, None] - current[None, :], axis=2).max()
            )
            self.metadata["maximum_actual_group_diameter_m"] = max(
                self.metadata.get("maximum_actual_group_diameter_m", 0), diameter
            )
        self.cursors = next_cursors
        self.cursor += 1
        for i, actor in enumerate(self.actors):
            if (np.linalg.norm(after[i] - before[i]) > 1e-8
                    or next_rotation_cursors[i] != self.rotation_cursors[i]):
                q = yaw_quaternion(self.q[i], self.angles[next_rotation_cursors[i], i])
                set_position(actor, [*after[i], self.z[i]], q)
        self.rotation_cursors = next_rotation_cursors
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


def prepare_memory_view(env, planner):
    """Use VideoRepick's low central viewing position for all video memory tasks."""
    import sapien

    native = env.unwrapped
    target = [0, 0, 0.20]
    quaternion = [0, 1, 0, 0]
    result = planner.move_to_pose_with_screw(
        sapien.Pose(target, quaternion), refine_steps=20
    )
    if isinstance(result, int) and result == -1:
        raise RuntimeError("Cannot reach the memory-video viewing pose")
    planner.open_gripper(t=6)
    actual = position(native.agent.tcp, initial=False)
    error = float(np.linalg.norm(actual - target))
    assert error < 0.02, f"Memory viewing pose not reached: {actual}"
    native._limit_view_ready = True
    native._limit_diagnostics["video_view_pose"] = {
        "reference": "VideoRepick solve_reset position, fixed downward orientation",
        "target_xyz_m": target,
        "target_quaternion_wxyz": quaternion,
        "actual_pose_wxyz": native.agent.tcp.pose.raw_pose.detach()
        .cpu()
        .numpy()
        .tolist(),
        "position_error_m": error,
        "ready_control_step": step_number(native),
    }


from robomme_plus.scoring.subgoals import hold as hold
from robomme_plus.scoring.subgoals import task as task
