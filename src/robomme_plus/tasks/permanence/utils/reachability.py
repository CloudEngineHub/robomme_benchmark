"""Joint-limited grasp/lift checks and complete route auditing."""

import numpy as np


class CoverReachability:
    """Check grasp and 20 cm hover/lift poses without stepping the simulator."""

    def __init__(self, native):
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )

        self.native = native
        self.solver = PandaArmMotionPlanningSolver(
            native,
            vis=False,
            debug=False,
            base_pose=native.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        self.planner = self.solver.planner
        self.qpos = (
            native.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1).copy()
        )
        self.cache = {}
        self.records = []

    def check(self, xy, cover_z, quaternion=(0, 1, 0, 0)):
        key = (*np.asarray(xy).tolist(), float(cover_z), *quaternion)
        if key in self.cache:
            return self.cache[key]
        qpos = self.qpos.copy()
        statuses = []
        for z in [float(cover_z) + 0.01, 0.20]:
            world = np.r_[xy, z, quaternion]
            goal = self.planner.transform_goal_to_wrt_base(world)
            status, solutions = self.planner.IK(
                goal, qpos, n_init_qpos=1, threshold=1e-3
            )
            if status != "Success":
                # A second fixed seed avoids excluding a reachable pose just
                # because the home configuration has a poor local IK basin.
                alternate = self.qpos.copy()
                alternate[:7] = [0, -0.6, 0, -2.2, 0, 1.6, 0.785398]
                status, solutions = self.planner.IK(
                    goal, alternate, n_init_qpos=1, threshold=1e-3
                )
            statuses.append(status)
            if status != "Success":
                break
            qpos = solutions[0]
        passed = len(statuses) == 2 and all(s == "Success" for s in statuses)
        self.cache[key] = passed
        self.records.append(
            {
                "xy_m": np.asarray(xy).tolist(),
                "cover_z_m": float(cover_z),
                "quaternion_wxyz": list(quaternion),
                "passed": passed,
                "grasp_and_lift_ik_status": statuses,
            }
        )
        return passed

    def path(self, positions, cover_z):
        # Every generated waypoint is checked, including all possible stopping
        # positions. These records remain simulator diagnostics, never prompts.
        return all(self.check(xy, cover_z) for xy in positions)

    def document(self):
        return {
            "method": "Panda joint-limited, self-collision-checked IK at grasp and 20 cm hover/lift",
            "grasp_offset_m": 0.01,
            "lift_height_m": 0.20,
            "environment_collision_free_path_guaranteed": False,
            "checked_positions": len(self.records),
            "checks": self.records,
            "privileged_information_sent_to_model": False,
        }


class ReachableArea:
    def __init__(self, native, cover_z, *, max_x=0.14):
        self.reach = CoverReachability(native)
        self.cover_z = float(cover_z)
        self.x = np.linspace(-0.49, 0.14, 33)
        # Preserve existing grid vertices and test every newly admitted column.
        if max_x > self.x[-1]:
            spacing = self.x[1] - self.x[0]
            extra = int(np.ceil((max_x - self.x[-1]) / spacing))
            self.x = np.r_[self.x, self.x[-1] + spacing * np.arange(1, extra + 1)]
        self.y = np.linspace(-0.34, 0.34, 35)
        self.vertex = np.array(
            [[self.reach.check([x, y], cover_z) for y in self.y] for x in self.x]
        )
        self.cells = (
            self.vertex[:-1, :-1]
            & self.vertex[1:, :-1]
            & self.vertex[:-1, 1:]
            & self.vertex[1:, 1:]
        )
        # Four passing corners guide proposals; route coordinates are then
        # individually checked by exact IK, rather than inferred from this mask.
        for i, j in np.argwhere(self.cells):
            center = [(self.x[i] + self.x[i + 1]) / 2, (self.y[j] + self.y[j + 1]) / 2]
            self.cells[i, j] &= self.reach.check(center, cover_z)

    def indexes(self, points):
        points = np.asarray(points)
        i = np.searchsorted(self.x, points[..., 0], side="right") - 1
        j = np.searchsorted(self.y, points[..., 1], side="right") - 1
        okay = (i >= 0) & (j >= 0) & (i < len(self.x) - 1) & (j < len(self.y) - 1)
        return np.clip(i, 0, len(self.x) - 2), np.clip(j, 0, len(self.y) - 2), okay

    def contains(self, points):
        i, j, okay = self.indexes(points)
        return okay & self.cells[i, j]

    def segment(self, a, b):
        count = max(2, int(np.ceil(np.linalg.norm(np.asarray(b) - a) / 0.005)) + 1)
        return bool(self.contains(np.linspace(a, b, count)).all())

    def sample(self, rng, low, high):
        for _ in range(10000):
            xy = rng.uniform(low, high)
            if self.contains(xy):
                return xy
        raise RuntimeError("No reachable route destination")

    def validate_paths(self, paths, heights):
        failures = []
        for i, height in enumerate(heights):
            route = paths[:, i]
            # Audit every control-step position and the midpoint of each move.
            points = np.r_[route, (route[:-1] + route[1:]) / 2]
            for xy in points:
                if not self.reach.check(xy, height):
                    failures.append(dict(container_index=i, xy_m=xy.tolist()))
                    ix, iy, _ = self.indexes(xy)
                    self.cells[ix, iy] = False
        return failures

    def document(self, paths):
        return dict(
            self.reach.document(),
            all_container_routes_checked=True,
            all_container_routes_reachable=True,
            container_count=int(paths.shape[1]),
            planned_control_step_positions=int(paths.shape[0] * paths.shape[1]),
            inter_step_midpoints_checked=int((paths.shape[0] - 1) * paths.shape[1]),
            grid_spacing_m=[float(self.x[1] - self.x[0]), float(self.y[1] - self.y[0])],
            retained_grid_cells=int(self.cells.sum()),
            exact_ik_after_route_generation=True,
            guarantee="joint-limited grasp and hover/lift reachability, not a proof of a collision-free robot approach path",
        )
