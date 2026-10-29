"""Continuous traffic with gripper approach stops and moving exposed cubes."""

import numpy as np
from transforms3d.quaternions import quat2mat

from robomme_plus.common.sim import position, scalar, set_position, step_number
from robomme_plus.common.motion import swept_gap
from .progress import cover_exposure
from .continuous_v1 import ContinuousRoutes as PreviousRoutes


def gripper_over_container(tcp, center, quaternion, half_size):
    """TCP in the exterior XY footprint and <5 cm above the grasp block.

    Dimensions match build_hollow_cover, including its exterior walls and
    handle. A TCP below the top still counts while fingers close or lift.
    """
    rotation = quat2mat(quaternion)
    outer = 1.25 * half_size + .005
    offset = (np.asarray(tcp)[:2] - np.asarray(center)[:2]) @ rotation[:2, :2]
    middle_z = (.002 + 2 * 1.25 * half_size - half_size) / 2
    extent_z = (.002 + 2 * 1.25 * half_size + half_size) / 2
    top = center[2] + rotation[2, 2] * middle_z + np.abs(rotation[2]) @ [outer, outer, extent_z]
    bottom = center[2] + rotation[2, 2] * middle_z - np.abs(rotation[2]) @ [outer, outer, extent_z]
    return bool(np.all(np.abs(offset) <= outer) and bottom <= tcp[2] < top + .05)


class ContinuousRoutes(PreviousRoutes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused = {}
        self.metadata.update(continuous_motion_version=2, gripper_stop_height_m=.05)

    def _traffic_state(self):
        env = self.env
        tcp = position(env.agent.tcp, False)
        cubes = {i: cube for cube, i, _ in env._limit_hidden}
        frozen, parked = {}, []
        for i, cover in enumerate(self.actors):
            cp = position(cover, False)
            cq = cover.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
            held = scalar(env.agent.is_grasping(cover))
            if self.covered[i]:
                if held or cp[2] > self.z[i] + .02:
                    frozen[i] = 'container_held'
                elif gripper_over_container(tcp, cp, cq, env.cube_half_size):
                    frozen[i] = 'gripper_near_container'
            else:
                cube = cubes[i]
                bp = position(cube, False)
                # A released tabletop cover stays where the robot put it;
                # only its uncovered cube rejoins the route stream.
                if held or cp[2] > self.z[i] + .008:
                    frozen[i] = 'container_held_or_unsettled'
                elif cover_exposure(cp, cq, bp,
                        cube.pose.q.detach().cpu().numpy().reshape(-1, 4)[0],
                        env.cube_half_size) is None:
                    frozen[i] = 'cover_recovering'
                elif gripper_over_container(tcp, bp + [0, 0, self.z[i] - bp[2]],
                                             self.q[i], env.cube_half_size):
                    frozen[i] = 'gripper_near_cube'
                if cp[2] <= self.z[i] + .02:
                    parked.append((cp[:2].copy(), np.sqrt(2) * (1.25 * env.cube_half_size + .005)))
        return frozen, parked

    def update(self):
        if not self.active:
            return
        self.observe_covers()
        before = self.positions.copy()
        frozen, parked = self._traffic_state()
        now = step_number(self.env)
        for i in sorted(set(frozen) | set(self._paused)):
            if frozen.get(i) != self._paused.get(i):
                self.events.append(dict(event='paused' if i in frozen else 'resumed',
                    actor_index=i, control_step=now, reason=frozen.get(i, self._paused.get(i)),
                    covered=bool(self.covered[i]), cube_xy_m=before[i].tolist()))
        self._paused = frozen.copy()
        self.positions = self._stream.send((before, set(frozen), parked))
        gap = float(swept_gap(before, self.positions, np.full(len(self.actors), self.radius)))
        self.metadata['minimum_actual_continuous_pair_gap_m'] = min(
            self.metadata.get('minimum_actual_continuous_pair_gap_m', float('inf')), gap)
        distance = np.linalg.norm(self.positions - before, axis=1)
        moving = distance > 1e-8
        for i, actor in enumerate(self.actors):
            if self.covered[i] and i not in frozen:
                set_position(actor, [*self.positions[i], self.z[i]], self.q[i])
        for cube, i, z in self.env._limit_hidden:
            set_position(cube, [*self.positions[i], z])
        self.cursors += moving
        self.lengths += distance
        self.cursor += 1
        self.history.append(self.positions.copy())
        self.metadata.update(movement_steps=self.cursor,
                             simultaneously_moving_max=max(self.metadata['simultaneously_moving_max'], int(moving.sum())),
                             current_covered=self.covered.tolist(),
                             paused_actor_indices=sorted(frozen),
                             parked_cover_obstacles=len(parked),
                             route_lengths_m=self.lengths.tolist())
