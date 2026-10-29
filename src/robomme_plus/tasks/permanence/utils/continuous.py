"""Continuous traffic with a quiet neighborhood around a low gripper."""

import numpy as np
from transforms3d.quaternions import quat2mat, axangle2quat, qmult

from robomme_plus.common.sim import position, scalar, set_position, step_number
from robomme_plus.common.motion import swept_gap
from .progress import cover_exposure
from .continuous_v1 import ContinuousRoutes as PreviousRoutes
from .rotation import SlowYaw


STOP_HEIGHT_M = .06
NEIGHBOR_RADIUS_M = .10
PICKUP_TRAFFIC_SPEED_SCALE = .5
BUTTON_PRESS_GRACE_STEPS = 150


def container_height_bounds(center, quaternion, half_size):
    """Physical bounds of the hollow cover, including its grasp block."""
    rotation = quat2mat(quaternion)
    outer = 1.25 * half_size + .005
    middle_z = (.002 + 2 * 1.25 * half_size - half_size) / 2
    extent_z = (.002 + 2 * 1.25 * half_size + half_size) / 2
    middle = center[2] + rotation[2, 2] * middle_z
    extent = np.abs(rotation[2]) @ [outer, outer, extent_z]
    return middle - extent, middle + extent


def gripper_over_container(tcp, center, quaternion, half_size):
    """TCP in the exterior XY footprint and <6 cm above the grasp block.

    Dimensions match build_hollow_cover, including its exterior walls and
    handle. A TCP below the top still counts while fingers close or lift.
    """
    rotation = quat2mat(quaternion)
    outer = 1.25 * half_size + .005
    offset = (np.asarray(tcp)[:2] - np.asarray(center)[:2]) @ rotation[:2, :2]
    bottom, top = container_height_bounds(center, quaternion, half_size)
    return bool(np.all(np.abs(offset) <= outer) and bottom <= tcp[2] < top + STOP_HEIGHT_M)


class ContinuousRoutes(PreviousRoutes):
    def __init__(self, *args, **kwargs):
        planner = kwargs['planner']
        def bypass_planner(*a, **kw):
            return planner(*a, pickup_bypass_margin=.08, pickup_passing=True, **kw)
        kwargs['planner'] = bypass_planner
        super().__init__(*args, **kwargs)
        self._paused = {}
        self._nearby = set()
        # Fixed tabletop height: a carried cover must not raise the pause
        # ceiling and keep its former neighbors stopped during transport.
        self._pause_ceiling = max(container_height_bounds(
            [0, 0, z], [0, 1, 0, 0], self.env.cube_half_size)[1]
            for z in self.z) + STOP_HEIGHT_M
        self._yaw_motion = SlowYaw(int(self.env.seed)+3927109, len(self.actors),
                                   1./self.env.control_freq)
        self._cube_q = None
        self.yaw_history = [np.zeros(len(self.actors))]
        self.radius_history = [np.full(len(self.actors), self.radius)]
        self.metadata.update(continuous_motion_version=6, gripper_stop_height_m=STOP_HEIGHT_M,
            gripper_pause_radius_m=NEIGHBOR_RADIUS_M,
            gripper_pause_grace_steps=BUTTON_PRESS_GRACE_STEPS,
            gripper_pause_grace_clock='execution simulator control steps; API waiting and action penalties excluded',
            gripper_pause_release='TCP lifts above tabletop container top plus stop height',
            pickup_traffic_speed_scale=PICKUP_TRAFFIC_SPEED_SCALE,
            rotation=dict(axis='world_z', speed_range_deg_s=list(SlowYaw.speed_range_deg_s),
                          target_change_interval_s=list(SlowYaw.change_interval_s),
                          acceleration_deg_s2=SlowYaw.acceleration_deg_s2,
                          independent_per_container=True, pauses_during_pickup=True,
                          parked_empty_covers_rotate=False, exposed_cubes_rotate=False))

    def _pickup_neighborhood(self, tcp, cubes):
        # Motion is updated before the next simulator step. Suppress proximity
        # stops through execution steps 1..150; enable them for step 151.
        enabled = (self.execution_start is not None
                   and step_number(self.env) - self.execution_start >= BUTTON_PRESS_GRACE_STEPS)
        self.metadata['gripper_proximity_pauses_enabled'] = enabled
        if not enabled or tcp[2] >= self._pause_ceiling:
            self._nearby.clear()
            return
        # Accumulate neighbors during a low approach. Sliding sideways or
        # leaving a single handle's footprint must not restart a paused pair.
        for i, cover in enumerate(self.actors):
            centers = [position(cover, False)[:2]]
            if i in cubes:
                centers.append(position(cubes[i], False)[:2])
            if any(np.linalg.norm(tcp[:2] - center) <= NEIGHBOR_RADIUS_M
                   for center in centers):
                self._nearby.add(i)

    def _traffic_state(self):
        env = self.env
        tcp = position(env.agent.tcp, False)
        cubes = {i: cube for cube, i, _ in env._limit_hidden}
        self._pickup_neighborhood(tcp, cubes)
        frozen = {i: 'pickup_neighborhood' for i in self._nearby}
        parked = []
        for i, cover in enumerate(self.actors):
            cp = position(cover, False)
            cq = cover.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
            held = scalar(env.agent.is_grasping(cover))
            if self.covered[i]:
                if held or cp[2] > self.z[i] + .02:
                    frozen[i] = 'container_held'
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
                else:
                    # A misplaced cover that still overlaps its cube cannot
                    # let the route controller slide the cube through a wall.
                    rotation = quat2mat(cq)
                    cube_rotation = quat2mat(cube.pose.q.detach().cpu().numpy().reshape(-1, 4)[0])
                    projected_half = (env.cube_half_size / 1.2) * np.abs(
                        cube_rotation.T @ rotation).sum(axis=0)[:2]
                    offset = (bp - cp) @ rotation
                    if np.all(np.abs(offset[:2]) <= 1.25 * env.cube_half_size + .005 + projected_half):
                        frozen[i] = 'cover_overlaps_cube'
                # Reserve a descending cover early, before its underside can
                # overlap neighboring tabletop containers.
                if cp[2] <= self.z[i] + .12:
                    parked.append((cp[:2].copy(), np.sqrt(2) * (1.25 * env.cube_half_size + .005)))
        return frozen, parked

    def update(self):
        if not self.active:
            return
        self.observe_covers()
        cubes = {i: cube for cube, i, _ in self.env._limit_hidden}
        # Physics can move a pair during contact. Plan from its actual pose;
        # never reset a paused cube to an older route coordinate.
        for i, cover in enumerate(self.actors):
            anchor = cover if self.covered[i] or i not in cubes else cubes[i]
            self.positions[i] = position(anchor, False)[:2]
        before = self.positions.copy()
        frozen, parked = self._traffic_state()
        speed_scale = PICKUP_TRAFFIC_SPEED_SCALE if self._nearby else 1.
        now = step_number(self.env)
        for i in sorted(set(frozen) | set(self._paused)):
            if frozen.get(i) != self._paused.get(i):
                self.events.append(dict(event='paused' if i in frozen else 'resumed',
                    actor_index=i, control_step=now, reason=frozen.get(i, self._paused.get(i)),
                    covered=bool(self.covered[i]), cube_xy_m=before[i].tolist()))
        if self._cube_q is None:
            self._cube_q = {i: cube.pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy()
                            for i, cube in cubes.items()}
        active_yaw = self.covered & ~np.isin(np.arange(len(self.actors)), list(frozen))
        delta_yaw = self._yaw_motion.advance(active_yaw) * speed_scale
        for i in np.flatnonzero(active_yaw):
            # Transport both physical poses by the same rigid transform. A
            # returned cover need not be perfectly centered over its cube.
            self.q[i] = self.actors[i].pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy()
            if i in cubes:
                self._cube_q[i] = cubes[i].pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy()
            turn = axangle2quat([0, 0, 1], delta_yaw[i])
            self.q[i] = qmult(turn, self.q[i])
            if i in self._cube_q:
                self._cube_q[i] = qmult(turn, self._cube_q[i])
        self._paused = frozen.copy()
        blockers = {i for i, reason in frozen.items() if reason in (
            'pickup_neighborhood', 'container_held', 'container_held_or_unsettled',
            'cover_overlaps_cube')}
        radii = np.where(self.covered, self.radius, np.sqrt(2)*self.env.cube_half_size/1.2)
        self.positions = self._stream.send((before, set(frozen), parked,
                                           dict(pickup_blockers=blockers, actor_radii_m=radii,
                                                speed_scale=speed_scale, allow_contact_recovery=True)))
        gap = float(swept_gap(before, self.positions, radii))
        self.metadata['minimum_actual_continuous_pair_gap_m'] = min(
            self.metadata.get('minimum_actual_continuous_pair_gap_m', float('inf')), gap)
        distance = np.linalg.norm(self.positions - before, axis=1)
        moving = distance > 1e-8
        for i, actor in enumerate(self.actors):
            if self.covered[i] and i not in frozen:
                set_position(actor, [*self.positions[i], self.z[i]], self.q[i])
        for cube, i, z in self.env._limit_hidden:
            if i in frozen:
                continue
            if self.covered[i]:
                turn = quat2mat(axangle2quat([0, 0, 1], delta_yaw[i]))
                relative = position(cube, False) - [*before[i], self.z[i]]
                point = np.array([*self.positions[i], self.z[i]]) + turn @ relative
                set_position(cube, point, self._cube_q[i])
            else:
                set_position(cube, [*self.positions[i], z])
        self.cursors += moving
        self.lengths += distance
        self.cursor += 1
        self.history.append(self.positions.copy())
        yaws = np.array([np.arctan2(quat2mat(q)[1, 0], quat2mat(q)[0, 0]) for q in self.q])
        self.yaw_history.append(yaws)
        self.radius_history.append(radii.copy())
        self.metadata.update(movement_steps=self.cursor,
                             simultaneously_moving_max=max(self.metadata['simultaneously_moving_max'], int(moving.sum())),
                             current_covered=self.covered.tolist(),
                             paused_actor_indices=sorted(frozen),
                             pickup_neighborhood_indices=sorted(self._nearby),
                             traffic_speed_scale=speed_scale,
                             parked_cover_obstacles=len(parked),
                             container_yaw_rad=yaws.tolist(),
                             angular_velocity_deg_s=np.rad2deg(self._yaw_motion.velocity * speed_scale).tolist(),
                             actor_footprint_radii_m=radii.tolist(),
                             route_lengths_m=self.lengths.tolist())
