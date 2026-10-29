"""Continuous ButtonUnmaskSwap traffic with per-cover uncover/re-cover state."""

import numpy as np

from robomme_plus.common.sim import position, scalar, set_position, step_number
from robomme_plus.common.motion import swept_gap
from .progress import cover_exposure


class ContinuousRoutes:
    def __init__(self, env, actors, radius, speed, *, planner, obstacles=(), steps=None):
        self.env, self.actors, self.radius = env, actors, radius
        (initial, self._stream), self.metadata = planner(
            len(actors), radius, int(env.seed) + 742931, continuous=True,
            speed=speed, obstacles=obstacles, initial=None, initial_reject=None)
        self.positions = np.asarray(initial).copy()
        self.paths = self.positions[None].copy()
        self.history = [self.positions.copy()]
        self.z = [float(position(a, False)[2]) for a in actors]
        self.q = [a.pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy() for a in actors]
        self.active, self.execution_start = True, None
        self.cursor = 0
        self.cursors = np.zeros(len(actors), dtype=int)
        self.lengths = np.zeros(len(actors))
        self.pinned = set()
        self.covered = np.ones(len(actors), dtype=bool)
        self.events = []
        self._recovering = {}
        self.metadata.update(actor_names=[a.name for a in actors], movement_steps=0,
                             simultaneously_moving_max=0, continuous_motion=True,
                             cover_events=self.events, motion_stop_control_step=None)

    def location(self, index):
        return self.positions[index].copy()

    def observe_covers(self):
        env = self.env
        if not getattr(env, '_limit_setup', False) or getattr(env, '_limit_failed', False):
            return
        now = step_number(env)
        for cube, index, z in env._limit_hidden:
            cover = self.actors[index]
            if cover.name not in env._permanence_cover_spawned:
                continue
            cp, bp = position(cover, False), position(cube, False)
            cq = cover.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
            bq = cube.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
            exposed = cover_exposure(cp, cq, bp, bq, env.cube_half_size) is not None
            if self.covered[index] and exposed:
                self.covered[index] = False
                self.positions[index] = bp[:2]
                self.events.append(dict(event='uncovered', actor_index=index,
                                        control_step=now, cube_xy_m=bp[:2].tolist()))
            elif not self.covered[index]:
                # Merely putting the empty cover elsewhere does not move its cube.
                recovered = (not exposed and cp[2] <= self.z[index] + .008
                             and not scalar(env.agent.is_grasping(cover)))
                if recovered:
                    since = self._recovering.setdefault(index, now)
                    if now - since >= 2:
                        self.covered[index] = True
                        self._recovering.pop(index, None)
                        self.events.append(dict(event='recovered', actor_index=index,
                                                control_step=now, cube_xy_m=bp[:2].tolist()))
                else:
                    self._recovering.pop(index, None)

    def update(self):
        if not self.active:
            return
        self.observe_covers()
        before = self.positions.copy()
        frozen = {i for i, actor in enumerate(self.actors)
                  if not self.covered[i] or scalar(self.env.agent.is_grasping(actor))
                  or position(actor, False)[2] > self.z[i] + .02}
        self.positions = self._stream.send((before, frozen))
        gap = float(swept_gap(before, self.positions, np.full(len(self.actors), self.radius)))
        self.metadata['minimum_actual_continuous_pair_gap_m'] = min(
            self.metadata.get('minimum_actual_continuous_pair_gap_m', float('inf')), gap)
        distance = np.linalg.norm(self.positions - before, axis=1)
        moving = distance > 1e-8
        for i, actor in enumerate(self.actors):
            if i not in frozen:
                set_position(actor, [*self.positions[i], self.z[i]], self.q[i])
        for cube, i, z in self.env._limit_hidden:
            # Uncovered cubes are stationary obstacles for the remaining traffic.
            set_position(cube, [*self.positions[i], z])
        self.cursors += moving
        self.lengths += distance
        self.cursor += 1
        self.history.append(self.positions.copy())
        self.metadata.update(movement_steps=self.cursor,
                             simultaneously_moving_max=max(self.metadata['simultaneously_moving_max'], int(moving.sum())),
                             current_covered=self.covered.tolist(),
                             route_lengths_m=self.lengths.tolist())

    def predict(self, index, steps):
        if len(self.history) < 2 or not self.covered[index]:
            return self.location(index)
        velocity = self.history[-1][index] - self.history[-2][index]
        return self.location(index) + min(max(int(steps), 0), 8) * velocity
