"""Independent seeded yaw velocities with smooth turns and pickup stops."""

import numpy as np


class SlowYaw:
    speed_range_deg_s = (3., 8.)
    change_interval_s = (3., 7.)
    acceleration_deg_s2 = 4.

    def __init__(self, seed, count, dt):
        self.dt = float(dt)
        self.rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]
        self.velocity = np.zeros(count)
        self.target = np.zeros(count)
        self.remaining = np.zeros(count)
        for i in range(count):
            self._sample(i)

    def _sample(self, i):
        rng = self.rng[i]
        self.target[i] = np.deg2rad(rng.uniform(*self.speed_range_deg_s)) * rng.choice([-1., 1.])
        self.remaining[i] = rng.uniform(*self.change_interval_s)

    def advance(self, active):
        active = np.asarray(active, dtype=bool)
        for i in np.flatnonzero(active & (self.remaining <= 0)):
            self._sample(int(i))
        limit = np.deg2rad(self.acceleration_deg_s2) * self.dt
        self.velocity += np.clip(self.target-self.velocity, -limit, limit)
        self.velocity[~active] = 0.
        self.remaining[active] -= self.dt
        return self.velocity * self.dt
