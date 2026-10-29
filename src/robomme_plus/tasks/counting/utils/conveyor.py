"""An unlimited, seeded distractor stream for the BinFill conveyor."""

import numpy as np


class DistractorStream:
    def __init__(self, seed, colors, half_size, *, gap_steps, width, center_x, entry_y, speed_per_step):
        if not colors or gap_steps[0] < 1 or gap_steps[1] < gap_steps[0]:
            raise ValueError("Distractor colors and positive arrival gaps required")
        self.rng = np.random.default_rng([int(seed), 81799])
        self.colors = tuple(colors)
        self.half_size, self.width = half_size, width
        self.center_x, self.entry_y, self.speed_per_step = center_x, entry_y, speed_per_step
        self.gap_steps = tuple(gap_steps)
        self.next_step = self._gap()
        self.emitted = 0

    def _gap(self):
        return int(self.rng.integers(self.gap_steps[0], self.gap_steps[1] + 1))

    def sample(self, step, reserved):
        """Spawn when due and clear; retry congestion next step, without an end time.

        Reservations include future targets and current physical cubes. Positions
        are projected to this step, keeping the finite target train unchanged.
        """
        if step < self.next_step:
            return None
        for _ in range(64):
            yaw = float(self.rng.uniform(-np.pi, np.pi))
            extent = self.half_size * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
            margin = self.width / 2 - extent - .008
            if margin <= 0:
                raise ValueError("Belt is too narrow for a distractor cube")
            x = self.center_x + float(self.rng.uniform(-margin, margin))
            y = self.entry_y + float(self.rng.uniform(-.01, .01))
            if any(
                max(abs(x - other['entry_x']),
                    abs(y - other['entry_y'] - (step - other['release']) * self.speed_per_step))
                < extent + other['extent_xy'] + .025
                for other in reserved
            ):
                continue
            item = dict(color=str(self.rng.choice(self.colors)), release=int(step),
                        entry_x=x, entry_y=y, yaw_rad=yaw, extent_xy=float(extent),
                        arrival_role='stream_distractor', spawn_index=self.emitted)
            self.emitted += 1
            self.next_step = int(step) + self._gap()
            return item
        return None


def inside_bin(point, center):
    return np.linalg.norm(np.asarray(point)[:2] - np.asarray(center)[:2]) <= .035 and 0 <= point[2] <= .08
