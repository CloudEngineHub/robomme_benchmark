"""Pattern segments and stationary endpoints, independent of the simulator."""

import math

import numpy as np


class PatternDwell:
    """Follow each segment; a sustained stop selects the next node or fails.

    Each phase owns a fresh instance. Repeated evaluation at one control step
    cannot accumulate time. Holding a confirmed node is allowed until departure.
    Once the first node is confirmed, drawing must stay within 2 cm of the
    finite segment from the last confirmed node to the next required node.
    Crossing intermediate nodes along that segment does not select them.
    Drawing continuity is enforced separately by the shared Drawing rules.
    """

    def __init__(self, xy, nodes, dt=0.05, dwell_seconds=0.3):
        self.xy = np.asarray(xy, dtype=float)
        self.nodes = list(nodes)
        self.dt = float(dt)
        self.dwell_steps = math.ceil(dwell_seconds / self.dt)
        self.radius_m = 0.01
        # Long diagonal motions need more tracking margin than a settled stop.
        # This is still less than half the grid's 9 cm spacing.
        self.segment_tolerance_m = 0.02
        self.stop_speed_m_s = 0.03
        self.height_m = 0.08
        self.accepted = []
        self.events = []
        self.failure = None
        self.last_step = self.last_tip = None
        self.candidate = None
        self.dwell = 0
        self.previous_released = False
        self.held_node = None

    @property
    def success(self):
        return len(self.accepted) == len(self.nodes) and self.failure is None

    def update(self, step, tip, *, drawing_allowed=True):
        step = int(step)
        if self.last_step is not None and step <= self.last_step:
            return None
        tip = np.asarray(tip, dtype=float)
        consecutive = self.last_step is not None and step == self.last_step + 1
        speed = (float(np.linalg.norm(tip - self.last_tip) / self.dt)
                 if consecutive else math.inf)
        self.last_step, self.last_tip = step, tip.copy()
        distances = np.linalg.norm(self.xy - tip[:2], axis=1)
        drawing = bool(drawing_allowed and np.isfinite(tip).all() and 0 <= tip[2] <= self.height_m)
        node = int(distances.argmin()) if drawing and distances.min() <= self.radius_m else None
        previous = self.accepted[-1] if self.accepted else None
        if previous is not None and (not drawing or distances[previous] > self.radius_m * 1.5):
            self.previous_released = True
        self.held_node = (previous if node == previous and not self.previous_released else None)
        if self.failure or self.success:
            return None
        if drawing and previous is not None:
            expected = self.nodes[len(self.accepted)]
            start = self.xy[previous]
            segment = self.xy[expected] - start
            fraction = np.clip(np.dot(tip[:2] - start, segment) / np.dot(segment, segment), 0, 1)
            deviation = float(np.linalg.norm(tip[:2] - (start + fraction * segment)))
            if deviation > self.segment_tolerance_m:
                self.failure = {"step": step, "reason": "wrong_drawing_pattern",
                                "from_node": previous, "to_node": expected,
                                "deviation_m": deviation}
                return None
        if node is None or self.held_node is not None:
            self.candidate, self.dwell = None, 0
            return None
        # A qualifying interval needs both endpoints inside the same target
        # and at drawing height. Arrival, fast motion and gaps start a fresh
        # timer at zero; the next stationary interval can then count.
        self.dwell = (self.dwell + 1
                      if self.candidate == node and consecutive and speed <= self.stop_speed_m_s
                      else 0)
        self.candidate = node
        if self.dwell < self.dwell_steps:
            return None
        event = {"step": step, "node": node, "expected": self.nodes[len(self.accepted)],
                 "dwell_steps": self.dwell}
        if node != event["expected"]:
            self.failure = dict(event, reason="wrong_endpoint_stop")
            return None
        self.accepted.append(node)
        self.events.append(event)
        self.previous_released = False
        self.held_node = node
        self.candidate, self.dwell = None, 0
        return event

    def summary(self):
        return {"required_dwell_seconds": self.dwell_steps * self.dt,
                "required_dwell_steps": self.dwell_steps,
                "stop_speed_m_s": self.stop_speed_m_s, "radius_m": self.radius_m,
                "segment_tolerance_m": self.segment_tolerance_m,
                "height_m": self.height_m, "accepted": list(self.accepted),
                "events": [dict(event) for event in self.events],
                "failure": self.failure, "success": self.success}
