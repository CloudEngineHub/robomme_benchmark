"""Continuous drawing for PatternLock and RouteStick (CPU-only rules).

A control step *draws* when the stick's real tip is at most ``height_m`` above
the table. Each drawing step leaves a white bubble at the tip that disappears on
its own clock (the native trail's 40 steps); nothing links two bubbles.

The drawing must be continuous: every drawing point lies within
``continuity_m`` (horizontally) of the previous drawing point. The stick may
lift and come down again, but the first point after a lift must also lie within
``continuity_m`` of the last point drawn, or the drawing has jumped. During
execution, a jump fails the episode, and so do more than ``idle_limit_steps``
consecutive steps without drawing (counted from the start of execution). The
demonstration follows the same drawing rules; its record is diagnostic only.
"""

import math

HEIGHT_M = 0.08
CONTINUITY_M = 0.03
IDLE_LIMIT_STEPS = 450


class Drawing:
    def __init__(
        self,
        height_m=HEIGHT_M,
        continuity_m=CONTINUITY_M,
        idle_limit_steps=IDLE_LIMIT_STEPS,
        max_angle_deg=None,
    ):
        self.height_m, self.continuity_m, self.idle_limit_steps = (
            height_m,
            continuity_m,
            idle_limit_steps,
        )
        self.phases = {}
        self.max_angle_deg = max_angle_deg
        self.phase = None
        self.failure = None

    def begin(self, phase, step):
        """Start a phase's own drawing (no continuity across phases)."""
        self.phase = phase
        self.phases[phase] = {
            "started_at_step": int(step),
            "points": [],
            "jumps": [],
            "resumes": 0,
            "idle_steps": 0,
            "max_idle_steps": 0,
            "first_point_step": None,
            "last_xy": None,
            "drawing": False,
        }

    def can_draw(self, tip_xyz, angle_deg=None):
        """Height and optional physical-axis angle share one contact rule."""
        if not all(math.isfinite(float(v)) for v in tip_xyz[:3]):
            return False
        if not 0 <= float(tip_xyz[2]) <= self.height_m:
            return False
        return (self.max_angle_deg is None or
                (angle_deg is not None and math.isfinite(angle_deg)
                 and 0 <= angle_deg < self.max_angle_deg))

    def update(self, step, tip_xyz, angle_deg=None):
        """Record one control step; return True when this step draws."""
        state = self.phases.get(self.phase)
        if state is None or self.failure is not None:
            return False
        x, y, z = map(float, tip_xyz[:3])
        if not self.can_draw(tip_xyz, angle_deg):
            state["drawing"] = False
            state["idle_steps"] += 1
            state["max_idle_steps"] = max(state["max_idle_steps"], state["idle_steps"])
            if (
                self.phase == "execution"
                and state["idle_steps"] > self.idle_limit_steps
            ):
                self.failure = {
                    "reason": "idle",
                    "step": int(step),
                    "idle_steps": state["idle_steps"],
                }
            return False
        if state["last_xy"] is not None:
            gap = math.dist(state["last_xy"], (x, y))
            if not state["drawing"]:
                state["resumes"] += 1
            if gap > self.continuity_m:
                jump = {
                    "step": int(step),
                    "gap_m": round(gap, 4),
                    "resume": not state["drawing"],
                }
                state["jumps"].append(jump)
                if self.phase == "execution":
                    self.failure = dict(jump, reason="discontinuous")
        elif state["first_point_step"] is None:
            state["first_point_step"] = int(step)
        state["points"].append([int(step), round(x, 4), round(y, 4)])
        state["last_xy"], state["drawing"], state["idle_steps"] = (x, y), True, 0
        return self.failure is None

    def summary(self):
        out = {
            "height_m": self.height_m,
            "max_angle_deg": self.max_angle_deg,
            "continuity_m": self.continuity_m,
            "idle_limit_steps": self.idle_limit_steps,
            "failure": self.failure,
            "phases": {},
        }
        for name, state in self.phases.items():
            out["phases"][name] = {
                key: value
                for key, value in state.items()
                if key not in ("last_xy", "drawing")
            }
        return out
