"""Small, simulator-independent measurements for method-based scoring."""

import numpy as np

TOOL_LIFT_M = 0.04
TOOL_LIFT_SECONDS = 0.10
# Maximum horizontal excursion of a loose tool's head or handle from its
# phase-start position. Unlike path length, this does not accumulate jitter.
MAX_DISTRACTOR_DISPLACEMENT_M = 0.05
HOOK_APPROACH_RADIUS_M = 0.08
CONTACT_MOTION_TOLERANCE_M = 0.002


class ToolLiftTracker:
    """A table bump is not use; both ends of a tool must leave their rest height."""

    def __init__(self):
        self.duration = {}
        self.used = set()
        self.max_rise = {}

    def update(self, method, rises, dt):
        if method == "cue":
            return False
        rise = float(min(rises))
        self.max_rise[method] = max(self.max_rise.get(method, 0.0), rise)
        self.duration[method] = self.duration.get(method, 0.0) + dt if rise >= TOOL_LIFT_M else 0.0
        if self.duration[method] + 1e-9 >= TOOL_LIFT_SECONDS:
            self.used.add(method)
        return method in self.used


def hook_inner_contact(point, normal, scale=1.0):
    """Inner crossbar or inward return-tip face; normals point tool -> cube."""
    x, y, _ = np.asarray(point, float) / scale
    nx, ny, nz = np.asarray(normal, float)
    back = abs(x + .006) <= .003 and abs(y) <= .035 and nx < -.5 and abs(nx) > abs(nz)
    side = (-.053 <= x <= -.003 and abs(abs(y) - .033) <= .003
            and y * ny < 0 and abs(ny) > .5 and abs(ny) > abs(nz))
    return bool(back or side)


def cue_point_velocity(slide_speed, yaw_speed, axis, offset_from_pivot):
    """Incoming velocity at any rod/handle/tip contact, including rotation."""
    return slide_speed * np.asarray(axis) + yaw_speed * np.cross([0, 0, 1], offset_from_pivot)
