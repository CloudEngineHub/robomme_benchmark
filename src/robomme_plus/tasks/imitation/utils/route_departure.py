"""Slow route interception triggered by arrival or departure at a route node."""

import math
from numbers import Real

import numpy as np


def interception_contact(env):
    """Use the full drawing-height rule even for an untilted straight stick."""
    from .split_stick import tip_angle_deg, tip_position

    tip = tip_position(env)
    drawing = getattr(env, "_drawing", None)
    angle = tip_angle_deg(env) if drawing is not None and drawing.max_angle_deg is not None else None
    return tip, drawing is not None and drawing.can_draw(tip, angle)


def resolve_departure(parameters):
    mode = parameters.get("mode", "departure_boards")
    arrival = mode == "arrival_boards"
    defaults = dict(move_s=2. if arrival else 1.5, hold_s=1.5,
                    return_s=2. if arrival else 1.5, stroke_m=.07 if arrival else .10)
    defaults.update(dict(arrival_radius_m=.03) if arrival else
                    dict(arm_radius_m=.03, departure_radius_m=.035))
    allowed = {"mode", "phases", "enabled_boards", *defaults}
    if set(parameters)-allowed:
        raise ValueError(f"Unknown RouteStick interception fields: {sorted(set(parameters)-allowed)}")
    values = {key:parameters.get(key,value) for key,value in defaults.items()}
    if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in values.values()):
        raise ValueError("RouteStick interception values must be finite numbers")
    if (values["move_s"] < 1. or values["return_s"] < 1. or values["hold_s"] < 0
            or not .03 <= values["stroke_m"] <= .14):
        raise ValueError("Route interception needs slow travel and a 3–14 cm stroke")
    if arrival:
        if not 0 < values["arrival_radius_m"] <= .03:
            raise ValueError("Arrival boards need a contact radius above zero and at most 3 cm")
    elif not 0 < values["arm_radius_m"] < values["departure_radius_m"] <= .05:
        raise ValueError("Departure boards need distinct arm/leave radii")
    phases = parameters.get("phases", ["execution"] if arrival else ["demonstration", "execution"])
    if (not isinstance(phases,(list,tuple)) or not phases
            or any(p not in ("demonstration","execution") for p in phases)
            or len(set(phases)) != len(phases)):
        raise ValueError("Interception phases must select demonstration and/or execution")
    resolved = dict(mode=mode,**{k:float(v) for k,v in values.items()},
                    phases=[p for p in ("demonstration","execution") if p in phases])
    if "enabled_boards" in parameters:
        enabled = parameters["enabled_boards"]
        if (not isinstance(enabled, (list, tuple)) or len(enabled) != 4
                or any(type(value) is not bool for value in enabled)):
            raise ValueError("RouteStick enabled_boards must contain four booleans")
        resolved["enabled_boards"] = list(enabled)
    return resolved


class DepartureCrossings:
    """Trigger on real node arrival or departure, once per curve, without jumps."""
    def __init__(self, parameters):
        self.parameters = parameters
        # Saved episodes without a mask retain their original all-board motion.
        self.enabled_boards = tuple(parameters.get("enabled_boards", [True] * 4))
        self.armed = set()
        self.fired = set()
        self.events = []

    def update(self, *, curve, start_index, end_index, direction, tip_xy, start_xy, drawing, time_s):
        if curve is None or curve in self.fired:
            return
        if (start_index not in (0,2,4,6,8) or end_index not in (0,2,4,6,8)
                or abs(end_index-start_index) != 2
                or direction not in ("clockwise","counterclockwise")):
            raise ValueError("Route interception needs adjacent route targets and a valid direction")
        distance = float(np.linalg.norm(np.asarray(tip_xy)-np.asarray(start_xy)))
        if not math.isfinite(distance):
            return
        arrival = self.parameters["mode"] == "arrival_boards"
        if arrival:
            if not drawing or distance > self.parameters["arrival_radius_m"]:
                return
        else:
            if drawing and distance <= self.parameters["arm_radius_m"]:
                self.armed.add(curve)
            if curve not in self.armed or distance <= self.parameters["departure_radius_m"]:
                return
        self.fired.add(curve)
        board = (min(start_index,end_index)) // 2
        if not self.enabled_boards[board]:
            return
        total = self.parameters["move_s"]+self.parameters["hold_s"]+self.parameters["return_s"]
        # An early reversal can revisit a still-moving board. Continue that
        # physical event rather than teleporting it back to start a new stroke.
        if any(e["board"] == board and time_s-e["started_s"] < total for e in self.events):
            return
        side = (1 if direction == "clockwise" else -1) * (1 if end_index > start_index else -1)
        event = dict(curve=int(curve),board=board,side=side,started_s=float(time_s),
                     start_index=start_index,end_index=end_index)
        event["arrival_distance_m" if arrival else "departure_distance_m"] = distance
        self.events.append(event)

    def offsets(self, time_s):
        offsets = np.zeros(4)
        states = [dict(board=i,stage="rest",curve=None) for i in range(4)]
        p = self.parameters
        for event in self.events:
            age = float(time_s)-event["started_s"]
            if age < 0:
                continue
            value,stage = 0.,"clear"
            if age < p["move_s"]:
                u,stage = age/p["move_s"],"entering"
                value = u*u*(3-2*u)
            elif age < p["move_s"]+p["hold_s"]:
                value,stage = 1.,"blocking"
            elif age < p["move_s"]+p["hold_s"]+p["return_s"]:
                u,stage = (age-p["move_s"]-p["hold_s"])/p["return_s"],"retracting"
                value = 1-u*u*(3-2*u)
            offsets[event["board"]] = event["side"] * p["stroke_m"] * value
            states[event["board"]] = dict(event,stage=stage,age_s=age)
        return offsets,states

    def positions(self, rest_xy, row_direction_xy, time_s, spacing_scale=1.):
        row = np.asarray(row_direction_xy,float)
        row = row / np.linalg.norm(row)
        normal = np.array([-row[1],row[0]])
        offsets,states = self.offsets(time_s)
        return np.asarray(rest_xy) + offsets[:,None]*spacing_scale*normal,states
