"""Seeded intermittent board crossings perpendicular to the RouteStick row."""

import math
from numbers import Real

import numpy as np


def resolve_crossing(seed, parameters):
    allowed = {"mode", "period_s", "move_s", "hold_s", "stroke_m", "probability",
               "phases", "schedule_seed", "start_delays_s"}
    if set(parameters) - allowed:
        raise ValueError(f"Unknown RouteStick crossing fields: {sorted(set(parameters)-allowed)}")
    defaults = dict(period_s=6., move_s=.75, hold_s=1.5, stroke_m=.10, probability=.5)
    values = {key: parameters.get(key, default) for key, default in defaults.items()}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"RouteStick crossing {key} must be finite")
    if (values["move_s"] <= 0 or values["hold_s"] < 0
            or values["period_s"] <= 2 * values["move_s"] + values["hold_s"]):
        raise ValueError("RouteStick crossing cycle needs positive travel time and a clear interval")
    if not .03 <= values["stroke_m"] <= .14 or not 0 <= values["probability"] <= 1:
        raise ValueError("RouteStick crossing stroke must be 3–14 cm and probability 0–1")
    phases = parameters.get("phases", ["demonstration", "execution"])
    if (not isinstance(phases, (list, tuple)) or not phases
            or any(p not in ("demonstration", "execution") for p in phases)
            or len(set(phases)) != len(phases)):
        raise ValueError("RouteStick crossing phases must select demonstration and/or execution")
    rng = np.random.default_rng([int(seed), 618009])
    schedule_seed = parameters.get("schedule_seed", int(rng.integers(2**31)))
    if type(schedule_seed) is not int or not 0 <= schedule_seed < 2**32:
        raise ValueError("RouteStick crossing schedule_seed must be an unsigned 32-bit integer")
    delays = parameters.get("start_delays_s", rng.uniform(0, values["period_s"], 4).tolist())
    if (not isinstance(delays, (list, tuple)) or len(delays) != 4
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                   or not 0 <= v < values["period_s"] for v in delays)):
        raise ValueError("RouteStick crossing needs four start delays within the cycle")
    return dict(mode="crossing_boards", **{k:float(v) for k,v in values.items()},
                phases=[p for p in ("demonstration", "execution") if p in phases],
                schedule_seed=schedule_seed, start_delays_s=list(map(float, delays)))


def crossing_state(parameters, time_s):
    """No online RNG draws: querying, pausing or resetting cannot change a decision."""
    offsets, states = [], []
    for index, delay in enumerate(parameters["start_delays_s"]):
        elapsed = float(time_s) - delay
        cycle = math.floor(elapsed / parameters["period_s"])
        local = elapsed - cycle * parameters["period_s"]
        value, sign, chosen, stage = 0., 0, False, "clear"
        if elapsed >= 0:
            rng = np.random.default_rng([parameters["schedule_seed"], index, cycle])
            chosen = bool(rng.random() < parameters["probability"])
            sign = int(rng.choice((-1, 1)))
            move, hold = parameters["move_s"], parameters["hold_s"]
            if chosen and local < move:
                u, stage = local / move, "entering"
                value = u*u*(3-2*u)
            elif chosen and local < move + hold:
                value, stage = 1., "blocking"
            elif chosen and local < 2*move + hold:
                u, stage = (local-move-hold) / move, "retracting"
                value = 1-u*u*(3-2*u)
        offsets.append(sign * parameters["stroke_m"] * value)
        states.append(dict(board=index, cycle=cycle, chosen=chosen, side=sign, stage=stage))
    return np.array(offsets), states


def crossing_positions(rest_xy, row_direction_xy, parameters, time_s, spacing_scale=1.):
    axis = np.asarray(row_direction_xy, float)
    axis = axis / np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    offsets, states = crossing_state(parameters, time_s)
    return np.asarray(rest_xy) + offsets[:, None] * spacing_scale * normal, states
