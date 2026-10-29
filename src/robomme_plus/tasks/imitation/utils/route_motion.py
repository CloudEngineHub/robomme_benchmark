"""Replayable RouteStick board swings, with optional rigid-layout motion."""

import math
from numbers import Real

import numpy as np


def resolve_board_motion(seed, parameters):
    if not isinstance(parameters, dict):
        raise ValueError("RouteStick board_motion must be a dictionary")
    if parameters.get("mode") == "crossing_boards":
        from .route_crossing import resolve_crossing
        return resolve_crossing(seed, parameters)
    if parameters.get("mode") in ("arrival_boards", "departure_boards"):
        from .route_departure import resolve_departure
        return resolve_departure(parameters)
    allowed = {"mode", "amplitude_deg", "period_s", "phases", "pivot_target_index",
               "direction_sign", "pivot_target_indices", "direction_signs"}
    if set(parameters) - allowed:
        raise ValueError(f"Unknown RouteStick board_motion fields: {sorted(set(parameters) - allowed)}")
    amplitude = parameters.get("amplitude_deg", 15.)
    period = parameters.get("period_s", 4.)
    phases = parameters.get("phases", ["demonstration", "execution"])
    # Independent swings are the current default; preserve explicit rigid records.
    legacy = "pivot_target_indices" in parameters or "direction_signs" in parameters
    rigid = "pivot_target_index" in parameters or "direction_sign" in parameters
    mode = parameters.get("mode", "rigid_layout" if rigid else "independent_boards")
    if mode not in ("rigid_layout", "independent_boards"):
        raise ValueError("RouteStick motion mode must be rigid_layout or independent_boards")
    rng = np.random.default_rng([int(seed), 618008])
    if (isinstance(amplitude, bool) or not isinstance(amplitude, Real)
            or not math.isfinite(amplitude) or not 0 < amplitude <= 30):
        raise ValueError("RouteStick board swing amplitude must be above 0 and at most 30 degrees")
    if (isinstance(period, bool) or not isinstance(period, Real)
            or not math.isfinite(period) or period <= 0):
        raise ValueError("RouteStick board swing period must be finite and positive")
    if (not isinstance(phases, (tuple, list)) or not phases
            or any(p not in ("demonstration", "execution") for p in phases)
            or len(set(phases)) != len(phases)):
        raise ValueError("RouteStick board motion phases must select demonstration and/or execution")
    common = dict(mode=mode, amplitude_deg=float(amplitude), period_s=float(period),
                  phases=[p for p in ("demonstration", "execution") if p in phases])
    if mode == "rigid_layout":
        if legacy:
            raise ValueError("Rigid RouteStick motion has one pivot and one direction")
        pivot = parameters.get("pivot_target_index", 4)
        sign = parameters.get("direction_sign", int(rng.choice((-1, 1))))
        if type(pivot) is not int or pivot not in (0, 2, 4, 6, 8):
            raise ValueError("RouteStick layout pivot must be a route target index")
        if type(sign) is not int or sign not in (-1, 1):
            raise ValueError("RouteStick layout direction_sign must be +1 or -1")
        return dict(common, pivot_target_index=pivot, direction_sign=sign)
    if "pivot_target_index" in parameters or "direction_sign" in parameters:
        raise ValueError("Independent board records require per-board pivots and directions")
    pivots = parameters.get("pivot_target_indices", [0, 2, 4, 6])
    signs = parameters.get("direction_signs", rng.choice((-1, 1), size=4).tolist())
    if (not isinstance(pivots, (tuple, list)) or len(pivots) != 4
            or any(type(p) is not int or p not in (2 * i, 2 * i + 2) for i, p in enumerate(pivots))):
        raise ValueError("Each RouteStick board must pivot about one of its two neighboring targets")
    if (not isinstance(signs, (tuple, list)) or len(signs) != 4
            or any(type(s) is not int or s not in (-1, 1) for s in signs)):
        raise ValueError("RouteStick board direction_signs must contain four +1/-1 integers")
    return dict(common, pivot_target_indices=list(pivots), direction_signs=list(signs))


def rotate_points(points_xy, pivot_xy, angle):
    """Rotate any planar points around a shared pivot; negate angle to invert."""
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    pivot = np.asarray(pivot_xy, float)
    return (np.asarray(points_xy, float) - pivot) @ rotation.T + pivot


def layout_swing(rest_xy, pivot_xy, parameters, time_s):
    """Apply one common angle to every board and target, preserving all distances."""
    angle = (math.radians(parameters["amplitude_deg"]) * parameters["direction_sign"]
             * math.sin(2 * math.pi * float(time_s) / parameters["period_s"]))
    return rotate_points(rest_xy, pivot_xy, angle), angle


def board_swing(rest_xy, pivot_xy, parameters, time_s):
    """Return board centers and yaw offsets; targets and orbit radii stay fixed."""
    rest, pivots = np.asarray(rest_xy, float), np.asarray(pivot_xy, float)
    angles = (np.radians(parameters["amplitude_deg"])
              * np.asarray(parameters["direction_signs"])
              * np.sin(2 * np.pi * float(time_s) / parameters["period_s"]))
    c, s = np.cos(angles), np.sin(angles)
    delta = rest - pivots
    moved = pivots + np.column_stack((c * delta[:, 0] - s * delta[:, 1],
                                     s * delta[:, 0] + c * delta[:, 1]))
    return moved, angles


def demonstrate_moving_route(env, planner, target, radius, direction):
    """Draw the native demonstration curve while following the rotating layout.

    Each IK waypoint is transformed at its scheduled control-step time. This
    keeps the reference curve attached to its moving endpoints; execution
    policies still have to follow the motion themselves.
    """
    from robomme_plus.common.errors import SceneGenerationError
    from robomme_plus.common.sim import position
    from .controllers import _follow, _ik_path, _poses, rotation
    from .split_stick import tip_position

    motion = env._route_motion_parameters
    pivot = env._route_motion_target_rest_poses[motion["pivot_target_index"]].p[:2]
    start = env._route_to_rest(tip_position(env)[:2])
    end = env._route_to_rest(position(target, False)[:2])
    chord = end - start
    length = np.linalg.norm(chord)
    if length < 1e-6:
        return None
    axis = chord / length
    sign = -1. if str(direction).lower() == "counterclockwise" else 1.
    control = (start + end) / 2 + sign * max(float(radius), 1e-4) * np.array([-axis[1], axis[0]])
    samples = np.r_[np.linspace(0., 1., 45), np.ones(5)]
    seconds = env._route_motion_physics_steps / env.sim_freq
    dt = env._sim_steps_per_control / env.sim_freq
    orientation = rotation(env.agent.tcp)
    moves = []
    for index, u in enumerate(samples):
        xy = (1 - u)**2 * start + 2 * (1 - u) * u * control + u**2 * end
        moved, _ = layout_swing(xy, pivot, motion, seconds + (index + 1) * dt)
        moves.append(([*moved, .07], orientation))
    path, stop = _ik_path(planner, _poses(moves))
    if stop:
        raise SceneGenerationError(f"Moving RouteStick demonstration could not follow curve: {stop}")
    return _follow(planner, path)
