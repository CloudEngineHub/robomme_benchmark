"""Simulator-independent geometry and scoring for the imitation extensions."""

import copy
import math

import numpy as np

from robomme_plus.tasks.imitation.utils.drawing import IDLE_LIMIT_STEPS

# InsertPeg keeps the original hard geometry exactly; easier tiers widen the
# square through-hole. Widths below are full widths, not half extents.
INSERT_HOLE_WIDTH_RANGES_M = {
    "easy": (0.023, 0.025),
    "medium": (0.022, 0.023),
    "hard": (0.0204, 0.0204),
}
PEG_HALF_WIDTH_M = 0.010
HOLE_CLEARANCE_M = 0.0002
HOLE_HALF_SIZE_M = PEG_HALF_WIDTH_M + HOLE_CLEARANCE_M
# The native stand: 10 cm along the hole axis, 8 cm across the entry face, 8 cm
# tall. Every hole keeps at least this much material between it and the edges
# of the face, so its center lies 1.5-6.5 cm high and within 2.5 cm of the
# face's vertical center line.
STAND_HALF_DEPTH_M, STAND_HALF_WIDTH_M, STAND_HEIGHT_M = 0.05, 0.04, 0.08
HOLE_EDGE_MARGIN_M = 0.0048
HOLE_HEIGHT_RANGE_M = (
    HOLE_HALF_SIZE_M + HOLE_EDGE_MARGIN_M,
    STAND_HEIGHT_M - HOLE_HALF_SIZE_M - HOLE_EDGE_MARGIN_M,
)
HOLE_LATERAL_LIMIT_M = STAND_HALF_WIDTH_M - HOLE_HALF_SIZE_M - HOLE_EDGE_MARGIN_M
INSERT_PHASES = ("demonstration", "execution")

PROFILES = {
    # Two stands at the same table position: A in the demonstration, B in
    # execution. Both share one seeded yaw (within +-40 degrees of the native
    # facing). Execution offsets the demonstrated hole by 5-10 mm per axis
    # on the same left/right entry face. See _resolve_insert.
    "insert_peg": {
        "task": "InsertPeg",
        "peg_count": 1,
        "aperture_half_size_m": HOLE_HALF_SIZE_M,
        "height_range_m": [round(v, 6) for v in HOLE_HEIGHT_RANGE_M],
        "lateral_range_m": [
            -round(HOLE_LATERAL_LIMIT_M, 6),
            round(HOLE_LATERAL_LIMIT_M, 6),
        ],
        "hole_noise_half_span_m": 0.010,
        "hole_noise_min_abs_m": 0.005,
        "stand_yaw_half_span_deg": 40.0,
        "shared_stand_orientation": True,
        "success_marker_seconds": 1.0,
        "stand_height_m": STAND_HEIGHT_M,
        "stand_depth_m": 2 * STAND_HALF_DEPTH_M,
        "stand_width_m": 2 * STAND_HALF_WIDTH_M,
    },
    # MoveCube: tool A in the demonstration and tool B (same method) in
    # execution, each at its own seeded resting pose; lengths differ by >= 5 cm
    # and other sizes are redrawn. Both phases sample their own workspace
    # layout; the tool-use method is what transfers between phases.
    "move_hook": {
        "task": "MoveCube",
        "method": "hook",
        "length_range_m": [0.35, 0.50],
        "min_length_difference_m": 0.05,
        "layout": "independent cube, target and tool poses across the workspace",
        "legal_contact_part": "hook head",
    },
    "move_poke": {
        "task": "MoveCube",
        "method": "poke",
        "length_range_m": [0.15, 0.30],
        "min_length_difference_m": 0.05,
        "shaft_radius_m": 0.003,
        "ball_radius_m": 0.016,
        "layout": "independent cube, target and tool poses across the workspace",
        "legal_contact_part": "ball tip",
    },
    "move_scoop_left": {
        "task": "MoveCube",
        "method": "scoop",
        "wall_side": "left",
        "length_range_m": [0.15, 0.25],
        "min_length_difference_m": 0.05,
        "layout": "independent cube, target, tool and oriented wall poses in each phase",
        "legal_contact_part": "pan floor, inner rim faces and underside",
    },
    "move_scoop_right": {
        "task": "MoveCube",
        "method": "scoop",
        "wall_side": "right",
        "length_range_m": [0.15, 0.25],
        "min_length_difference_m": 0.05,
        "layout": "independent cube, target, tool and oriented wall poses in each phase",
        "legal_contact_part": "pan floor, inner rim faces and underside",
    },
    # Tracing defaults; ``resolve_config(..., level)`` sets the path range, the
    # execution stick and the board. The RouteStick keys from
    # ``obstacle_contact_fails`` on apply only when the stick is cut short.
    "pattern_trace": {
        "task": "PatternLock",
        "board_width_m": 0.055,
        "board": True,
        "drawing_height_m": 0.08,
        "drawing_continuity_m": 0.03,
        "drawing_idle_limit_steps": IDLE_LIMIT_STEPS,
        "board_motion": "follow gripper left-right",
        "board_phase": "demonstration only",
        "path_node_range": [9, 15],
        "grid_size": 5,
    },
    "route_trace": {
        "task": "RouteStick",
        "board_width_m": 0.055,
        "board": True,
        "drawing_height_m": 0.08,
        "drawing_continuity_m": 0.03,
        "drawing_idle_limit_steps": IDLE_LIMIT_STEPS,
        "trace_lateral_ratio": 1.3,
        "trace_endpoint_margin_m": 0.03,
        "obstacle_contact_fails": True,  # the lowered hand must turn around the full-height boards
        # Targets 8 cm apart (native 7 cm): turning with the route clears the
        # boards, while any fixed hand yaw on a route as deep as the
        # demonstration's (10 cm) hits one (see route_hand_clearance).
        "target_spacing_m": 0.08,
        "route_depth_m": 0.10,
        "board_motion": "follow gripper left-right",
        "board_phase": "demonstration only",
        "segment_count_range": [8, 12],
        "target_position_count": 9,
    },
}

# The general Scoop case rotates its wall through the full circle. Explicit
# left/right names remain available as directional compatibility variants.
PROFILES["move_scoop"] = dict(PROFILES["move_scoop_left"])
PROFILES["move_scoop"].pop("wall_side")


def recorded_move_config(parameters, profile, seed):
    """Copy a recorded scene without re-running a newer layout sampler."""
    if not isinstance(parameters, dict) or (
        parameters.get("task") != "MoveCube"
        or parameters.get("profile") != profile
        or parameters.get("seed") != seed
        or parameters.get("layout_version") not in (4, 5, 6)
        or profile not in PROFILES
        or parameters.get("method") != PROFILES[profile].get("method")
    ):
        raise ValueError("Recorded MoveCube scene identity or layout version is invalid")
    config = copy.deepcopy(parameters)
    if config["method"] == "scoop" and config["layout_version"] == 4:
        # Version 4 used one axis-aligned wall in both phases. Adapt its
        # representation while preserving every recorded physical pose.
        direction = [0.0, 1.0 if config["wall_side"] == "left" else -1.0]
        for prefix in ("", "execution_"):
            config[prefix + "wall_xy_m"] = list(config["wall_xy_m"])
            config[prefix + "wall_direction_xy"] = direction.copy()
            config[prefix + "wall_yaw_rad"] = 0.0
    return config


def resolve_config(profile, seed, level=None, *, stick_change_version=1,
                   execution_stick_angle_deg=None):
    """Seeded scene parameters; tracing tasks take their path range and board from ``level``."""
    config = dict(PROFILES[profile], profile=profile, seed=int(seed))
    if config['task'] == 'RouteStick':
        if type(stick_change_version) is not int or stick_change_version != 1:
            raise ValueError('RouteStick only supports the straight stick (stick_change_version=1)')
        if execution_stick_angle_deg is not None:
            raise ValueError('execution_stick_angle_deg is only supported for PatternLock')
    config["layout_control"] = "seed"
    config["difficulty_levels"] = level is not None
    if level is not None and config["task"] == "InsertPeg":
        if level not in INSERT_HOLE_WIDTH_RANGES_M:
            raise ValueError("InsertPeg difficulty must be easy, medium or hard")
        config["level"] = level
        if level != "hard":
            width = float(np.random.default_rng(int(seed) + 610140).uniform(
                *INSERT_HOLE_WIDTH_RANGES_M[level]))
            half = width / 2
            config.update(
                aperture_half_size_m=half,
                oracle_grasp_offset_m=INSERT_VISIBLE_GRASP_OFFSET_M,
                height_range_m=[half + HOLE_EDGE_MARGIN_M,
                                STAND_HEIGHT_M - half - HOLE_EDGE_MARGIN_M],
                lateral_range_m=[-STAND_HALF_WIDTH_M + half + HOLE_EDGE_MARGIN_M,
                                 STAND_HALF_WIDTH_M - half - HOLE_EDGE_MARGIN_M],
            )
    elif level is not None:
        from robomme_plus.tasks.imitation.utils.variants import TRACE_LEVELS

        config.update(TRACE_LEVELS[config["task"]][level], level=level)
        if config["task"] == "PatternLock" and stick_change_version in (4, 5, 6):
            from .variants import PATTERN_LEVELS_V4
            config.update(PATTERN_LEVELS_V4[level])
    if config["task"] in ("PatternLock", "RouteStick"):
        from robomme_plus.tasks.imitation.utils.variants import tracing_stick_fraction

        config["execution_stick_fraction"] = tracing_stick_fraction(
            config["task"], seed, stick_change_version=stick_change_version, level=level)
        if config.get("shorten_stick") is False:
            config["execution_stick_fraction"] = 1.0
        if config['task'] == 'PatternLock' and stick_change_version == 2:
            from .variants import tracing_stick_tilt
            config.update(stick_change_version=2, drawing_max_angle_deg=8.0,
                          execution_stick_tilt_deg=(tracing_stick_tilt(seed)
                                                    if config['execution_stick_fraction'] < 1 else None))
        if config['task'] == 'PatternLock' and stick_change_version == 3:
            from .variants import pattern_stick_tilt
            tilt = pattern_stick_tilt(seed, execution_stick_angle_deg)
            config.update(stick_change_version=3, drawing_max_angle_deg=8.0,
                          execution_stick_tilt_deg=tilt if config['execution_stick_fraction'] < 1 else None,
                          execution_stick_angle_deg=tilt[0])
        if config['task'] == 'PatternLock' and stick_change_version in (4, 5, 6):
            from .variants import pattern_level_tilt
            tilt = pattern_level_tilt(seed, level, execution_stick_angle_deg)
            config.update(stick_change_version=stick_change_version, drawing_max_angle_deg=8.0,
                          execution_stick_tilt_deg=tilt if config['execution_stick_fraction'] < 1 else None,
                          execution_stick_angle_deg=tilt[0])
    if "path_node_range" in config:
        low, high = config["path_node_range"]
        config["path_node_count"] = int(
            np.random.default_rng(int(seed) + 610134).integers(low, high + 1)
        )
    if "segment_count_range" in config:
        low, high = config["segment_count_range"]
        config["segment_count"] = int(
            np.random.default_rng(int(seed) + 610135).integers(low, high + 1)
        )
    if config["task"] == "InsertPeg":
        _resolve_insert(config, int(seed))
    if config["task"] == "MoveCube":
        _resolve_move(config, int(seed))
    return config


# Panda base and shoulder (``_load_agent`` pose plus joint 2 height).
ROBOT_BASE_XY = (-0.615, 0.0)
# Horizontal TCP distance from the robot base at the oracle's key tool poses.
# This is a coarse horizontal reach filter. Orientation, dynamics and grasp
# stability still need simulator rollout verification.
TCP_REACH_RANGE_M = (0.30, 0.78)
CUBE_HALF_M = 0.02
# MoveCube layouts are checked against tools spanning the whole length range
# (``REACH_LENGTH_SAMPLES`` lengths) and the extremes of the head sizes that
# move the oracle's hand, so a layout never depends on either tool's size.
REACH_LENGTH_SAMPLES = 6
HEAD_SIZE_EXTREMES = {
    "hook": {"hook_head_scale": (0.90, 1.15)},
    "poke": {"ball_radius_m": (0.013, 0.019)},
    "scoop": {"pan_length_m": (0.10, 0.12)},
}
# The same broad workspace is offered to every method. Rejections follow
# reach/clearance geometry, without assigning cubes or targets to strips.
MOVE_CUBE_BOUNDS_M = ((-0.22, 0.30), (-0.28, 0.28))
MOVE_TARGET_BOUNDS_M = ((-0.25, 0.28), (-0.28, 0.28))
MOVE_TABLE_BOUNDS_M = ((-0.45, 0.42), (-0.36, 0.36))
REST_HANDLE_BOUNDS_M = ((-0.34, 0.20), (-0.30, 0.30))
REST_MIN_SHIFT_M = 0.05
# Smallest gaps between a resting tool's table footprint and the cube (its
# corners), the target disc, the band the cube sweeps on its way (to the target,
# or to the wall for scoop), and the wall.
REST_CLEARANCE_M = {"cube": 0.04, "target": 0.01, "path": 0.02, "wall": 0.04}
TARGET_RADIUS_M = 0.04
WALL_HALF_M = (0.12, 0.01)  # local wall footprint: length, thickness


def _tool_sizes(method, size_rng):
    """One tool's grip, head, and shaft sizes (lengths are drawn separately)."""
    # A long hook needs a broader grip to resist its head's torque.
    grip_width = (0.013, 0.016) if method == "hook" else (0.009, 0.011)
    grip_height = (0.011, 0.014) if method == "hook" else (0.009, 0.011)
    tool = {
        "handle_half_size_m": [
            0.035,
            float(size_rng.uniform(*grip_width)),
            float(size_rng.uniform(*grip_height)),
        ]
    }
    if method == "hook":
        tool["hook_head_scale"] = float(size_rng.uniform(0.90, 1.15))
        tool["shaft_radius_m"] = float(size_rng.uniform(0.005, 0.007))
    elif method == "poke":
        tool["ball_radius_m"] = float(size_rng.uniform(0.013, 0.019))
        tool["shaft_radius_m"] = float(size_rng.uniform(0.0025, 0.0035))
    else:
        tool["pan_length_m"] = float(size_rng.uniform(0.10, 0.12))
        tool["pan_width_m"] = float(size_rng.uniform(0.09, 0.11))
        tool["shaft_radius_m"] = float(size_rng.uniform(0.005, 0.007))
    return tool


def scoop_push_distance_m(cube_xy, wall_xy, direction):
    """Signed cube-to-wall-face gap along an arbitrary horizontal normal."""
    return float((np.asarray(wall_xy) - cube_xy) @ direction - 0.01 - CUBE_HALF_M)


def poke_shaft_axis(direction):
    """Perpendicular probe axis, choosing the sign that keeps the hand nearer the robot."""
    d = np.asarray(direction, float)[:2]
    axis = np.array([d[1], -d[0]])
    if axis[0] < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def oracle_tcp_waypoints(method, cube, target, tool, length, wall=None, direction=None):
    """Approximate hand xy along contact, withdrawal and transport, for any direction."""
    cube, target = np.asarray(cube, float), np.asarray(target, float)
    if method == "hook":
        d = (target - cube) / np.linalg.norm(target - cube)
        scale = tool["hook_head_scale"]
        across = np.array([d[1], -d[0]])
        return [
            cube + d * (length - 0.024 - 0.006 * scale) - across * 0.060 * scale,
            target + d * (length - 0.011 - 0.006 * scale) - across * 0.060 * scale,
        ]
    if method == "poke":
        d = (target - cube) / np.linalg.norm(target - cube)
        axis = poke_shaft_axis(d)
        ball = tool["ball_radius_m"]
        return [
            cube - d * (CUBE_HALF_M + ball + 0.003) - axis * length,
            target - d * (CUBE_HALF_M + ball - 0.009) - axis * length,
        ]
    d = np.asarray(direction, float)
    pan_half = tool["pan_length_m"] / 2
    extent = CUBE_HALF_M * np.abs(d).sum()
    start = cube - d * (extent + pan_half + 0.025 + length)
    end = cube + d * (float((np.asarray(wall) - cube) @ d) - .026 - length)
    return [
        start,
        end,
        end - d * (pan_half + .020),
        target - d * (length + pan_half + .045),
    ]


def _reachable(points):
    low, high = TCP_REACH_RANGE_M
    p = np.asarray(points) - ROBOT_BASE_XY
    distances = np.linalg.norm(p, axis=1)
    if np.any(distances < low) or np.any(distances > high):
        return False
    # A segment with reachable endpoints can still cross the inner exclusion.
    delta = np.diff(p, axis=0)
    t = np.clip(
        -np.sum(p[:-1] * delta, axis=1)
        / np.maximum(np.sum(delta * delta, axis=1), 1e-12), 0, 1
    )
    return bool(np.all(np.linalg.norm(p[:-1] + delta * t[:, None], axis=1) >= low))


def reach_envelope(config):
    """Tools across the length range and the head-size extremes of one method."""
    low, high = config["length_range_m"]
    heads = [{}]
    for key, values in HEAD_SIZE_EXTREMES[config["method"]].items():
        heads = [dict(head, **{key: value}) for head in heads for value in values]
    return [
        dict(head, tool_length_m=float(length))
        for length in np.linspace(low, high, REACH_LENGTH_SAMPLES)
        for head in heads
    ]


def layout_reachable(config, cube, target, wall=None, direction=None):
    """Whether every tool in the envelope reaches the oracle's key poses."""
    return all(
        _reachable(
            oracle_tcp_waypoints(
                config["method"],
                cube,
                target,
                tool,
                tool["tool_length_m"],
                wall,
                direction,
            )
        )
        for tool in reach_envelope(config)
    )


def _move_lengths(config, seed):
    """Tool A and tool B lengths: two independent draws at least the minimum apart."""
    rng = np.random.default_rng(seed + 610131)
    low, high = config["length_range_m"]
    for _ in range(1000):
        lengths = rng.uniform(low, high, 2)
        if abs(lengths[0] - lengths[1]) >= config["min_length_difference_m"]:
            return float(lengths[0]), float(lengths[1])
    raise ValueError("Could not sample two distinct MoveCube tool lengths")


def _move_layout(config, seed):
    """Independent phase layouts, without fixed regions or a required side switch.

    The sampling streams and tool-envelope checks do not read the drawn tool
    sizes. Changing tool A or B therefore does not move the cube or target.
    """
    method = config["method"]
    minimum = 0.12
    layout = {"minimum_cube_target_distance_m": minimum}
    for phase, prefix in enumerate(("", "execution_")):
        rng = np.random.default_rng(seed + 610140 + phase * 100003)
        direction = None
        if method == "scoop":
            # Draw the normal once, before rejection sampling. Otherwise the
            # larger feasible area for side-facing walls overwhelms front/back
            # cases and produces the same directional bias as fixed walls.
            angle = rng.uniform(-np.pi, np.pi)
            if config.get("wall_side") in ("left", "right"):
                angle = np.pi / 2 if config["wall_side"] == "left" else -np.pi / 2
            direction = np.array([np.cos(angle), np.sin(angle)])
        for _ in range(10000):
            cube = rng.uniform(*np.asarray(MOVE_CUBE_BOUNDS_M).T)
            target = rng.uniform(*np.asarray(MOVE_TARGET_BOUNDS_M).T)
            if math.dist(cube, target) < minimum:
                continue
            if phase and (
                math.dist(cube, layout["cube_xy_m"]) < 0.08
                or math.dist(target, layout["target_xy_m"]) < 0.08
            ):
                continue
            extra, wall = {}, None
            if method == "scoop":
                tangent = np.array([direction[1], -direction[0]])
                gap = rng.uniform(0.08, 0.24)
                wall = (cube + direction * (gap + 0.01 + CUBE_HALF_M)
                        + tangent * rng.uniform(-0.035, 0.035))
                wall_yaw = angle - np.pi / 2
                wall_rect = _footprint(wall, tangent, WALL_HALF_M)
                if (
                    not inside_move_table(_footprint_points(wall_rect, 2))
                    or _footprint_distance(target, wall_rect) < TARGET_RADIUS_M + 0.04
                    or math.dist(target, cube + direction * gap) < 0.15
                ):
                    continue
                extra = {
                    "wall_xy_m": wall.tolist(), "wall_direction_xy": direction.tolist(),
                    "wall_yaw_rad": float(wall_yaw), "cube_wall_gap_m": float(gap),
                }
            if not layout_reachable(config, cube, target, wall, direction):
                continue
            layout.update({
                prefix + "cube_xy_m": cube.tolist(),
                prefix + "target_xy_m": target.tolist(),
                **{prefix + key: value for key, value in extra.items()},
            })
            break
        else:
            raise ValueError(f"Could not sample a reachable {method} phase {phase} layout")
    return layout


def inside_move_table(points):
    """All footprint vertices stay inside the tabletop's working bounds."""
    bounds = np.asarray(MOVE_TABLE_BOUNDS_M)
    points = np.asarray(points)
    return bool(np.all(points >= bounds[:, 0]) and np.all(points <= bounds[:, 1]))


def _rect_distance(points, rect):
    """Signed distances from points to a footprint rectangle (negative inside)."""
    center, u, v, (hu, hv) = rect
    d = np.atleast_2d(np.asarray(points, float)) - center
    dx, dy = np.abs(d @ u) - hu, np.abs(d @ v) - hv
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return np.where(outside > 0, outside, np.maximum(dx, dy))


def tool_footprint(handle_xy, yaw, tool, method):
    """Table footprint of a resting tool: handle, shaft, and head rectangles."""
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    normal = np.array([-axis[1], axis[0]])
    origin = np.asarray(handle_xy, float)

    def rect(along, across, half_along, half_across):
        return _footprint(
            origin + axis * along + normal * across, axis, (half_along, half_across)
        )

    length, radius = tool["tool_length_m"], tool["shaft_radius_m"]
    parts = [rect(0, 0, *tool["handle_half_size_m"][:2])]
    if method == "hook":
        s = tool["hook_head_scale"]
        parts += [
            rect(length / 2, 0, length / 2, radius),
            rect(length - 0.022 * s, 0.0475 * s, 0.028 * s, 0.0535 * s),
        ]
    elif method == "poke":
        ball = tool["ball_radius_m"]
        parts += [rect(length / 2, 0, length / 2, radius), rect(length, 0, ball, ball)]
    else:
        pan_half, shaft = (
            tool["pan_length_m"] / 2,
            tool["tool_length_m"] - tool["pan_length_m"] / 2,
        )
        parts += [
            rect(shaft / 2, 0, shaft / 2, radius),
            rect(length, 0, pan_half, tool["pan_width_m"] / 2 + 0.002),
        ]
    return parts


def rest_clearance(parts, cube, target, path, wall=None, wall_yaw=0.0):
    """Gaps (m) between a resting tool's footprint and the cube's corners, the
    target disc, the band swept by the cube along ``path`` segments, and the wall."""

    def gap(points, radius):
        return float(min(_rect_distance(points, part).min() for part in parts) - radius)

    samples = np.concatenate(
        [np.linspace(a, b, max(2, int(math.dist(a, b) / 0.005) + 1)) for a, b in path]
    )
    gaps = {
        "cube": gap(cube, CUBE_HALF_M * math.sqrt(2)),
        "target": gap(target, TARGET_RADIUS_M),
        "path": gap(samples, CUBE_HALF_M),
    }
    if wall is not None:
        wall_rect = _footprint(wall, (math.cos(wall_yaw), math.sin(wall_yaw)), WALL_HALF_M)
        gaps["wall"] = float(
            min(
                min(
                    _rect_distance(_footprint_points(part), wall_rect).min()
                    for part in parts
                ),
                gap(_footprint_points(wall_rect), 0.0),
            )
        )
    return gaps


def _rest_pose(
    rng, method, tool, cube, target, path, wall=None, wall_yaw=0.0, other_handle=None
):
    """A seeded resting pose (handle xy, yaw) clear of the scene, and its gaps."""
    bounds = np.asarray(REST_HANDLE_BOUNDS_M)
    for _ in range(4096):
        handle = rng.uniform(bounds[:, 0], bounds[:, 1])
        yaw = rng.uniform(-np.pi, np.pi)
        if not _reachable([handle]):
            continue
        if (
            other_handle is not None
            and math.dist(handle, other_handle) < REST_MIN_SHIFT_M
        ):
            continue
        parts = tool_footprint(handle, yaw, tool, method)
        if not all(inside_move_table(_footprint_points(part, 2)) for part in parts):
            continue
        gaps = rest_clearance(parts, cube, target, path, wall, wall_yaw)
        if all(gaps[key] >= REST_CLEARANCE_M[key] for key in gaps):
            return [float(v) for v in handle], float(yaw), gaps
    raise ValueError("Could not place the MoveCube tool clear of the scene")


def _resolve_move(config, seed):
    """MoveCube: a demonstration tool (A), an execution tool (B), and both phases' layouts.

    Both tools share the method; their lengths differ by at least
    ``min_length_difference_m`` and every other size is redrawn. The layout
    (``_move_layout``) reads neither tool. Each tool then rests at its own
    seeded pose (stream ``seed + 610139``), with a full-circle heading, clear
    of that phase's cube, target, cube path, and oriented wall. Tool B's handle
    rests at least ``REST_MIN_SHIFT_M`` from tool A's.
    """
    method = config["method"]
    # Keep size draws independent of length/layout rejection sampling.
    size_rng = np.random.default_rng(seed + 610137)
    config["tool_size_version"] = 2
    demo_tool, exec_tool = _tool_sizes(method, size_rng), _tool_sizes(method, size_rng)
    if method == "hook":
        config["hook_head_scale_range"] = [0.90, 1.15]
    elif method == "poke":
        config["ball_radius_range_m"] = [0.013, 0.019]
    else:
        config["pan_length_range_m"] = [0.10, 0.12]
        config["pan_width_range_m"] = [0.09, 0.11]
    demo_tool["tool_length_m"], exec_tool["tool_length_m"] = _move_lengths(config, seed)
    layout = _move_layout(config, seed)
    rest_rng = np.random.default_rng(seed + 610139)
    rests = {}
    for phase, tool, prefix in (
        ("demonstration", demo_tool, ""),
        ("execution", exec_tool, "execution_"),
    ):
        cube, target = layout[prefix + "cube_xy_m"], layout[prefix + "target_xy_m"]
        wall = layout.get(prefix + "wall_xy_m")
        wall_yaw = layout.get(prefix + "wall_yaw_rad", 0.0)
        if wall is None:
            path = [(cube, target)]
        else:
            direction = np.asarray(layout[prefix + "wall_direction_xy"])
            path = [(cube, np.asarray(cube) + direction * layout[prefix + "cube_wall_gap_m"])]
        other = rests["demonstration"][0] if rests else None
        rests[phase] = _rest_pose(
            rest_rng, method, tool, cube, target, path, wall, wall_yaw, other
        )
    # Top-level tool keys describe the demonstration tool (A).
    config.update(demo_tool)
    config.update(layout)
    yaw_rng = np.random.default_rng(seed + 610149)
    config["cube_yaw_rad"] = float(yaw_rng.uniform(-np.pi, np.pi))
    config["execution_cube_yaw_rad"] = float(yaw_rng.uniform(-np.pi, np.pi))
    config.update(
        execution_tool=exec_tool,
        handle_xy_m=rests["demonstration"][0],
        handle_yaw_rad=rests["demonstration"][1],
        execution_handle_xy_m=rests["execution"][0],
        execution_handle_yaw_rad=rests["execution"][1],
        tool_rest_clearance_m={phase: rest[2] for phase, rest in rests.items()},
        layout_version=6,
    )


# Scoop pan, in the pan head's frame (x toward the open lip, z up; see
# ``tools.build_tool_articulation``): a wedge floor 5 mm thick at the back and
# 0.2 mm at the lip, and three rims 4 mm thick and 3 cm tall (back, two sides).
PAN_RIM_THICKNESS_M = 0.004
PAN_RIM_TOP_M = 0.028
PAN_FLOOR_TOP_M = 0.003  # at the back rim; the floor slopes down to the lip
PAN_CONTACT_TOLERANCE_M = 0.0015
LEGAL_PAN_REGIONS = ("floor", "inner rim", "underside")


def pan_contact_region(
    point, normal, pan_length, pan_width, tolerance=PAN_CONTACT_TOLERANCE_M
):
    """Which pan surface a cube contact touches.

    ``point`` is the contact point and ``normal`` the contact normal pointing
    from the pan into the cube, both in the pan head's frame. The cube may touch
    the pan floor (including the lip, the floor's leading edge), the inner faces
    of the back and side rims, and the underside. An outer rim face, a rim top,
    and a side rim's front end are illegal, like the shaft or the handle.

    The normal tells the faces of one rim apart (inner faces point into the
    cavity, outer faces away from it); the point only separates surfaces that
    lie centimetres apart (which rim; floor or rim top; lip or rim end), so the
    few millimetres a moving pan travels within one physics step do not matter.
    """
    x, y, z = (float(v) for v in point)
    nx, ny, nz = np.asarray(normal, float) / max(float(np.linalg.norm(normal)), 1e-12)
    if abs(nz) >= max(abs(nx), abs(ny)):
        if nz < 0:
            return "underside"
        return "rim top" if z > (PAN_RIM_TOP_M + PAN_FLOOR_TOP_M) / 2 else "floor"
    if abs(ny) > abs(nx):
        # A side rim's inner face points toward the pan's axis.
        return "inner rim" if ny * y < 0 else "outer rim"
    if nx < 0:
        return "outer rim"  # the back rim, from behind
    if x < 0:
        return "inner rim"  # the back rim, from inside
    # Facing forward at the open front: the lip between the side rims, or the
    # front end of a side rim.
    inner_side = pan_width / 2 - PAN_RIM_THICKNESS_M / 2
    at_lip_height = z <= PAN_FLOOR_TOP_M + tolerance
    return "floor" if at_lip_height and abs(y) <= inner_side + tolerance else "rim front end"


def insertion_choice(seed, available_pegs):
    """Choose one demonstrated peg and grasp end without changing native RNG."""
    if available_pegs < 1:
        raise ValueError("Insertion needs at least one available peg")
    rng = np.random.default_rng(int(seed) + 610132)
    return int(rng.integers(available_pegs)), str(rng.choice(["head", "tail"]))


def _resolve_insert(config, seed):
    """Seeded yaw and hole center of stand A (demonstration) and stand B (execution).

    Both stands share the table position, orientation and native left/right
    entry face. Sample the demonstration hole across the full allowed face,
    then sample execution offsets uniformly from [-10, -5] or [5, 10] mm
    in each of the face's lateral and vertical axes. Redraw only the noise
    outside those bands or the stand limits, preserving the demonstration
    distribution without clipping holes onto the boundaries.
    """
    hole_rng = np.random.default_rng(seed + 610138)
    low = np.array([config["lateral_range_m"][0], config["height_range_m"][0]])
    high = np.array([config["lateral_range_m"][1], config["height_range_m"][1]])
    demonstration = hole_rng.uniform(low, high)
    noise_span = config["hole_noise_half_span_m"]
    min_offset = config["hole_noise_min_abs_m"]
    for _ in range(1000):
        offset = hole_rng.uniform(-noise_span, noise_span, 2)
        if np.any(np.abs(offset) < min_offset):
            continue
        execution = demonstration + offset
        if np.all(execution >= low) and np.all(execution <= high):
            break
    else:
        raise ValueError("Could not sample an InsertPeg hole offset within the stand")
    yaw_rng = np.random.default_rng(seed + 610139)
    span = config["stand_yaw_half_span_deg"]
    yaw = float(yaw_rng.uniform(-span, span))
    config["stands"] = {
        phase: {
            "yaw_offset_deg": float(yaw),
            "hole_lateral_m": float(hole[0]),
            "hole_height_m": float(hole[1]),
        }
        for phase, hole in zip(INSERT_PHASES, (demonstration, execution))
    }


def insertion_stand_boxes(
    hole_height,
    hole_lateral,
    aperture=HOLE_HALF_SIZE_M,
    half_depth=STAND_HALF_DEPTH_M,
    half_width=STAND_HALF_WIDTH_M,
    height=STAND_HEIGHT_M,
):
    """``(center, half_size)`` of the four walls around a straight square through-hole.

    Stand frame: origin at the center of the footprint on the table, x along the
    hole (out of the entry face), y across the face, z up. The walls below and
    above the hole span its width; the two side walls fill the rest of the face.
    """
    low, high = hole_lateral - aperture, hole_lateral + aperture
    bottom, top = hole_height - aperture, hole_height + aperture
    boxes = []
    for y0, y1, z0, z1 in (
        (-half_width, low, 0.0, height),
        (high, half_width, 0.0, height),
        (low, high, 0.0, bottom),
        (low, high, top, height),
    ):
        if y1 <= y0 or z1 <= z0:
            raise ValueError("The hole must lie inside the stand face")
        boxes.append(
            (
                [0.0, (y0 + y1) / 2, (z0 + z1) / 2],
                [half_depth, (y1 - y0) / 2, (z1 - z0) / 2],
            )
        )
    return boxes


def insertion_stand_frame(yaw_offset_deg, direction):
    """Stand rotation, turned ``yaw_offset_deg`` from the native facing (yaw pi/2).

    Its x axis is the outward normal of the entry face on the ``direction``
    side (-1 left, +1 right), which is also the hole axis.
    """
    return frame_from_axis(
        outward_axis(math.pi / 2 + math.radians(yaw_offset_deg), direction)
    )


def insertion_hole(stand_xy, stand, direction):
    """Hole center on the stand's middle plane and the outward hole axis."""
    frame = insertion_stand_frame(stand["yaw_offset_deg"], direction)
    center = np.array([*stand_xy[:2], 0.0]) + frame @ [
        0.0,
        stand["hole_lateral_m"],
        stand["hole_height_m"],
    ]
    return center, frame[:, 0].copy()


# InsertPeg oracle: the peg tip's standoffs along the hole axis, measured from
# the hole center on the stand's middle plane (the standoff, first lifted
# above the stand and then at hole height; 1 cm outside the entry face; the
# insertion depth). The fingers close 5 mm from the grasped link's center
# toward the peg's middle (``INSERT_GRASP_OFFSET_M``), close to the center of
# mass, so the peg hardly pivots in the grasp; the TCP is then 4.5 cm beyond
# the tip link's center (the link centers are 5 cm apart). At the insertion
# depth the hand stays 9 mm clear of the entry face.
INSERT_GRASP_OFFSET_M = -0.005
# Easier demonstrations grip 10 mm farther toward the selected end: 30 mm
# from the full peg's midpoint, leaving 20 mm to that end. Hard keeps the
# legacy offset and wrist choice above.
INSERT_VISIBLE_GRASP_OFFSET_M = 0.005
INSERT_STANDOFF_M, INSERT_LIFT_M = 0.15, 0.12
INSERT_TIP_OFFSETS_M = (INSERT_STANDOFF_M, 0.085, 0.035)
INSERT_TCP_BEYOND_TIP_M = 0.05 + INSERT_GRASP_OFFSET_M
# Horizontal TCP distance from the robot base at those poses: Panda IK for the
# oracle's level-peg hand orientations succeeds up to about 0.79 m.
INSERT_REACH_RANGE_M = (0.30, 0.76)
# Native stand position: +-10 cm around the table center in x and y.
STAND_CENTER_RANGE_M = 0.10


def insertion_tcp_points(hole, outward, grasp_offset_m=INSERT_GRASP_OFFSET_M):
    """The oracle's key TCP positions at one stand (see ``INSERT_TIP_OFFSETS_M``)."""
    tcp_beyond_tip = 0.05 + grasp_offset_m
    points = [
        hole + outward * (offset + tcp_beyond_tip)
        for offset in INSERT_TIP_OFFSETS_M
    ]
    return np.array(points + [points[0] + [0.0, 0.0, INSERT_LIFT_M]])


def insertion_reachable(stand_xy, stands, direction, grasp_offset_m=INSERT_GRASP_OFFSET_M):
    """Whether every key TCP position at every stand lies within ``INSERT_REACH_RANGE_M``."""
    low, high = INSERT_REACH_RANGE_M
    for stand in stands:
        distance = np.linalg.norm(
            insertion_tcp_points(*insertion_hole(stand_xy, stand, direction), grasp_offset_m)[:, :2]
            - ROBOT_BASE_XY,
            axis=1,
        )
        if distance.min() < low or distance.max() > high:
            return False
    return True


def insertion_stand_center(seed, stands, direction, grasp_offset_m=INSERT_GRASP_OFFSET_M):
    """Table position shared by both stands, in the native +-10 cm square.

    Drawn from its own stream and redrawn until the oracle can reach both
    stands; the stand yaws and holes stay as drawn. ``direction`` is the
    native entry side (-1 left, +1 right), known only once the scene starts.
    """
    rng = np.random.default_rng(int(seed) + 610140)
    for _ in range(1000):
        xy = rng.uniform(-STAND_CENTER_RANGE_M, STAND_CENTER_RANGE_M, 2)
        if insertion_reachable(xy, stands, direction, grasp_offset_m):
            return xy
    raise ValueError("Could not place the InsertPeg stands within reach")


# Distractor pegs keep 6 cm from each stand footprint and 3 cm from its
# approach corridor: the strip in front of the entry face that the held peg and
# the fingers sweep at hole height, out to 23 cm from the stand's middle plane.
PEG_STAND_CLEARANCE_M = 0.06
APPROACH_CORRIDOR_M = (
    (0.05, 0.23),
    0.04,
)  # along the hole axis; half width across the face
PEG_CORRIDOR_CLEARANCE_M = 0.03
# The Panda hand over a top grasp of either peg end (TCP: see
# ``INSERT_GRASP_OFFSET_M``; footprint ``HAND_FOOTPRINT_HALF_M`` along and across the peg)
# keeps this far from every stand: at a grasp it is below the stand top.
HAND_STAND_CLEARANCE_M = 0.015
_HAND_GRID = np.array(
    [[x, y] for x in np.linspace(-1, 1, 5) for y in np.linspace(-1, 1, 21)]
)


def _rect_gap(local, center, half):
    return np.linalg.norm(np.maximum(np.abs(local - center) - half, 0), axis=1).min()


def grasp_hand_points(root, axis, grasp_offset_m=INSERT_GRASP_OFFSET_M):
    """Points (1 cm apart) covering the hand's footprint at both grasp points of a peg."""
    across = np.array([-axis[1], axis[0]])
    grid = _HAND_GRID * HAND_FOOTPRINT_HALF_M
    return np.concatenate(
        [
            root + offset * axis + grid[:, :1] * axis + grid[:, 1:] * across
            for offset in (grasp_offset_m, -0.05 - grasp_offset_m)
        ]
    )  # head, tail


def insertion_peg_roots(seed, stand_xy, stand_yaws, peg_yaws, grasp_offset_m=INSERT_GRASP_OFFSET_M):
    """Peg roots that leave room to grasp either end beside every stand.

    ``stand_yaws`` are the yaws of the stands' outward hole axes (shared
    center, one yaw per stand); no peg lies in a stand's approach corridor,
    and the hand grasping either end of a peg clears every stand.
    """
    rng = np.random.default_rng(int(seed) + 610133)
    rotations = [
        frame_from_axis([math.cos(yaw), math.sin(yaw), 0])[:2, :2]
        for yaw in np.atleast_1d(stand_yaws)
    ]
    (near, far), corridor_half = APPROACH_CORRIDOR_M
    placed, occupied = [], []
    for yaw in peg_yaws:
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        for _ in range(2000):
            root = rng.uniform([-0.32, -0.28], [-0.05, 0.28])
            samples = root + np.linspace(-0.075, 0.025, 7)[:, None] * axis
            locals_ = [(samples - stand_xy) @ r for r in rotations]
            if any(
                _rect_gap(local, [0.0, 0.0], [STAND_HALF_DEPTH_M, STAND_HALF_WIDTH_M])
                < PEG_STAND_CLEARANCE_M
                or _rect_gap(
                    local, [(near + far) / 2, 0.0], [(far - near) / 2, corridor_half]
                )
                < PEG_CORRIDOR_CLEARANCE_M
                for local in locals_
            ):
                continue
            hand = grasp_hand_points(root, axis, grasp_offset_m)
            if any(
                _rect_gap(
                    (hand - stand_xy) @ r,
                    [0.0, 0.0],
                    [STAND_HALF_DEPTH_M, STAND_HALF_WIDTH_M],
                )
                < HAND_STAND_CLEARANCE_M
                for r in rotations
            ):
                continue
            if any(
                np.linalg.norm(samples[:, None] - other[None, :], axis=2).min() < 0.055
                for other in occupied
            ):
                continue
            placed.append(root.tolist())
            occupied.append(samples)
            break
        else:
            raise ValueError("Could not place pegs with grasp clearance")
    return placed


# RouteStick: footprints at board-top height. The hand is the convex hull of the
# panda_hand collision mesh sliced at its lowest 1.3 cm (the part below the board
# tops while the short tip draws); half extents along its local x and y.
HAND_FOOTPRINT_HALF_M = (0.021, 0.101)
BOARD_FOOTPRINT_HALF_M = (0.015, 0.03)  # native board: 3 cm along the row, 6 cm across


# The upright hand may leave and reach a target with a horizontal yaw offset
# from straight across the row; this yaw adjustment fades in or
# out over ``ROUTE_YAW_BLEND`` of the route at each end.
ROUTE_YAW_OFFSETS_DEG = (0, -10, 10, -20, 20)
ROUTE_YAW_BLEND = 0.25


def _ramp(k, fraction):
    w = np.clip(k / fraction, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * w)


def half_ellipse_route(
    start, end, depth, counterclockwise, count=101, start_yaw_offset=0.0, end_yaw_offset=0.0
):
    """Tip points from ``start`` to ``end`` around the board between them, and the
    hand yaw change (from its yaw at ``start``) that keeps the hand's long side
    along the direction of travel.

    The bulge is on the right of start->end for counterclockwise (the native
    ``solve_swingonto_withDirection`` convention), so the yaw turns by +pi
    (counterclockwise) or -pi over the route. ``start_yaw_offset`` is how far the
    hand's yaw at ``start`` is off straight across the row (removed over the
    first quarter of the route); ``end_yaw_offset`` is its arrival yaw offset.
    """
    start, end = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
    half = float(np.linalg.norm(end - start)) / 2
    u = (end - start) / (2 * half)
    n = np.array([u[1], -u[0]]) if counterclockwise else np.array([-u[1], u[0]])
    ts = np.linspace(math.pi, 0, count)
    xy = (
        (start + end) / 2
        + np.outer(half * np.cos(ts), u)
        + np.outer(depth * np.sin(ts), n)
    )
    travel = np.outer(half * np.sin(ts), u) - np.outer(depth * np.cos(ts), n)
    yaw = np.unwrap(np.arctan2(travel[:, 1], travel[:, 0]))
    k = np.linspace(0, 1, count)
    yaw_offset = -start_yaw_offset * _ramp(k, ROUTE_YAW_BLEND) + end_yaw_offset * _ramp(
        k - (1 - ROUTE_YAW_BLEND), ROUTE_YAW_BLEND
    )
    return xy, yaw - yaw[0] + yaw_offset


def _footprint(center, axis, half):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.asarray(center, float)[:2], u, np.array([-u[1], u[0]]), half


def _footprint_distance(point, rect):
    """Signed distance from a point to a footprint (negative inside)."""
    center, u, v, (hu, hv) = rect
    d = np.asarray(point, float) - center
    dx, dy = abs(d @ u) - hu, abs(d @ v) - hv
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    return outside if outside > 0 else max(dx, dy)


def _footprint_points(rect, n=8):
    center, u, v, (hu, hv) = rect
    edge = np.linspace(-1, 1, n)
    return [center + u * hu * a + v * hv * b for a in edge for b in (-1, 1)] + [
        center + u * hu * a + v * hv * b for a in (-1, 1) for b in edge
    ]


def lateral_occluder_y(eye, gripper, board_x):
    """Intersect the camera-to-gripper ray with the fixed board x plane."""
    fraction = (board_x - eye[0]) / (gripper[0] - eye[0])
    return float(eye[1] + fraction * (gripper[1] - eye[1]))


def frame_from_axis(axis):
    """Orthonormal tool frame whose x axis follows an arbitrary hole axis."""
    x = np.asarray(axis, dtype=float)
    x = x / np.linalg.norm(x)
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    )
    y = np.cross(reference, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def outward_axis(yaw, direction):
    """Horizontal outward normal of the face on the ``direction`` side of a stand with ``yaw``."""
    axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    if axis[1] * direction > 0:
        axis *= -1
    return axis


def insertion_metrics(tip, grasp, center, outward):
    tip, grasp, center, outward = map(np.asarray, (tip, grasp, center, outward))
    offset = tip - center
    axial = float(offset @ outward)
    radial = float(np.linalg.norm(offset - axial * outward))
    # Offsets along the square tunnel's two cross-section axes (across the
    # stand face, and vertical).
    frame = frame_from_axis(outward)
    shaft = grasp - tip
    alignment = float(shaft @ outward / max(np.linalg.norm(shaft), 1e-9))
    return {
        "axial_m": axial,
        "radial_m": radial,
        "alignment": alignment,
        "lateral_m": float(offset @ frame[:, 1]),
        "normal_m": float(offset @ frame[:, 2]),
        "grasp_side_m": float((grasp - center) @ outward),
    }


def insertion_valid(metrics, aperture, peg_radius, depth):
    # The inserting link's center must enter the physical tunnel: at least
    # 1 cm deep, and off the axis by less than the clearance between the
    # square peg and the square hole along each cross-section axis. A nearby
    # perpendicular peg or a peg approaching the opposite opening cannot pass.
    clearance = aperture - peg_radius
    return (
        0 <= metrics["axial_m"] <= min(0.045, depth - 0.01)
        and max(abs(metrics["lateral_m"]), abs(metrics["normal_m"])) < clearance
        and metrics["alignment"] > math.cos(math.radians(8))
        and metrics["grasp_side_m"] > metrics["axial_m"]
    )


def move_failure_code(method, evidence):
    if evidence.get("cube_grasped"):
        return "robot_grasped_cube"
    if evidence.get("gripper_contact") or evidence.get("robot_contact_steps", 0):
        return "robot_or_gripper_touched_cube"
    if (evidence.get("shaft_contact_steps", 0) or evidence.get("handle_contact_steps", 0)
            or evidence.get("illegal_pan_contact_steps", 0)):
        return "illegal_tool_surface_touched_cube"
    if method != "scoop" and evidence.get("cube_lifted"):
        return "cube_lifted_for_non_scoop_method"
    return None


def move_failed(method, evidence):
    return move_failure_code(method, evidence) is not None


def move_complete(method, evidence, near_target, on_table, released):
    if move_failed(method, evidence):
        return False
    if not (near_target and on_table and released):
        return False
    if method == "scoop":
        return (
            evidence.get("wall_contact_steps", 0) >= 2
            and evidence.get("supported_lift_steps", 0) >= 3
            and evidence.get("carried_distance_m", 0) > 0.06
        )
    return (
        not evidence.get("cube_lifted")
        and evidence.get("correct_contact_steps", 0) >= 3
        and evidence.get("forward_motion_m", 0) > 0.06
    )


def ray_intersects_box(eye, point, center, half_size):
    eye, point, center, half_size = map(np.asarray, (eye, point, center, half_size))
    near, far = 0.0, 1.0
    for origin, direction, lower, upper in zip(
        eye, point - eye, center - half_size, center + half_size
    ):
        if abs(direction) < 1e-10:
            if not lower <= origin <= upper:
                return False
            continue
        a, b = sorted(((lower - origin) / direction, (upper - origin) / direction))
        near, far = max(near, a), min(far, b)
        if near > far:
            return False
    return near < 1 and far > 0
