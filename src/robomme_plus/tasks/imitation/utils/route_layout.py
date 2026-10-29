"""Replayable RouteStick row transforms and physical board heights."""

import math
from numbers import Real

import numpy as np

SPACING_SCALES = (.8, .9, 1.1, 1.2)
ALLOWED_SPACING_SCALES = tuple(sorted((*SPACING_SCALES, 1.0)))


def default_route_layout(seed, level):
    """Fixed 13 cm boards; every Hard episode samples per-board motion once."""
    if level not in ("easy", "medium", "hard"):
        raise ValueError("RouteStick difficulty must be easy, medium or hard")
    parameters = {"version": 3}
    if level == "easy":
        parameters.update(spacing_scale=1.0, translation_xy_m=[0., 0.], yaw_deg=0.)
    elif level == "hard":
        # Keep these draws independent of the existing row geometry. Save the
        # realized choices so reset/replay never rerolls a board on arrival.
        rng = np.random.default_rng([int(seed), 618011])
        rng.random()  # Reserve the former episode-gate draw; keep board masks stable.
        parameters["board_motion"] = {
            "mode": "arrival_boards",
            "enabled_boards": (rng.random(4) < .5).tolist(),
        }
    return resolve_route_layout(seed, parameters)


def resolve_route_layout(seed, parameters):
    """Resolve partial layout parameters into an exactly replayable transform.

    Versions 1/2 retain the earlier discrete spacing experiments. Version 3
    samples continuous 0.9–1.0 spacing and defaults to fixed 13 cm boards with
    physical collisions active and contact-failure scoring disabled.
    """
    if not isinstance(parameters, dict):
        raise ValueError("execution_route_layout must be a dictionary")
    allowed = {"version", "spacing_scale", "board_height_scale", "translation_xy_m", "yaw_deg", "board_collision", "board_contact_fails", "board_motion"}
    if set(parameters) - allowed:
        raise ValueError(f"Unknown execution_route_layout fields: {sorted(set(parameters) - allowed)}")
    version = parameters.get("version", 1)
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("execution_route_layout version must be 1, 2 or 3")
    for field in ("board_collision", "board_contact_fails"):
        if field in parameters and type(parameters[field]) is not bool:
            raise ValueError(f"RouteStick {field} must be a boolean")
    rng = np.random.default_rng([int(seed), 618007 if version == 3 else 618006])
    spacing = parameters.get("spacing_scale", float(rng.uniform(.9, 1.0)) if version == 3
                             else float(rng.choice(SPACING_SCALES)))
    height = parameters.get("board_height_scale", 1.3 if version == 3 else float(rng.uniform(1, 1.5)))
    axes = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))
    translation = parameters.get("translation_xy_m", (np.array(axes[int(rng.integers(4))]) * .05).tolist())
    yaw = parameters.get("yaw_deg", float(rng.choice((-1, 1)) * rng.uniform(5, 10)))
    valid_spacing = (isinstance(spacing, Real) and not isinstance(spacing, bool)
                     and (.9 <= spacing <= 1.0 if version == 3 else spacing in ALLOWED_SPACING_SCALES))
    if not valid_spacing:
        allowed_spacing = "between 0.9 and 1.0" if version == 3 else "0.8, 0.9, 1.0, 1.1 or 1.2"
        raise ValueError(f"RouteStick spacing_scale must be {allowed_spacing}")
    maximum_height = 2.0 if version >= 2 else 1.5
    if isinstance(height, bool) or not isinstance(height, Real) or not 1 <= height <= maximum_height:
        raise ValueError(f"RouteStick board_height_scale must be between 1 and {maximum_height:g}")
    if (isinstance(yaw, bool) or not isinstance(yaw, Real)
            or not (5 <= abs(yaw) <= 10 or (version >= 2 and yaw == 0))):
        raise ValueError("RouteStick execution yaw must have magnitude 5–10 degrees (or zero in versions 2/3)")
    if (not isinstance(translation, (list, tuple)) or len(translation) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in translation)
            or not (math.isclose(math.hypot(*translation), .05, abs_tol=1e-9)
                    or (version >= 2 and math.hypot(*translation) == 0))):
        raise ValueError("RouteStick translation_xy_m must be a finite 5 cm displacement (or zero in versions 2/3)")
    resolved = dict(version=version, spacing_scale=float(spacing), board_height_scale=float(height),
                    translation_xy_m=list(map(float, translation)), yaw_deg=float(yaw))
    # Keep versions 1/2 compatible with existing saved layouts. The current
    # setup keeps contact physics while allowing board touches in task scoring.
    for field in ("board_collision", "board_contact_fails"):
        if field in parameters:
            resolved[field] = parameters[field]
        elif version == 3:
            resolved[field] = field == "board_collision"
    if "board_motion" in parameters:
        if version != 3:
            raise ValueError("RouteStick board_motion requires layout version 3")
        from .route_motion import resolve_board_motion
        resolved["board_motion"] = resolve_board_motion(seed, parameters["board_motion"])
    return resolved


def disable_board_collisions(boards):
    """Keep the visible boards while removing every shape from physics contacts."""
    for board in boards:
        for body in board._bodies:
            for shape in body.get_collision_shapes():
                shape.set_collision_groups([0, 0, 0, 0])


def rotation(layout):
    angle = np.radians(layout["yaw_deg"])
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def execution_points(points, center, layout):
    """Scale and rotate about the demonstrated row center, then translate."""
    return ((np.asarray(points) - center) * layout["spacing_scale"] @ rotation(layout).T
            + center + np.asarray(layout["translation_xy_m"]))
