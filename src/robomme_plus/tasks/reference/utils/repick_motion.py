"""Seeded, independently curving VideoRepick routes with continuous clearance."""

import math

import numpy as np

from robomme_plus.common.motion import swept_gap as swept_gap


def initial_yaws(count, seed):
    """Independent uniform upright headings, separate from layout and motion."""
    return np.random.default_rng(int(seed) + 41732).uniform(0, 2 * math.pi, count)


def plan_rotations(count, seed, steps, control_freq=20):
    """Independent fixed directions and distinct slow speeds for each object.

    Stratified 5–20 degree/second speeds ensure a mix of slower and faster
    objects. A one-second ramp starts the turns smoothly. The separate RNG
    leaves the translation paths unchanged.
    """
    angles = np.zeros((steps + 1, count))
    rng = np.random.default_rng(seed)
    directions = rng.choice([-1, 1], size=count)
    speeds = 5 + 15 * (rng.permutation(count) + rng.uniform(size=count)) / count
    ramp = np.minimum(np.arange(1, steps + 1) / control_freq, 1.0)
    angles[1:] = np.cumsum(
        ramp[:, None] * np.radians(directions * speeds)[None, :] / control_freq,
        axis=0,
    )
    return angles, {
        "axis": "world vertical (yaw)",
        "seed": int(seed),
        "target_speed_range_deg_s": [5, 20],
        "directions": [
            "counterclockwise" if direction > 0 else "clockwise"
            for direction in directions
        ],
        "direction_signs": directions.tolist(),
        "direction_rule": "independently sampled once per object",
        "angular_speeds_deg_s": speeds.tolist(),
        "speed_rule": "fixed per object after the initial ramp",
        "ramp_seconds": 1,
        "control_freq_hz": control_freq,
    }


def yaw_quaternion(quaternion, angle):
    """Apply a world-vertical turn to a wxyz quaternion."""
    w, x, y, z = quaternion
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return np.array([c * w - s * z, c * x - s * y,
                     c * y + s * x, c * z + s * w])


def sample_cluster(
    rng, count, radius, bounds, diameter, obstacles, reject=None, minimum=None
):
    """Seeded random packing with full restarts, including ten-cube groups.

    ``minimum`` overrides the smallest center distance between any two objects.
    """
    minimum = (
        2 * radius + (0.004 if count >= 7 else 0.012) + 0.009
        if minimum is None
        else minimum
    )
    for attempt in range(2000):
        center = rng.uniform(bounds[:, 0] + diameter / 2, bounds[:, 1] - diameter / 2)
        points = []
        for index in range(count):
            for trial in range(400):
                angle = rng.uniform(0, 2 * math.pi)
                distance = (diameter - 0.010) / 2 * math.sqrt(rng.uniform())
                p = center + distance * np.array([math.cos(angle), math.sin(angle)])
                if np.any(p < bounds[:, 0] + radius) or np.any(
                    p > bounds[:, 1] - radius
                ):
                    continue
                if any(np.linalg.norm(p - q) < minimum for q in points):
                    continue
                if any(
                    np.linalg.norm(p - q) < radius + r + 0.018 for q, r in obstacles
                ):
                    continue
                points.append(p)
                break
            else:
                break
        if len(points) == count:
            points = spread_out(np.asarray(points), max(0.10, minimum), radius,
                                bounds, diameter, obstacles, rng)
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            if (distances[~np.eye(count, dtype=bool)].min(initial=np.inf) < max(0.10, minimum)
                or distances.max() > diameter
                or any(np.any(np.linalg.norm(points - p, axis=1) < radius + r + 0.018)
                       for p, r in obstacles)):
                continue
            # Assign actor roles only after the whole pool is clear. Where a
            # role requires visibility, choose uniformly among its eligible
            # positions rather than repeatedly discarding a valid full pack.
            remaining = list(rng.permutation(count))
            order = []
            for index in range(count):
                eligible = [j for j in remaining if reject is None or not reject(index, points[j])]
                if not eligible:
                    break
                chosen = int(rng.choice(eligible))
                order.append(chosen)
                remaining.remove(chosen)
            if len(order) == count:
                return points[order]
    raise ValueError(f"Cannot sample a compact group of {count} cubes")


def spread_out(
    points, spacing, radius, bounds, diameter, obstacles, rng, iterations=2000
):
    """The current arrangement pushed apart until all objects are ``spacing`` apart.

    Close pairs repel and distant pairs pull together to respect the group
    diameter, inside ``bounds`` and clear of ``obstacles``. Objects keep their
    relative places, so reaching the result needs only short, non-crossing moves.
    """
    p = np.asarray(points, dtype=float).copy()
    # A fixed circle about the original centroid can trap dense groups against
    # an obstacle. Constrain the actual pair distances instead, leaving room
    # for the whole group to shift away from the obstacle while spreading out.
    reach = diameter - 0.012
    for _ in range(iterations):
        moved = False
        for a in range(len(p)):
            for b in range(a + 1, len(p)):
                d = p[b] - p[a]
                distance = float(np.linalg.norm(d))
                if distance < spacing + 1e-3:
                    d = (
                        d / distance
                        if distance > 1e-9
                        else rng.normal(size=2) / math.sqrt(2)
                    )
                    push = (spacing + 2e-3 - distance) / 2
                    p[a] -= push * d
                    p[b] += push * d
                    moved = True
                elif distance > reach:
                    pull = (distance - reach) / 2
                    p[a] += pull * d / distance
                    p[b] -= pull * d / distance
                    moved = True
        for i in range(len(p)):
            for q, r in obstacles:
                away = p[i] - np.asarray(q, float)
                gap = radius + r + 0.018
                if np.linalg.norm(away) < gap:
                    p[i] = (
                        np.asarray(q, float)
                        + away / max(np.linalg.norm(away), 1e-9) * gap
                    )
            p[i] = np.clip(p[i], bounds[:, 0] + radius, bounds[:, 1] - radius)
        if not moved:
            break
    return p


def plan_routes(
    count,
    radius,
    seed,
    steps=720,
    speed=0.0045,
    obstacles=(),
    initial=None,
    initial_reject=None,
    workspace_bounds=None,
    max_group_diameter=None,
    exchange_destinations=False,
    settle_steps=0,
    settle_spacing=None,
    goal_refresh_range=(20, 50),
):
    """Combine asynchronous curves with destinations refreshed every 20–50 steps.

    Every segment is checked against all other moving objects and stationary
    obstacles. Rejected motions become waits rather than overlapping actors.
    The separate RNG never consumes native benchmark random draws. With
    ``settle_spacing``, the last ``settle_steps`` steps head for a fresh seeded
    arrangement whose objects are all at least ``settle_spacing`` apart (room
    for the gripper around any of them), and the plan is rejected if the
    objects end closer than that (less 1 cm).
    """
    rng = np.random.default_rng(seed)
    refresh = np.asarray(goal_refresh_range)
    if (
        refresh.shape != (2,)
        or not np.issubdtype(refresh.dtype, np.integer)
        or refresh[0] < 1
        or refresh[1] < refresh[0]
    ):
        raise ValueError("Goal refresh range must be two positive ordered integers")
    obstacle_xy = np.asarray([p for p, _r in obstacles], dtype=float).reshape(-1, 2)
    obstacle_radii = np.asarray([r for _p, r in obstacles], dtype=float)
    bounds = np.asarray(
        workspace_bounds
        if workspace_bounds is not None
        else [[-0.31, 0.12], [-0.32, 0.32]],
        dtype=float,
    )
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] - bounds[:, 0] <= 2 * radius)
    ):
        raise ValueError(
            "Workspace bounds must contain the moving objects with clearance"
        )
    if max_group_diameter is not None and (
        not np.isfinite(max_group_diameter)
        or max_group_diameter <= 2 * radius + 0.012 + 4 * speed
    ):
        raise ValueError("Group diameter is too small for separated moving objects")
    points = []
    margin = 0.004
    if initial is None and max_group_diameter is not None:
        initial = sample_cluster(
            rng, count, radius, bounds, max_group_diameter, obstacles, initial_reject
        )
    for index in range(count if initial is None else 0):
        for _ in range(20000):
            p = rng.uniform(bounds[:, 0] + radius, bounds[:, 1] - radius)
            if initial_reject is not None and initial_reject(index, p):
                continue
            if max_group_diameter is not None and any(
                np.linalg.norm(p - q) > max_group_diameter - 4 * speed for q in points
            ):
                continue
            if all(
                np.linalg.norm(p - q) > 2 * radius + margin + 2 * speed for q in points
            ) and all(
                np.linalg.norm(p - q) > radius + r + margin + speed
                for q, r in obstacles
            ):
                points.append(p)
                break
        else:
            raise ValueError(f"Cannot fit {count} moving objects with radius {radius}")
    points = np.array(points if initial is None else initial, dtype=float)
    if swept_gap(points, points, np.full(count, radius)) < margin:
        raise ValueError("Initial moving-object footprints overlap or lack clearance")
    if (
        max_group_diameter is not None
        and np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2))
        > max_group_diameter
    ):
        raise ValueError("Initial objects exceed the maximum swarm diameter")
    paths = [points.copy()]
    radii = np.full(count, radius)
    goals = rng.uniform(bounds[:, 0] + radius, bounds[:, 1] - radius, (count, 2))
    angles = np.linspace(0, 2 * math.pi, 25, endpoint=False)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    heading = np.zeros((count, 2))
    minimum = float("inf")
    destination_events = []
    goal_events = []
    motion_events = []
    next_goal_step = 0
    # Each actor runs its own cycle. A curve can continue across a destination
    # refresh, so loops and reversals have time to complete rather than sharing
    # the group's change times.
    styles = ("travel", "s_curve", "loop", "reversal")
    style_orders = [rng.permutation(len(styles)) for _ in range(count)]
    style_cursors = np.zeros(count, dtype=int)
    motions = [None] * count
    settle_start = (
        steps - settle_steps if settle_spacing is not None and settle_steps else None
    )
    for step in range(steps):
        before = points.copy()
        if step == settle_start:
            goals = spread_out(
                points,
                settle_spacing,
                radius,
                bounds,
                max_group_diameter if max_group_diameter is not None else 0.30,
                obstacles,
                rng,
            )
        elif step == next_goal_step and (settle_start is None or step < settle_start):
            if max_group_diameter is None:
                goals = rng.uniform(
                    bounds[:, 0] + radius, bounds[:, 1] - radius, (count, 2)
                )
            else:
                # Draw each destination afresh. Transforming the original
                # arrangement preserves its shape even when the group travels
                # and turns; independent offsets require actual reconfiguration.
                center = rng.uniform(bounds[:, 0] + radius, bounds[:, 1] - radius)
                angles_to_goals = rng.uniform(-math.pi, math.pi, count)
                distances_to_goals = (max_group_diameter / 2 - radius) * np.sqrt(
                    rng.uniform(size=count)
                )
                offsets = distances_to_goals[:, None] * np.column_stack((
                    np.cos(angles_to_goals), np.sin(angles_to_goals)
                ))
                goals = np.clip(
                    center + offsets,
                    bounds[:, 0] + radius,
                    bounds[:, 1] - radius,
                )
            if exchange_destinations:
                order = rng.permutation(count)
                # An exchange event must actually exchange destinations.
                if np.array_equal(order, np.arange(count)) and count > 1:
                    order = np.roll(order, 1)
                goals = goals[order]
                destination_events.append(
                    {"step": step, "assignment": order.tolist()}
                )
            goal_events.append({"step": step, "positions_xy_m": goals.tolist()})
            next_goal_step = step + int(rng.integers(refresh[0], refresh[1] + 1))
        for index in rng.permutation(count):
            direction = goals[index] - points[index]
            if max_group_diameter is None and np.linalg.norm(direction) < 0.04:
                goals[index] = rng.uniform(bounds[:, 0] + radius, bounds[:, 1] - radius)
                direction = goals[index] - points[index]
            preferred = direction / max(np.linalg.norm(direction), 1e-9)
            settling = settle_start is not None and step >= settle_start
            if not settling:
                motion = motions[index]
                if motion is None or step >= motion["end_step"]:
                    style = styles[style_orders[index][style_cursors[index] % len(styles)]]
                    style_cursors[index] += 1
                    axis = preferred if np.linalg.norm(preferred) > 0 else unit[0]
                    side = int(rng.choice([-1, 1]))
                    normal = side * np.array([-axis[1], axis[0]])
                    loop_radius = float(rng.uniform(0.025, 0.045))
                    duration = int(rng.integers(60, 101))
                    motion = {
                        "actor": int(index),
                        "style": style,
                        "start_step": step,
                        "end_step": step + duration,
                        "axis": axis.tolist(),
                        "normal": normal.tolist(),
                        "loop_center_xy_m": (points[index] + loop_radius * normal).tolist(),
                        "loop_radius_m": loop_radius,
                        "side": side,
                    }
                    motions[index] = motion
                    motion_events.append(motion)
                phase = (step - motion["start_step"]) / (
                    motion["end_step"] - motion["start_step"]
                )
                axis, normal = np.asarray(motion["axis"]), np.asarray(motion["normal"])
                if motion["style"] == "s_curve":
                    preferred = axis + 1.2 * math.sin(2 * math.pi * phase) * normal
                elif motion["style"] == "reversal":
                    # Turn around smoothly, travel back, then turn forward again.
                    angle = math.pi * (0.5 - 0.5 * math.cos(2 * math.pi * phase))
                    preferred = math.cos(angle) * axis + math.sin(angle) * normal
                elif motion["style"] == "loop":
                    radial = points[index] - motion["loop_center_xy_m"]
                    distance = max(float(np.linalg.norm(radial)), 1e-9)
                    radial /= distance
                    tangent = motion["side"] * np.array([-radial[1], radial[0]])
                    preferred = tangent + np.clip(
                        (motion["loop_radius_m"] - distance) / motion["loop_radius_m"],
                        -1, 1,
                    ) * radial
                preferred /= max(float(np.linalg.norm(preferred)), 1e-9)
            # Vary speed independently, with smooth changes within each route.
            velocity = speed * (0.65 + 0.35 * np.sin(step / 29 + index) ** 2)
            candidates = points[index] + velocity * unit
            cost = (
                np.linalg.norm(candidates - goals[index], axis=1)
                if settling
                else -velocity * (unit @ preferred)
            )
            cost -= 0.0008 * (unit @ heading[index])
            for k in np.argsort(cost):
                p = candidates[k]
                if np.any(p < bounds[:, 0] + radius) or np.any(
                    p > bounds[:, 1] - radius
                ):
                    continue
                other = np.arange(count) != index
                # Extra allowance covers the other actor's within-step motion.
                if np.any(
                    np.linalg.norm(points[other] - p, axis=1)
                    < 2 * radius + margin + 2 * speed
                ):
                    continue
                if max_group_diameter is not None and np.any(
                    np.linalg.norm(points[other] - p, axis=1)
                    > max_group_diameter - 2 * speed
                ):
                    continue
                if len(obstacle_xy) and np.any(
                    np.linalg.norm(obstacle_xy - p, axis=1)
                    < radius + obstacle_radii + margin + speed
                ):
                    continue
                points[index] = p
                heading[index] = unit[k]
                break
        gap = swept_gap(before, points, radii)
        if gap < margin:
            raise AssertionError(f"Moving objects intersect: gap {gap}")
        minimum = min(minimum, gap)
        paths.append(points.copy())
    paths = np.asarray(paths)
    lengths = np.linalg.norm(np.diff(paths, axis=0), axis=2).sum(axis=0)
    tracking = paths[:settle_start + 1] if settle_start is not None else paths
    pair_distances = np.linalg.norm(
        tracking[:, :, None] - tracking[:, None, :], axis=3
    )
    pair_i, pair_j = np.triu_indices(count, 1)
    distance_ranges = np.ptp(pair_distances[:, pair_i, pair_j], axis=0)
    pair_distances[:, np.arange(count), np.arange(count)] = np.inf
    neighbors = np.argmin(pair_distances, axis=2)
    relative_changes = {
        "tracking_steps": len(tracking) - 1,
        "pair_count": len(pair_i),
        "pairs_changing_distance_by_at_least_4cm": int(np.sum(distance_ranges >= .04)),
        "median_pair_distance_range_m": float(np.median(distance_ranges)),
        "cubes_changing_nearest_neighbor": int(sum(
            len(np.unique(neighbors[:, i])) > 1 for i in range(count)
        )),
    }
    diameter = float(
        np.linalg.norm(paths[:, :, None] - paths[:, None, :], axis=3).max()
    )
    if np.min(lengths) < 0.15:
        raise ValueError(f"A route stalled: lengths={lengths}")
    settle = None
    if settle_start is not None:
        final = paths[-1]
        closest = float(
            min(
                np.linalg.norm(final[a] - final[b])
                for a in range(count)
                for b in range(a + 1, count)
            )
        )
        if closest < settle_spacing - 0.01:
            raise ValueError(
                f"The swarm did not settle with room for the gripper (closest pair {closest:.3f} m)"
            )
        settle = {
            "start_step": settle_start,
            "spacing_m": settle_spacing,
            "final_closest_pair_m": closest,
        }
    return paths, {
        "actor_count": count,
        "radius_m": radius,
        "workspace_xy_bounds_m": bounds.tolist(),
        "max_group_diameter_m": max_group_diameter,
        "maximum_planned_group_diameter_m": diameter,
        "steps": steps,
        "max_speed_m_per_control_step": speed,
        "minimum_continuous_pair_gap_m": minimum,
        "route_lengths_m": lengths.tolist(),
        "seed": seed,
        "destination_permutations": destination_events,
        "goal_refresh_range_control_steps": refresh.tolist(),
        "destination_updates": goal_events,
        "motion_primitives": motion_events,
        "relative_position_changes": relative_changes,
        "settle": settle,
        "strategy": "independent S-curves, loops and reversals with changing destinations and continuous separation checks",
        "obstacles": [{"xy": list(p), "radius_m": r} for p, r in obstacles],
    }
