"""Independent continuously separated routes within an audited reachable area."""

import math

import numpy as np

from robomme_plus.common.motion import swept_gap as swept_gap


def plan_routes(
    count,
    radius,
    seed,
    steps=720,
    speed=0.0045,
    obstacles=(),
    initial=None,
    initial_reject=None,
    reachable_area=None,
):
    """Each object pursues its own changing goals; no discrete swap boundaries.

    Every segment is checked against all other moving objects and stationary
    obstacles. Rejected motions become waits rather than overlapping actors.
    The separate RNG never consumes native benchmark random draws.
    """
    rng = np.random.default_rng(seed)
    obstacle_xy = np.asarray([p for p, _r in obstacles], dtype=float).reshape(-1, 2)
    obstacle_radii = np.asarray([r for _p, r in obstacles], dtype=float)
    bounds = np.array([[-0.31, 0.12], [-0.32, 0.32]])
    points = []
    margin = 0.012
    for index in range(count if initial is None else 0):
        for _ in range(20000):
            p = rng.uniform(bounds[:, 0] + radius, bounds[:, 1] - radius)
            if not reachable_area.contains(p):
                continue
            if initial_reject is not None and initial_reject(index, p):
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
    paths = [points.copy()]
    radii = np.full(count, radius)
    goals = np.array(
        [
            reachable_area.sample(rng, bounds[:, 0] + radius, bounds[:, 1] - radius)
            for _ in range(count)
        ]
    )
    angles = np.linspace(0, 2 * math.pi, 25, endpoint=False)
    heading = np.zeros((count, 2))
    minimum = float("inf")
    for step in range(steps):
        before = points.copy()
        if step % 80 == 0:
            goals = np.array(
                [
                    reachable_area.sample(
                        rng, bounds[:, 0] + radius, bounds[:, 1] - radius
                    )
                    for _ in range(count)
                ]
            )
        for index in rng.permutation(count):
            direction = goals[index] - points[index]
            if np.linalg.norm(direction) < 0.04:
                goals[index] = reachable_area.sample(
                    rng, bounds[:, 0] + radius, bounds[:, 1] - radius
                )
                direction = goals[index] - points[index]
            preferred = direction / max(np.linalg.norm(direction), 1e-9)
            unit = np.column_stack((np.cos(angles), np.sin(angles)))
            # Vary speed independently, with smooth changes within each route.
            velocity = speed * (0.65 + 0.35 * np.sin(step / 29 + index) ** 2)
            candidates = points[index] + velocity * unit
            cost = np.linalg.norm(candidates - goals[index], axis=1)
            cost -= 0.0008 * (unit @ heading[index])
            for k in np.argsort(cost):
                p = candidates[k]
                if not reachable_area.segment(points[index], p):
                    continue
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
    if np.min(lengths) < 0.15:
        raise ValueError(f"A route stalled: lengths={lengths}")
    return paths, {
        "actor_count": count,
        "radius_m": radius,
        "steps": steps,
        "max_speed_m_per_control_step": speed,
        "minimum_continuous_pair_gap_m": minimum,
        "route_lengths_m": lengths.tolist(),
        "seed": seed,
        "strategy": "independent changing goals with continuously checked separation",
        "obstacles": [{"xy": list(p), "radius_m": r} for p, r in obstacles],
    }
