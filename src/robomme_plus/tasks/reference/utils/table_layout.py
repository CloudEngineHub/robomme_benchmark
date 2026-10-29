"""Seeded tabletop packing with shared proposal rules for all object roles."""

import numpy as np


TABLE_BOUNDS = np.array([[-0.40, 0.14], [-0.34, 0.34]])
ROBOT_BASE_XY = np.array([-0.615, 0.0])


def sample_cube_pool(seed, count, half_size, gap, obstacles=(), *, bounds=TABLE_BOUNDS):
    """Pack the entire pool before assigning roles, retaining full size/gaps.

    Circumscribed discs conservatively contain every upright cube rotation.
    Full-layout restarts avoid returning a truncated population when a greedy
    prefix blocks the last object. The final permutation removes packing order
    from the task's object indices.
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    radius = np.sqrt(2) * half_size
    clearance = np.sqrt(2) * gap
    yaws = rng.uniform(-np.pi, np.pi, count)
    for _ in range(128):
        points = []
        for _ in range(count):
            for _ in range(2000):
                point = rng.uniform(bounds[:, 0] + radius,
                                    bounds[:, 1] - radius)
                distance = np.linalg.norm(point - ROBOT_BASE_XY)
                if not 0.28 + radius <= distance <= 0.76 - radius:
                    continue
                if any(np.linalg.norm(point - p) < 2 * radius + clearance for p in points):
                    continue
                if any(np.linalg.norm(point - p) < radius + r + clearance for p, r in obstacles):
                    continue
                points.append(point)
                break
            else:
                break
        if len(points) == count:
            order = rng.permutation(count)
            return np.asarray(points)[order], yaws[order]
    raise RuntimeError(f"Cannot pack all {count} tabletop cubes with physical clearance")
