"""Irregular continuous target and clutter sampling."""

import numpy as np

MIN_GAP_M = 0.012


def sample_target_positions(
    rng, count, bounds, radius, eligible, free_space=lambda xy: True
):
    """All color roles receive identically distributed collision-free proposals."""
    bounds = np.asarray(bounds, dtype=float)
    if not count:
        return np.empty((0, 2))
    points = []
    for _ in range(20000):
        xy = rng.uniform(bounds[:, 0], bounds[:, 1])
        if (
            free_space(xy)
            and all(np.linalg.norm(xy - p) > 2 * radius + MIN_GAP_M for p in points)
            and eligible(xy)
        ):
            points.append(xy)
        if len(points) == count:
            return np.asarray(points)
    raise RuntimeError(
        f"Only {len(points)}/{count} valid reachable target positions sampled"
    )


def packed_points(
    rng,
    count,
    bounds,
    radius,
    occupied,
    allowed,
    accept=lambda xy: True,
    initial_counts=None,
    gap=0.012,
):
    bounds = np.asarray(bounds)
    prior = np.asarray(occupied, dtype=float).reshape(-1, 2)
    counts = (
        np.zeros(9, dtype=int)
        if initial_counts is None
        else np.asarray(initial_counts).copy()
    )
    if not count:
        return np.empty((0, 2)), counts
    minimum = 2 * radius + gap
    for _ in range(6):
        pool = rng.uniform(bounds[:, 0], bounds[:, 1], size=(4000, 2))
        distances = (
            np.linalg.norm(pool[:, None] - prior[None], axis=2).min(axis=1)
            if len(prior)
            else np.full(len(pool), np.inf)
        )
        valid = distances > minimum
        pool, distances = pool[valid], distances[valid]
        valid = np.array([allowed(xy) for xy in pool], dtype=bool)
        pool, distances = pool[valid], distances[valid]
        chosen = []
        for _ in range(count):
            candidates = np.flatnonzero(distances > minimum)
            if not len(candidates):
                break
            # Uniform among the feasible proposals. Farthest-point selection
            # made distractors systematically occupy the edges of the scene.
            while len(candidates):
                index = int(rng.choice(candidates))
                if accept(pool[index]):
                    break
                distances[index] = -np.inf
                candidates = candidates[candidates != index]
            else:
                break
            chosen.append(pool[index].copy())
            distances = np.minimum(
                distances, np.linalg.norm(pool - pool[index], axis=1)
            )
        if len(chosen) == count:
            points = np.asarray(chosen)[rng.permutation(count)]
            cells = np.clip(
                np.floor(
                    (points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) * 3
                ).astype(int),
                0,
                2,
            )
            counts += np.bincount(3 * cells[:, 0] + cells[:, 1], minlength=9)
            return points, counts
    raise RuntimeError(
        f"Cannot place {count} packed distractors with physical clearance"
    )
