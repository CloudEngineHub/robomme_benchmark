"""Continuous motion geometry shared across task suites."""

import numpy as np


def swept_gap(before, after, radii):
    """Minimum bounding-circle gap over a whole linear control-step interval."""
    before, after = np.asarray(before), np.asarray(after)
    i, j = np.triu_indices(len(before), 1)
    if not len(i):
        return float("inf")
    relative = before[i] - before[j]
    velocity = (after[i] - before[i]) - (after[j] - before[j])
    t = np.clip(
        -np.sum(relative * velocity, axis=1)
        / np.maximum(np.sum(velocity * velocity, axis=1), 1e-20),
        0,
        1,
    )
    distances = np.linalg.norm(relative + t[:, None] * velocity, axis=1)
    return float(np.min(distances - np.asarray(radii)[i] - np.asarray(radii)[j]))
