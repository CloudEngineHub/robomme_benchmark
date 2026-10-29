"""Private reference-plan speed changes; the environment stays at 20 Hz."""

from contextlib import contextmanager
import math

import numpy as np


@contextmanager
def dense_motion(env, enabled=True):
    """Keep every planned sample through a contact-sensitive operation."""
    base = getattr(env, "unwrapped", env)
    previous = getattr(base, "_reference_dense_motion", False)
    base._reference_dense_motion = previous or enabled
    try:
        yield
    finally:
        base._reference_dense_motion = previous


def retime_path(path, factor=2):
    """Halve moving path intervals, preserving endpoints and velocity timing.

    Sampling remains at the native control frequency. Interpolate when the
    interval count is odd so the final interval is not abruptly half as long.
    Constant paths are intentional contact dwells and retain their duration.
    Gripper and settling commands are outside this function.
    """
    if factor not in (1, 2):
        raise ValueError("Reference plan factor must be 1 or 2")
    positions = np.asarray(path["position"])
    n = len(positions)
    if factor == 1 or n <= 3 or np.max(np.abs(positions - positions[0])) < 1e-5:
        return path
    count = math.ceil((n - 1) / factor) + 1
    source = np.arange(n)
    selected = np.linspace(0, n - 1, count)
    scale = (n - 1) / (count - 1)

    def sample(values):
        return np.stack([np.interp(selected, source, values[:, j])
                         for j in range(values.shape[1])], axis=1)

    result = dict(path, position=sample(positions))
    if "velocity" in path:
        result["velocity"] = sample(np.asarray(path["velocity"])) * scale
    if "acceleration" in path:
        result["acceleration"] = sample(np.asarray(path["acceleration"])) * scale**2
    if "time" in path:
        result["time"] = np.interp(selected, source, path["time"]) / scale
    return result


def reference_path(base, path):
    """Apply only an explicitly selected handheld reference speed profile."""
    factor = getattr(base, "_reference_plan_stride", 1)
    dense = getattr(base, "_reference_dense_motion", False)
    if getattr(base, "_required_method", None) not in ("hook", "poke", "scoop"):
        factor = 1
    # Pan insertion, ramp separation and unloading are contact-sensitive.
    if (getattr(base, "_required_method", None) == "scoop"
            and getattr(base, "_reference_scoop_stage", "") in
            ("loading", "lifting", "unloading")):
        dense = True
    result = retime_path(path, 1 if dense else factor)
    stats = getattr(base, "_reference_plan_stats", None)
    if stats is not None:
        stats["planned_samples"] += len(path["position"])
        stats["executed_samples"] += len(result["position"])
        stats["dense_samples"] += len(path["position"]) if dense else 0
    return result
