"""Shared constants and sampling helpers for the V4 xhard tier (NEWTASK_RELEASE_V4_PLAN 2.0①).

Only read on xhard branches; native tiers never import anything from this module for value selection, so V0/V1 are unaffected.

* ``DISTRACTOR_COLORS``: global color pool for "other color" distractors (B2: yellow/cyan/magenta, shared by six envs, A5 fixes 3).
* ``corner_push``: corner-bias mapping. Pushes a uniform sample on ``[0,1]`` toward the ends according to ``corner_bias``;
  with ``corner_bias=0`` it **returns the same object unchanged** (callers rely on this to stay identical to the existing uniform sampling).
* ``cube_obb2d_exact`` (V5 2.0①): precomputed 2D obstacle ``(c, A, h)`` from the cube's true yaw, replacing the actor path of
  ``object_generation._trimesh_box_to_obb2d``, which degenerates to a line segment on a cube.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

# Order is fixed: names appear in task text and spec records; rgba must match the B2 decision exactly.
DISTRACTOR_COLORS = (
    {"name": "yellow", "rgba": (1, 1, 0, 1)},
    {"name": "cyan", "rgba": (0, 1, 1, 1)},
    {"name": "magenta", "rgba": (1, 0, 1, 1)},
)

# Fourth distractor color for PickXtimes/SwingXtimes only; the three-color pool above is shared by the four Unmask/Swap tasks (sampler requires exact equality) and must not change.
BLOCK_DISTRACTOR_COLORS = DISTRACTOR_COLORS + (
    {"name": "orange", "rgba": (1, 0.5, 0, 1)},
)

# Any-hue range with saturation >= 0.5 and value >= 0.4 (excludes near-white, near-black, near-gray); still draws only 3 uniform numbers like the original RGB sampling.
HSV_FLOOR_COLOR = {"h_range": [0.0, 1.0], "s_range": [0.5, 1.0], "v_range": [0.4, 1.0]}


def hsv_floor_rgb(u, cfg=None):
    """Map 3 uniform [0,1) numbers to an RGB triple (floats) within the restricted color range."""
    cfg = HSV_FLOOR_COLOR if cfg is None else cfg
    (h0, h1), (s0, s1), (v0, v1) = cfg["h_range"], cfg["s_range"], cfg["v_range"]
    h = h0 + float(u[0]) * (h1 - h0)
    s = s0 + float(u[1]) * (s1 - s0)
    v = v0 + float(u[2]) * (v1 - v0)
    return list(colorsys.hsv_to_rgb(h % 1.0, s, v))


CORNER_GAIN = 4.0


def corner_push(u, corner_bias):
    """Push a single ``u ∈ [0,1]`` toward the 0 or 1 end according to the corner bias.

    Mapping: ``t = 2u-1``, ``t' = sign(t)·|t|^p``, ``p = 1/(1+CORNER_GAIN·b)``, returns ``(t'+1)/2``.
    Monotonic, endpoint-preserving, symmetric about 0.5; the two coordinates are pushed independently, so the joint distribution favors the four corners.
    ``b`` must lie in ``[0,1]``; when ``b == 0`` no float arithmetic is done and ``u`` is returned as is.
    """
    b = float(corner_bias)
    if b == 0.0:
        return u
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"corner_bias must be in [0,1], got {corner_bias}")
    t = 2.0 * float(u) - 1.0
    p = 1.0 / (1.0 + CORNER_GAIN * b)
    mag = abs(t) ** p
    return (1.0 + (mag if t >= 0 else -mag)) / 2.0


def _as_numpy(value):
    """Convert a torch tensor / sapien array / list to a float64 numpy array (no torch import; duck-typed)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _xy_yaw_from_pose(pose):
    """Get the center xy and yaw from a pose (``.p`` is xyz, ``.q`` is a wxyz quaternion; a batch dim ``[1, ·]`` is allowed).

    yaw is the heading on the table of the most horizontal of the three body axes: under any rotation the squared z components
    of the three axes sum to 1, so the most horizontal axis has |z| ≤ 1/√3 and an xy projection length ≥ √(2/3), hence **never degenerate**;
    the four side faces of a cube are equivalent, so whichever horizontal axis is chosen gives the same square.
    For an upright cube (rotated only about z) column 0 is the body x axis and yaw matches the yaw used to build the cube (mod 2π).
    """
    p = _as_numpy(pose.p).reshape(-1)[:3]
    w, x, y, z = _as_numpy(pose.q).reshape(-1)[:4]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("cube_obb2d_exact: pose quaternion is zero")
    w, x, y, z = w / n, x / n, y / n, z / n
    cols = (
        (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
        (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
        (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)),
    )
    k = min(range(3), key=lambda i: abs(cols[i][2]))
    return float(p[0]), float(p[1]), math.atan2(cols[k][1], cols[k][0])


def cube_obb2d_exact(pose_or_xy_yaw, half, pad=0.0):
    """Exact 2D obstacle ``(c, A, h)`` for a cube, usable directly in the ``avoid`` of ``spawn_random_cube`` / ``spawn_random_target``.

    * ``pose_or_xy_yaw``: three numbers ``(x, y, yaw)`` (yaw in radians about z), a pose with ``.p`` / ``.q``
      (mani_skill ``Pose``, ``sapien.Pose``), or an actor with ``.pose`` (its current pose is used).
    * ``half``: cube half edge length (meters, scalar); ``pad``: margin added to each half axis, with the same semantics as
      ``extra_pad`` in the actor path ``avoid=[(actor, pad)]``.
    * Returns ``(c, A, h)``: ``c`` has shape ``(2,)``, ``A`` has shape ``(2, 2)`` (each column is a unit axis, ``[[cos, -sin], [sin, cos]]``),
      ``h`` has shape ``(2,)``, all float64 ``np.ndarray`` -- exactly the "precomputed obstacle" format both spawn functions recognize
      (a triple whose first two items are ``np.ndarray``), and bitwise isomorphic to ``_build_new_cube_obb2d(x, y, half, yaw, pad)``.

    Pure function: draws no random numbers and reads or writes no environment state; the two axes are always orthonormal, so unlike
    ``_trimesh_box_to_obb2d`` it never degenerates to a line segment when the vertical axis falls into the first two columns (plan 2.0①).
    """
    if hasattr(pose_or_xy_yaw, "p") and hasattr(pose_or_xy_yaw, "q"):
        x, y, yaw = _xy_yaw_from_pose(pose_or_xy_yaw)
    elif hasattr(pose_or_xy_yaw, "pose"):
        x, y, yaw = _xy_yaw_from_pose(pose_or_xy_yaw.pose)
    else:
        values = _as_numpy(pose_or_xy_yaw).reshape(-1)
        if values.shape != (3,):
            raise ValueError(
                f"cube_obb2d_exact: expected three numbers (x, y, yaw), got shape {values.shape}"
            )
        x, y, yaw = (float(v) for v in values)
    half = float(half)
    pad = float(pad)
    if not half > 0.0:
        raise ValueError(f"cube_obb2d_exact: half must be positive, got {half}")
    if pad < 0.0:
        raise ValueError(f"cube_obb2d_exact: pad must not be negative, got {pad}")
    # Same construction as object_generation._build_new_cube_obb2d so the same (x, y, yaw) gives a bitwise-identical array
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    h = np.array([half + pad, half + pad], dtype=np.float64)
    return c, A, h
