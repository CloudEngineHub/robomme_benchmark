from typing import Optional, Sequence, Tuple, Union

import numpy as np
import sapien
import torch
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    get_actor_obb,
)
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from transforms3d.euler import euler2quat

from robomme_plus.simulation.specification import (
    EpisodeSpecError as _EpisodeSpecError,
)
from robomme_plus.simulation.geometry import corner_push


def _color_to_rgba(
    color: Union[str, Sequence[float]],
) -> Tuple[float, float, float, float]:
    """Convert a hex string or RGB/RGBA tuple to an RGBA tuple accepted by SAPIEN."""
    if isinstance(color, str):
        return sapien_utils.hex2rgba(color)
    if len(color) == 3:
        return (float(color[0]), float(color[1]), float(color[2]), 1.0)
    if len(color) == 4:
        return tuple(float(c) for c in color)
    raise ValueError("color must be a hex string or a sequence of 3/4 floats")


def build_peg(
    env_or_scene,
    length: float,
    radius: float,
    *,
    initial_pose: Optional["sapien.Pose"] = None,
    head_color: str = "#EC7357",
    tail_color: str = "#F5F5F5",
    density: float = 1200.0,
    name: str = "peg",
) -> Tuple["sapien.Articulation", "sapien.Link", "sapien.Link"]:
    """Construct a peg articulation with head and tail links tied by a fixed joint.

    Args:
        env_or_scene: Environment or scene providing `create_articulation_builder`.
        length: Total length of the peg (meters).
        radius: Half-width of the rectangular cross section (meters).
        initial_pose: Optional pose for the articulation root; defaults to placing
            the head centered at positive x.
        head_color: Hex color for the head visual.
        tail_color: Hex color for the tail visual.
        density: Collision density (kg/m^3) shared by both links.
        name: Name assigned to the articulation.

    Returns:
        The articulation along with the head and tail links.
    """

    scene = getattr(env_or_scene, "scene", env_or_scene)
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[length / 2, 0.0, radius], q=[1, 0, 0, 0])

    builder = scene.create_articulation_builder()
    builder.initial_pose = initial_pose

    head_builder = builder.create_link_builder()
    head_builder.set_name("peg_head")
    head_builder.add_box_collision(
        half_size=[length / 2 * 0.9, radius, radius], density=density
    )
    head_material = sapien.render.RenderMaterial(
        base_color=_color_to_rgba(head_color),
        roughness=0.5,
        specular=0.5,
    )
    head_builder.add_box_visual(
        half_size=[length / 2, radius, radius],
        material=head_material,
    )

    tail_builder = builder.create_link_builder(head_builder)
    tail_builder.set_name("peg_tail")
    tail_builder.set_joint_name("peg_fixed_joint")
    tail_builder.set_joint_properties(
        type="fixed",
        limits=[[0.0, 0.0]],
        pose_in_parent=sapien.Pose(p=[-length, 0.0, 0.0], q=[1, 0, 0, 0]),
        pose_in_child=sapien.Pose(p=[0.0, 0.0, 0.0], q=[1, 0, 0, 0]),
        friction=0.0,
        damping=0.0,
    )
    tail_builder.add_box_collision(
        half_size=[length / 2 * 0.9, radius, radius], density=density
    )
    tail_material = sapien.render.RenderMaterial(
        base_color=_color_to_rgba(tail_color),
        roughness=0.5,
        specular=0.5,
    )
    tail_builder.add_box_visual(
        half_size=[length / 2, radius, radius],
        material=tail_material,
    )

    peg = builder.build(name=name, fix_root_link=False)
    link_map = {link.get_name(): link for link in peg.get_links()}
    peg_head = link_map["peg_head"]
    peg_tail = link_map["peg_tail"]
    return peg, peg_head, peg_tail


def _safe_unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def _trimesh_box_to_obb2d(obb_box, extra_pad=0.0):
    """
    Convert trimesh.primitives.Box (world frame) to 2D OBB representation: center c(2,), axes A(2x2), half-extents h(2,)
    extra_pad: Margins to expand outward on XY plane (meters)
    """
    # Compatible with obb potentially wrapped in .primitive
    b = getattr(obb_box, "primitive", obb_box)
    T = np.asarray(b.transform, dtype=np.float64)  # 4x4
    ex = np.asarray(b.extents, dtype=np.float64)  # 3

    R = T[:3, :3]
    t = T[:3, 3]

    c = t[:2].copy()

    # Take projection of X, Y axes on plane as two axes of 2D OBB
    u = _safe_unit(R[:2, 0])  # x-axis projection
    v = _safe_unit(R[:2, 1])  # y-axis projection
    A = np.stack([u, v], axis=1)  # 2x2, each column is an axis

    h = 0.5 * ex[:2].astype(np.float64)
    if extra_pad > 0:
        h = h + float(extra_pad)
    return c, A, h


def _obb2d_intersect(c1, A1, h1, c2, A2, h2):
    """
    2D OBB SAT detection. c*: (2,), A*: (2x2) columns are axes, h*: (2,)
    Returns True indicating intersection (including contact), False indicating separation
    """
    d = c2 - c1
    axes = [A1[:, 0], A1[:, 1], A2[:, 0], A2[:, 1]]

    for a in axes:
        a = _safe_unit(a)
        # Projected radius
        r1 = abs(np.dot(A1[:, 0], a)) * h1[0] + abs(np.dot(A1[:, 1], a)) * h1[1]
        r2 = abs(np.dot(A2[:, 0], a)) * h2[0] + abs(np.dot(A2[:, 1], a)) * h2[1]
        dist = abs(np.dot(d, a))
        if dist > (r1 + r2):
            return False  # Separating axis exists -> No intersection
    return True  # All axes overlap -> Intersection/Contact


def _yaw_to_quat_tensor(yaw: float, device):
    """
    Get quaternion consistent with ManiSkill/your conversion tools using z-axis Euler angle (shape [1,4], float32, device aligned)
    """
    # euler_angles_to_matrix accepts [roll, pitch, yaw] (radians), returns Nx3x3
    angles = torch.tensor([[0.0, 0.0, float(yaw)]], dtype=torch.float32, device=device)
    R = euler_angles_to_matrix(angles, convention="XYZ")  # (1, 3, 3)
    q = matrix_to_quaternion(R)  # (1, 4) Convention same as ManiSkill
    return q


def _build_new_cube_obb2d(x, y, half_size_xy, yaw, pad_xy=0.0):
    """
    Construct 2D OBB for "cube ready to be placed": center/axes/half-extents
    half_size_xy: float, half length of cube on XY
    yaw: rotation around z-axis (radians)
    pad_xy: extra padding on half length on XY (for minimum gap)
    """
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array(
        [[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64
    )  # Columns are axes
    h = np.array([half_size_xy + pad_xy, half_size_xy + pad_xy], dtype=np.float64)
    return c, A, h


def _center_rule_point_xy(point):
    """Center-distance reference point -> (2,) float64; a prefabricated obstacle triple ``(c, A, h)`` contributes its center ``c``."""
    if (
        isinstance(point, tuple)
        and len(point) == 3
        and isinstance(point[0], np.ndarray)
        and isinstance(point[1], np.ndarray)
    ):
        point = point[0]
    if hasattr(point, "detach"):
        point = point.detach().cpu().numpy()
    xy = np.asarray(point, dtype=np.float64).reshape(-1)
    if xy.shape != (2,):
        raise ValueError(
            f"center-distance reference point must be an xy pair or a prefabricated obstacle (c, A, h), got shape {xy.shape}"
        )
    return xy


def _normalize_center_rules(min_center_dist, center_exclusion, who):
    """V5 (L4 b): normalize the two center-type rejection parameters into plain numbers; called only when at least one is not None.

    * ``min_center_dist = (d, points)``: reject if the distance from the candidate center to any point in ``points`` is ``< d``;
      elements of ``points`` are xy (list/array/tensor) or a prefabricated obstacle triple (its center is used); may be empty.
    * ``center_exclusion = (center_xy, radius)``: reject if the distance from the candidate center to ``center_xy`` is ``< radius``.
    Returns ``(d, points(N,2), zone_center(2,), zone_radius)``; entries of disabled rules are None.
    """
    d = points = zone_center = zone_radius = None
    if min_center_dist is not None:
        try:
            d, raw_points = min_center_dist
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{who}: min_center_dist must be (d, points), got {min_center_dist!r}"
            ) from exc
        d = float(d)
        if not d >= 0.0:
            raise ValueError(f"{who}: d of min_center_dist must be ≥ 0, got {d}")
        pts = [_center_rule_point_xy(point) for point in raw_points]
        points = np.stack(pts) if pts else np.zeros((0, 2), dtype=np.float64)
    if center_exclusion is not None:
        try:
            raw_center, zone_radius = center_exclusion
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{who}: center_exclusion must be (center_xy, radius), got {center_exclusion!r}"
            ) from exc
        zone_center = _center_rule_point_xy(raw_center)
        zone_radius = float(zone_radius)
        if not zone_radius >= 0.0:
            raise ValueError(
                f"{who}: radius of center_exclusion must be ≥ 0, got {zone_radius}"
            )
    return d, points, zone_center, zone_radius


def _center_rules_violation(rules, x, y):
    """Pure numeric check (draws no random numbers): returns a description if any center rule is violated, else None."""
    d, points, zone_center, zone_radius = rules
    c = np.array([x, y], dtype=np.float64)
    if d is not None and len(points):
        dist = np.linalg.norm(points - c, axis=1)
        k = int(np.argmin(dist))
        if dist[k] < d:
            return (
                f"center distance to reference point {k} is {float(dist[k]):.6f} < {d}"
            )
    if zone_center is not None:
        r = float(np.linalg.norm(c - zone_center))
        if r < zone_radius:
            return (
                f"distance from center to exclusion-zone center {r:.6f} < {zone_radius}"
            )
    return None


def _assert_center_rules_hold(rules, x, y, who, spec_path):
    """N17: replayed frozen / injected values bypass the rejection loop and must be rechecked with the same rules; a violation raises instead of silently returning an invalid layout."""
    why = _center_rules_violation(rules, x, y)
    if why is not None:
        raise _EpisodeSpecError(
            f"{who}: frozen or injected center ({x}, {y}) ({spec_path}) violates the center rules: {why}"
        )


def _normalize_region_rules(annulus, base_band, segment_clearance, push_feasible, who):
    """V6: normalize the four region-type parameters into plain numbers; called only when at least one is not None (draws no random numbers).

    * ``annulus = (center_xy, r_in, r_out)``: the candidate center's distance to ``center_xy`` must lie in ``[r_in, r_out]``;
    * ``base_band = (base_xy, d_min, d_max)``: the candidate center's horizontal distance to the robot base must lie in ``[d_min, d_max]``;
    * ``segment_clearance = [(a_xy, b_xy, gap), ...]``: the candidate center's nearest distance to each segment ``ab`` must be ``≥ gap``;
    * ``push_feasible = (target_xy, d_min, d_max, backoff, lateral, base_xy, r_min, r_max)``:
      the candidate (cube) center's distance to ``target_xy`` must lie in ``[d_min, d_max]``, and the push start points
      ``c − backoff·d − s·lateral·n`` (``d`` the unit push direction, ``n`` its left normal, ``s∈{0,+1,−1}``)
      must all lie within ``[r_min, r_max]`` of ``base_xy``.
    Returns a dict; entries of disabled rules are None.
    """
    rules = {"annulus": None, "base_band": None, "segments": None, "push": None}
    if annulus is not None:
        try:
            c, r_in, r_out = annulus
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{who}: annulus must be (center_xy, r_in, r_out), got {annulus!r}"
            ) from exc
        r_in, r_out = float(r_in), float(r_out)
        if not (0.0 <= r_in <= r_out):
            raise ValueError(
                f"{who}: annulus must satisfy 0 ≤ r_in ≤ r_out, got ({r_in}, {r_out})"
            )
        rules["annulus"] = (_center_rule_point_xy(c), r_in, r_out)
    if base_band is not None:
        try:
            b, lo, hi = base_band
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{who}: base_band must be (base_xy, d_min, d_max), got {base_band!r}"
            ) from exc
        lo, hi = float(lo), float(hi)
        if not (0.0 <= lo <= hi):
            raise ValueError(
                f"{who}: base_band must satisfy 0 ≤ d_min ≤ d_max, got ({lo}, {hi})"
            )
        rules["base_band"] = (_center_rule_point_xy(b), lo, hi)
    if segment_clearance is not None:
        segs = []
        for item in segment_clearance:
            try:
                a, b, gap = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{who}: segment_clearance items must be (a_xy, b_xy, gap), got {item!r}"
                ) from exc
            gap = float(gap)
            if not gap >= 0.0:
                raise ValueError(
                    f"{who}: gap of segment_clearance must be ≥ 0, got {gap}"
                )
            segs.append((_center_rule_point_xy(a), _center_rule_point_xy(b), gap))
        rules["segments"] = segs
    if push_feasible is not None:
        try:
            tgt, d_min, d_max, backoff, lateral, base, r_min, r_max = push_feasible
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{who}: push_feasible must be (target_xy, d_min, d_max, backoff, lateral, base_xy, "
                f"r_min, r_max), got {push_feasible!r}"
            ) from exc
        vals = [float(v) for v in (d_min, d_max, backoff, lateral, r_min, r_max)]
        if not (
            0.0 < vals[0] <= vals[1]
            and vals[2] >= 0.0
            and vals[3] >= 0.0
            and 0.0 <= vals[4] <= vals[5]
        ):
            raise ValueError(f"{who}: invalid push_feasible values: {push_feasible!r}")
        rules["push"] = (
            _center_rule_point_xy(tgt),
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            _center_rule_point_xy(base),
            vals[4],
            vals[5],
        )
    return rules


def point_segment_distance_xy(p, a, b):
    """Nearest distance from point ``p`` to segment ``ab`` (xy, float64; degenerates to point distance when ``a == b``)."""
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom <= 0.0 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _region_rules_violation(rules, x, y):
    """Pure numeric check (draws no random numbers): returns a description if any region rule is violated, else None."""
    c = np.array([x, y], dtype=np.float64)
    if rules["annulus"] is not None:
        center, r_in, r_out = rules["annulus"]
        r = float(np.linalg.norm(c - center))
        if not (r_in <= r <= r_out):
            return (
                f"center is {r:.6f} from the annulus center, outside [{r_in}, {r_out}]"
            )
    if rules["base_band"] is not None:
        base, lo, hi = rules["base_band"]
        r = float(np.linalg.norm(c - base))
        if not (lo <= r <= hi):
            return f"center is {r:.6f} from the base, outside [{lo}, {hi}]"
    if rules["segments"]:
        for k, (a, b, gap) in enumerate(rules["segments"]):
            dist = point_segment_distance_xy(c, a, b)
            if dist < gap:
                return f"center is {dist:.6f} from segment {k} < {gap}"
    if rules["push"] is not None:
        tgt, d_min, d_max, backoff, lateral, base, r_min, r_max = rules["push"]
        d = tgt - c
        n = float(np.linalg.norm(d))
        if not (d_min <= n <= d_max):
            return f"push distance {n:.6f} is outside [{d_min}, {d_max}]"
        d = d / n
        lat = np.array([-d[1], d[0]], dtype=np.float64)
        for s in (0.0, 1.0, -1.0):
            start = c - backoff * d - s * lateral * lat
            r = float(np.linalg.norm(start - base))
            if not (r_min <= r <= r_max):
                return f"push start (lateral offset {s:+.0f}) is {r:.6f} from the base, outside [{r_min}, {r_max}]"
    return None


def _assert_region_rules_hold(rules, x, y, who, spec_path):
    """N17: replayed frozen / injected values are rechecked with the same region rules; a violation raises."""
    why = _region_rules_violation(rules, x, y)
    if why is not None:
        raise _EpisodeSpecError(
            f"{who}: frozen or injected center ({x}, {y}) ({spec_path}) violates the region rules: {why}"
        )


def spawn_random_cube(
    self,
    region_center=[0, 0],
    region_half_size=0.1,
    half_size=0.01,
    color=(1, 0, 0, 1),
    name_prefix="cube_extra",
    min_gap=0.005,
    max_trials=256,
    avoid=None,
    random_yaw=True,
    include_existing=True,
    include_goal=True,
    generator=None,
    fixed_xy=None,
    fixed_yaw=None,
    recorder=None,
    spec_path=None,
    corner_bias=0.0,
    min_center_dist=None,
    center_exclusion=None,
    annulus=None,
    base_band=None,
    segment_clearance=None,
    push_feasible=None,
    object_builder=None,
):
    """Drop a cube (onto table) in rectangular region using rejection sampling, and return the cube actor.
    - Uses OBB precise collision (2D projection + SAT), places only if min_gap is satisfied.
    - avoid: Input a list of objects. Can be [actor, ...] or [(actor, pad), ...] (pad in meters).
    - generator: Must pass torch.Generator for randomization.
    - object_builder: Optional task-specific factory with the same call arguments
      as actors.build_cube; its geometry must fit inside the cube envelope used
      for placement checks. Other tasks retain the native cube factory.

    New-value injection channel (NEW_VALUE_INJECTION_TEST_PLAN step 0a): when ``fixed_xy`` (meters) and optionally
    ``fixed_yaw`` (radians) are passed, the **rejection-sampling loop is skipped** and the cube is built directly at that
    pose without drawing any random number, so this path does not even need ``generator``. Creation still goes through
    the same ``_finalize_cube`` below (the original ``actors.build_cube`` segment), so the injected and original versions
    build identical actors apart from pose. With neither argument, the function behaves exactly as before.

    V5 center-type rejection rules (NEWTASK_RELEASE_V5_PLAN L4 b, xhard only, default None):
    ``min_center_dist=(d, points)`` / ``center_exclusion=(center_xy, radius)`` test the **cube center**
    (the ``(x, y)`` in the rejection loop below, also the xy used to build the cube). They are evaluated after the
    existing OBB/circle criteria and before ``recorder.value``; a violation ``continue``s to resample, draws no random
    numbers itself, and only changes the number of trials. Replay (``recorder.value`` returning a frozen value) and
    ``fixed_xy`` injection bypass the rejection loop and are rechecked with the same rules; a violation raises
    ``EpisodeSpecError`` (N17). With both parameters None no new check is executed and behavior is bit-identical to
    before the change.

    V6 region-type rejection rules (plan 2.6, for the MoveCube unified region U only, default None): ``annulus`` /
    ``base_band`` / ``segment_clearance`` / ``push_feasible`` (see ``_normalize_region_rules`` for semantics) likewise
    test the **cube center**, are ordered after the V5 center rules and before ``recorder.value``, and a violation
    ``continue``s; replay and injection are rechecked with the same rules (N17). With all four parameters None the whole
    block is skipped and existing callers are bit-identical.
    """
    # Cache
    if not hasattr(self, "_spawned_cubes"):
        self._spawned_cubes = []
        self._spawned_count = 0

    center = np.array(
        region_center if region_center is not None else self.cube_spawn_center,
        dtype=np.float64,
    )

    # Support two types of input: scalar or 2D array
    if region_half_size is None:
        region_half_size = self.cube_spawn_half_size

    # Compatible with two input formats
    if isinstance(region_half_size, (list, tuple, np.ndarray)):
        # 2D array input: independent control for xy
        area_half = np.array(region_half_size, dtype=np.float64)
        if area_half.shape == ():  # Handle 0-dim array
            area_half = np.array([float(area_half), float(area_half)], dtype=np.float64)
        elif len(area_half) == 1:
            area_half = np.array(
                [float(area_half[0]), float(area_half[0])], dtype=np.float64
            )
        elif len(area_half) != 2:
            raise ValueError(
                "region_half_size array must contain 1 or 2 elements [x_half, y_half]"
            )
    else:
        # Scalar input: xy remain consistent
        area_half = np.array(
            [float(region_half_size), float(region_half_size)], dtype=np.float64
        )

    hs_new = float(half_size if half_size is not None else self.cube_half_size)

    # Let cube fall completely inside region (independent control for xy)
    x_low = center[0] - area_half[0] + hs_new
    x_high = center[0] + area_half[0] - hs_new
    y_low = center[1] - area_half[1] + hs_new
    y_high = center[1] + area_half[1] - hs_new
    if x_low > x_high or y_low > y_high:
        raise ValueError(
            "spawn_random_cube: Sampling region too small, cannot fit cube of this size."
        )

    # === Assemble Obstacle OBB (2D) List ===
    obb2d_list = []  # [(c, A, h), ...]

    if include_existing:
        # Main cube
        if hasattr(self, "cube") and self.cube is not None:
            _append_actor_obstacles(obb2d_list, self.cube, pad=0.0)

        # Historically spawned cubes
        for ac in self._spawned_cubes:
            _append_actor_obstacles(obb2d_list, ac, pad=0.0)

    # User specified extra avoidance
    if avoid:
        for it in avoid:
            if isinstance(it, tuple):
                # Check if it's a pre-made OBB tuple (c, A, h) or (actor, pad)
                if (
                    len(it) == 3
                    and isinstance(it[0], np.ndarray)
                    and isinstance(it[1], np.ndarray)
                ):
                    # Pre-made OBB: (center, axes, half_sizes)
                    obb2d_list.append(it)
                else:
                    # Actor with padding
                    act_i, pad_i = it
                    _append_actor_obstacles(obb2d_list, act_i, pad=float(pad_i))
            else:
                _append_actor_obstacles(obb2d_list, it, pad=0.0)

    # Target point (if no mesh), supplement with "circle + circumscribed circle" conservative approximation (optional)
    circle_list = []  # [(xy(2,), R)], for objects without mesh

    def _actor_xy(actor):
        p = actor.pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        return np.array(p[:2], dtype=np.float64)

    if include_goal and hasattr(self, "goal_site") and self.goal_site is not None:
        try:
            # If goal_site has mesh, it will be covered in _push_actor_as_obb2d, here only as a fallback
            _append_actor_obstacles(obb2d_list, self.goal_site, pad=0.0)
        except Exception:
            # Degrade to circle approximation: goal radius + new cube circumscribed circle radius
            R_goal = float(getattr(self, "goal_thresh", 0.03))
            R_new_ext = np.sqrt(2.0) * hs_new
            circle_list.append(
                (_actor_xy(self.goal_site), R_goal + R_new_ext + min_gap)
            )

    # === Sampling Iteration ===
    # The fixed-value path draws no random numbers, so generator is not required there.
    if generator is None and fixed_xy is None:
        raise ValueError(
            "spawn_random_cube: generator argument must be explicitly passed for randomization"
        )

    device = self.device

    center_rules = None
    if min_center_dist is not None or center_exclusion is not None:
        center_rules = _normalize_center_rules(
            min_center_dist, center_exclusion, "spawn_random_cube"
        )
    region_rules = None
    if (
        annulus is not None
        or base_band is not None
        or segment_clearance is not None
        or push_feasible is not None
    ):
        region_rules = _normalize_region_rules(
            annulus, base_band, segment_clearance, push_feasible, "spawn_random_cube"
        )

    def _finalize_cube(x, y, yaw):
        """The creation segment formerly inlined in the loop, shared by the original path and the fixed-value path so the two copies of creation code cannot drift."""
        q = _yaw_to_quat_tensor(yaw, device=device)

        cube = (actors.build_cube if object_builder is None else object_builder)(
            self.scene,
            half_size=hs_new,
            color=color,
            name=name_prefix,  # Use name_prefix directly, do not add counter
            initial_pose=Pose.create_from_pq(
                torch.tensor([[x, y, hs_new]], device=device, dtype=torch.float32),
                q,
            ),
        )
        cube._cube_half_size = hs_new
        self._spawned_cubes.append(cube)
        self._spawned_count += 1
        return cube

    if fixed_xy is not None:
        # Pose is fixed by the external spec: skip rejection sampling and geometry checks (feasibility was screened before freezing).
        if center_rules is not None:
            _assert_center_rules_hold(
                center_rules,
                float(fixed_xy[0]),
                float(fixed_xy[1]),
                "spawn_random_cube",
                spec_path,
            )
        if region_rules is not None:
            _assert_region_rules_hold(
                region_rules,
                float(fixed_xy[0]),
                float(fixed_xy[1]),
                "spawn_random_cube",
                spec_path,
            )
        return _finalize_cube(
            float(fixed_xy[0]),
            float(fixed_xy[1]),
            float(fixed_yaw) if fixed_yaw is not None else 0.0,
        )

    for trial in range(int(max_trials)):
        # Use simple uniform sampling to ensure good spatial coverage
        # Complex sampling strategies often reduce coverage

        u1 = torch.rand(1, generator=generator).item()
        u2 = torch.rand(1, generator=generator).item()
        if corner_bias:
            # Deterministic mapping only: must not draw extra random numbers; skipped when corner_bias=0.
            u1 = corner_push(u1, corner_bias)
            u2 = corner_push(u2, corner_bias)

        # Map directly to sampling region - Uniform distribution provides best spatial coverage
        x = float(x_low + u1 * (x_high - x_low))
        y = float(y_low + u2 * (y_high - y_low))

        if random_yaw:
            # Use more random yaw generation, sampling from full [0, 2π] range
            yaw_sample = torch.rand(1, generator=generator).item()
            yaw = float(yaw_sample * 2 * np.pi)
        else:
            yaw = 0.0

        # New cube's 2D OBB (reflect min_gap in "new object half length expansion", avoid adding to both sides causing double)
        c_new, A_new, h_new = _build_new_cube_obb2d(
            x, y, hs_new, yaw, pad_xy=float(min_gap)
        )

        # Check collision one by one with OBB obstacles
        hit = False
        for c_obs, A_obs, h_obs in obb2d_list:
            if _obb2d_intersect(c_obs, A_obs, h_obs, c_new, A_new, h_new):
                hit = True
                break
        if hit:
            continue

        # Check circular conservative obstacles (if present)
        for xy_c, R_c in circle_list:
            if np.linalg.norm(np.asarray([x, y], dtype=np.float64) - xy_c) < R_c:
                hit = True
                break
        if hit:
            continue

        # Center rules run after the existing criteria and before recorder.value; skipped when None.
        if (
            center_rules is not None
            and _center_rules_violation(center_rules, x, y) is not None
        ):
            continue
        # Region rules run after the center rules; skipped when None.
        if (
            region_rules is not None
            and _region_rules_violation(region_rules, x, y) is not None
        ):
            continue

        # Passing detection, create cube (pose and collision detection use same yaw to ensure consistency)
        if recorder is not None and spec_path is not None:
            # Failed rejection-sampling attempts still happen (random stream must not drift); only the accepted pose is frozen.
            x, y, yaw = recorder.value(spec_path, [x, y, yaw])
            if center_rules is not None:
                _assert_center_rules_hold(
                    center_rules, x, y, "spawn_random_cube", spec_path
                )
            if region_rules is not None:
                _assert_region_rules_hold(
                    region_rules, x, y, "spawn_random_cube", spec_path
                )
        return _finalize_cube(x, y, yaw)

    raise RuntimeError(
        "spawn_random_cube: Region crowded or constraints too tight, no feasible position found. Try: increase region/decrease cube/decrease min_gap."
    )


def spawn_random_target(
    self,
    region_center=[0, 0],
    region_half_size=0.1,
    radius=0.01,
    thickness=0.005,
    name_prefix="target_extra",
    min_gap=0.005,
    max_trials=256,
    avoid=None,  # Supports [actor, ...] or [(actor, pad), ...]
    include_existing=True,  # Whether to automatically avoid existing main target and generated extra targets
    include_goal=True,  # Whether to treat goal_site as obstacle (approximate with circle, conservative)
    generator=None,
    randomize=True,  # Control whether to randomize position
    target_style="purple",  # Choose which color scheme target to create
    recorder=None,
    spec_path=None,
    min_center_dist=None,
    center_exclusion=None,
    annulus=None,
    base_band=None,
    segment_clearance=None,
    push_feasible=None,
):
    """Drop a target (onto table) in rectangular region using rejection sampling, and return the target actor.
    - Uses OBB precise collision (2D projection + SAT), places only if min_gap is satisfied.
    - avoid: Input a list of objects. Can be [actor, ...] or [(actor, pad), ...] (pad in meters).
    - generator: Must pass torch.Generator for randomization (when randomize=True).
    - randomize: Control whether to randomize position. If False, generate directly at region_center.

    V5 center-type rejection rules (L4 b, xhard only, default None): same semantics as the same-named parameters of
    ``spawn_random_cube``, but they test the **disc center** (the ``(x, y)`` in the rejection loop); replayed frozen
    values are rechecked with the same rules (N17). With both parameters None no new check is executed and behavior is
    bit-identical to before the change.
    V6 region rules (``annulus`` / ``base_band`` / ``segment_clearance`` / ``push_feasible``) are as in
    ``spawn_random_cube`` and test the disc center; with all four None the whole block is skipped.
    """
    # Cache
    random_yaw = False
    if not hasattr(self, "_spawned_targets"):
        self._spawned_targets = []
        self._spawned_target_count = 0

    center = np.array(
        region_center
        if region_center is not None
        else getattr(self, "target_spawn_center", [0, 0]),
        dtype=np.float64,
    )
    area_half = float(
        region_half_size
        if region_half_size is not None
        else getattr(self, "target_spawn_half_size", 0.1)
    )
    target_radius = float(
        radius if radius is not None else getattr(self, "target_radius", 0.01)
    )
    target_thickness = float(
        thickness if thickness is not None else getattr(self, "target_thickness", 0.005)
    )

    # Let target fall completely inside region
    x_low = center[0] - area_half + target_radius
    x_high = center[0] + area_half - target_radius
    y_low = center[1] - area_half + target_radius
    y_high = center[1] + area_half - target_radius
    if x_low > x_high or y_low > y_high:
        raise ValueError(
            "spawn_random_target: Sampling region too small, cannot fit target of this size."
        )

    # === Assemble Obstacle OBB (2D) List ===
    obb2d_list = []  # [(c, A, h), ...]

    if include_existing:
        # Main cube
        if hasattr(self, "cube") and self.cube is not None:
            _append_actor_obstacles(obb2d_list, self.cube, pad=0.0)

        # Main target
        if hasattr(self, "target") and self.target is not None:
            _append_actor_obstacles(obb2d_list, self.target, pad=0.0)

        # Historically spawned cubes
        if hasattr(self, "_spawned_cubes"):
            for ac in self._spawned_cubes:
                _append_actor_obstacles(obb2d_list, ac, pad=0.0)

    # Target point (if no mesh), supplement with "circle + circumscribed circle" conservative approximation (optional)
    circle_list = []  # [(xy(2,), R)], for objects without mesh

    def _actor_xy(actor):
        p = actor.pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        return np.array(p[:2], dtype=np.float64)

    # Historically spawned targets - Treat as circular obstacles
    if include_existing:
        for ac in self._spawned_targets:
            target_r = getattr(ac, "_target_radius", target_radius)
            circle_list.append((_actor_xy(ac), target_r))

    # User specified extra avoidance
    if avoid:
        for it in avoid:
            if isinstance(it, tuple):
                # Check if it's a pre-made OBB tuple (c, A, h) or (actor, pad)
                if (
                    len(it) == 3
                    and isinstance(it[0], np.ndarray)
                    and isinstance(it[1], np.ndarray)
                ):
                    # Pre-made OBB: (center, axes, half_sizes)
                    obb2d_list.append(it)
                else:
                    # Actor with padding
                    act_i, pad_i = it
                    # Check if it is a target (circular)
                    if hasattr(act_i, "_target_radius"):
                        target_r = getattr(act_i, "_target_radius", target_radius)
                        circle_list.append((_actor_xy(act_i), target_r + float(pad_i)))
                    else:
                        _append_actor_obstacles(obb2d_list, act_i, pad=float(pad_i))
            else:
                # Check if it is a target (circular)
                if hasattr(it, "_target_radius"):
                    target_r = getattr(it, "_target_radius", target_radius)
                    circle_list.append((_actor_xy(it), target_r))
                else:
                    _append_actor_obstacles(obb2d_list, it, pad=0.0)

    if include_goal and hasattr(self, "goal_site") and self.goal_site is not None:
        try:
            # If goal_site has mesh, it will be covered in _push_actor_as_obb2d, here only as a fallback
            _append_actor_obstacles(obb2d_list, self.goal_site, pad=0.0)
        except Exception:
            # Degrade to circle approximation: goal radius + new target circumscribed circle radius
            R_goal = float(getattr(self, "goal_thresh", 0.03))
            R_new_ext = target_radius
            circle_list.append(
                (_actor_xy(self.goal_site), R_goal + R_new_ext + min_gap)
            )

    # === Sampling Iteration ===
    if generator is None:
        raise ValueError(
            "spawn_random_target: generator argument must be explicitly passed for randomization"
        )

    device = self.device

    target_builders = {
        "purple": build_purple_white_target,
        "gray": build_gray_white_target,
        "green": build_green_white_target,
        "red": build_red_white_target,
    }
    if isinstance(target_style, str):
        builder_key = target_style.lower()
        if builder_key not in target_builders:
            raise ValueError(
                f"spawn_random_target: Unknown target_style '{target_style}'. Supported: {list(target_builders.keys())}"
            )
        target_builder = target_builders[builder_key]
    elif callable(target_style):
        target_builder = target_style
    else:
        raise ValueError(
            "spawn_random_target: target_style must be a string or callable builder function"
        )

    center_rules = None
    if min_center_dist is not None or center_exclusion is not None:
        center_rules = _normalize_center_rules(
            min_center_dist, center_exclusion, "spawn_random_target"
        )
    region_rules = None
    if (
        annulus is not None
        or base_band is not None
        or segment_clearance is not None
        or push_feasible is not None
    ):
        region_rules = _normalize_region_rules(
            annulus, base_band, segment_clearance, push_feasible, "spawn_random_target"
        )

    for _ in range(int(max_trials)):
        x = float(torch.rand(1, generator=generator).item() * (x_high - x_low) + x_low)
        y = float(torch.rand(1, generator=generator).item() * (y_high - y_low) + y_low)

        if random_yaw:
            yaw = float(torch.rand(1, generator=generator).item() * 2 * np.pi - np.pi)
        else:
            yaw = 0.0

        # New target's circular collision detection (target is circular, circular detection is more accurate)
        target_pos = np.array([x, y], dtype=np.float64)
        target_collision_radius = target_radius + min_gap

        # Check collision with OBB obstacles (check circular target against square obstacles)
        hit = False
        for c_obs, A_obs, h_obs in obb2d_list:
            # Calculate minimum distance from circle center to OBB
            # Convert circle center to OBB local coordinate system
            local_pos = A_obs.T @ (target_pos - c_obs)
            # Calculate closest point from circle center to OBB
            closest_point = np.clip(local_pos, -h_obs, h_obs)
            # Convert back to world coordinate system
            closest_world = c_obs + A_obs @ closest_point
            # Calculate distance
            dist = np.linalg.norm(target_pos - closest_world)
            if dist < target_collision_radius:
                hit = True
                break
        if hit:
            continue

        # Check collision with circular obstacles (circle vs circle)
        for xy_c, R_c in circle_list:
            if np.linalg.norm(target_pos - xy_c) < (target_collision_radius + R_c):
                hit = True
                break
        if hit:
            continue

        # Center rules run after the existing criteria and before recorder.value; skipped when None.
        if (
            center_rules is not None
            and _center_rules_violation(center_rules, x, y) is not None
        ):
            continue
        # Region rules run after the center rules; skipped when None.
        if (
            region_rules is not None
            and _region_rules_violation(region_rules, x, y) is not None
        ):
            continue

        # Passed detection, create target (pose and collision detection use same yaw to ensure consistency)
        if recorder is not None and spec_path is not None:
            # Failed attempts still happen (random stream must not drift); only the accepted xy is frozen.
            x, y = recorder.value(spec_path, [x, y])
            if center_rules is not None:
                _assert_center_rules_hold(
                    center_rules, x, y, "spawn_random_target", spec_path
                )
            if region_rules is not None:
                _assert_region_rules_hold(
                    region_rules, x, y, "spawn_random_target", spec_path
                )
        rotate = np.array(
            [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        )  # Quaternion for z-axis rotation
        angles = torch.deg2rad(
            torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32)
        )  # (3,)
        rotate = matrix_to_quaternion(euler_angles_to_matrix(angles, convention="XYZ"))
        target = target_builder(
            scene=self.scene,
            radius=target_radius,
            thickness=target_thickness,
            name=name_prefix,  # Use name_prefix directly, do not add counter
            body_type="kinematic",  # Visualization only
            add_collision=False,  # Disable collision
            initial_pose=sapien.Pose(p=[x, y, target_thickness], q=rotate),
        )
        target._target_radius = target_radius
        self._spawned_targets.append(target)
        self._spawned_target_count += 1
        return target

    raise RuntimeError(
        "spawn_random_target: Region crowded or constraints too tight, no feasible position found. Try: increase region/decrease target/decrease min_gap."
    )


def create_button_obb(center_xy=(-0.3, 0), half_size=0.05):
    """
    Create a manual OBB for button collision avoidance.

    Args:
        center_xy: Button center position (x, y)
        half_size: Safe zone half-size around button (default 0.05m)

    Returns:
        Tuple (center, axes, half_sizes) for use in avoid lists
    """
    return (
        np.array(center_xy, dtype=np.float64),  # center
        np.eye(2, dtype=np.float64),  # axes (identity for axis-aligned)
        np.array([half_size, half_size], dtype=np.float64),  # half-sizes
    )


def build_button(
    self,
    center_xy=(0.15, 0.10),  # Button (x,y) on table
    base_half=[0.025, 0.025, 0.005],  # Base half-size [x,y,z]
    cap_radius=0.015,  # Button cap radius
    cap_half_len=0.006,  # Button cap half-length
    travel=None,  # Press travel
    stiffness=800.0,
    damping=40.0,
    scale: float = None,  # ⭐ New: scaling factor
    generator=None,
    name: str = "button",  # ⭐ New: button name
    randomize: bool = True,  # ⭐ New: whether to randomize position
    randomize_range=(0.1, 0.4),  # ⭐ New: randomization range, (range_x, range_y)
    recorder=None,
    spec_path=None,
):
    # ------- Scaling and Travel -------
    if scale is None:
        # If not passed, use default scaling from environment
        scale = getattr(self, "button_scale", 1.0)
    scale = float(scale)

    # Travel priority: argument > environment base
    if travel is None:
        # Scale proportionally using base travel
        base_travel = getattr(self, "_button_travel_base", 0.1)
        travel = base_travel * scale
    else:
        # If travel explicitly passed, also follow scale (to keep absolute value, change next line to pass)
        travel = float(travel) * scale

    # Size scaling
    base_half = [bh * scale for bh in base_half]
    cap_radius = float(cap_radius) * scale
    cap_half_len = float(cap_half_len) * scale

    # Record current button travel for other functions
    self.button_travel = float(travel)

    # ------- Position Randomization -------
    cx, cy = float(center_xy[0]), float(center_xy[1])

    if randomize:
        if not isinstance(randomize_range, (tuple, list, np.ndarray)):
            raise TypeError("randomize_range must be a sequence of length 2.")
        if len(randomize_range) != 2:
            raise ValueError("randomize_range must contain exactly two elements.")
        range_x, range_y = float(randomize_range[0]), float(randomize_range[1])
        offset = torch.rand(2, generator=generator) - 0.5
        cx += float(offset[0]) * range_x
        cy += float(offset[1]) * range_y
    if recorder is not None and spec_path is not None:
        # Original sampling still runs (random stream must not drift); in injection mode the frozen value builds the button.
        cx, cy = recorder.value(spec_path, [cx, cy])
    center_xy = (cx, cy)

    scene = self.scene
    builder = scene.create_articulation_builder()

    # Initial pose: lift base center to z=base_half[2]
    builder.initial_pose = sapien.Pose(p=[cx, cy, base_half[2]])

    # Root: Base
    base = builder.create_link_builder()
    base.set_name("button_base")
    base.add_box_collision(half_size=base_half, density=200000)
    base.add_box_visual(half_size=base_half)

    # Child: Button cap (vertical sliding)
    cap = builder.create_link_builder(base)
    cap.set_name("button_cap")
    cap.set_joint_name("button_joint")

    R_up = euler2quat(0, -np.pi / 2, 0)  # Align joint x-axis with world z

    cap.set_joint_properties(
        type="prismatic",
        limits=[[-travel, 0.0]],  # Negative direction is pressed
        pose_in_parent=sapien.Pose(p=[0, 0, base_half[2]], q=R_up),
        pose_in_child=sapien.Pose(p=[0, 0, 0.0], q=R_up),
        friction=0.0,
        damping=0.0,
    )

    cap.add_cylinder_collision(
        half_length=cap_half_len,
        radius=cap_radius,
        pose=sapien.Pose(p=[0, 0, cap_half_len], q=R_up),
        density=1500,
    )
    material = sapien.render.RenderMaterial()
    material.set_base_color([0.5, 0.5, 0.5, 1.0])
    cap.add_cylinder_visual(
        half_length=cap_half_len,
        radius=cap_radius,
        pose=sapien.Pose(p=[0, 0, cap_half_len], q=R_up),
        material=material,
    )

    button = builder.build(name=name, fix_root_link=True)

    j = {j.name: j for j in button.get_joints()}["button_joint"]
    j.set_drive_properties(stiffness=stiffness, damping=damping)
    j.set_drive_target(0.0)

    self.button = button
    self.button_joint = j

    cap_link = next(
        link for link in button.get_links() if link.get_name() == "button_cap"
    )
    cap_link = next(
        link for link in button.get_links() if link.get_name() == "button_cap"
    )
    if not hasattr(self, "cap_links"):
        self.cap_links = {}
    self.cap_links[name] = [cap_link]  # name is "button_left", "button_right", etc.
    self.cap_link = self.cap_links[name]  # Compatible with old logic

    # Provide an OBB for downstream placement logic using the scaled button footprint
    button_obb = create_button_obb(
        center_xy=center_xy,
        half_size=max(base_half[0], base_half[1]) * 1.5,
    )
    return button_obb


def spawn_fixed_cube(
    self,
    position,  # [x, y, z] fixed position
    half_size=None,
    color=(1, 0, 0, 1),
    name_prefix="fixed_cube",
    yaw=0.0,  # rotation around z-axis (radians)
    dynamic=False,
):
    """
    Generate a cube at fixed position, no collision detection.
    Use builder pattern to create dynamic object, refer to build_bin implementation.
    """
    hs = float(half_size if half_size is not None else self.cube_half_size)

    # Ensure position is array format
    pos = np.array(position, dtype=np.float64)
    if len(pos) == 2:
        # If only x,y provided, set z to cube half height (let cube bottom sit on table)
        pos = np.append(pos, hs)

    # Create actor builder
    builder = self.scene.create_actor_builder()

    # Generate rotation quaternion (rotate yaw angle around z-axis)
    if yaw != 0.0:
        angles = torch.tensor([0.0, 0.0, float(yaw)], dtype=torch.float32)
        R = euler_angles_to_matrix(angles.unsqueeze(0), convention="XYZ")[0]
        q = matrix_to_quaternion(R.unsqueeze(0))[0]
        rotate = q
    else:
        rotate = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    # Set initial position and rotation
    builder.set_initial_pose(
        sapien.Pose(
            p=[pos[0], pos[1], pos[2]],
            q=rotate.numpy() if isinstance(rotate, torch.Tensor) else rotate,
        )
    )

    # Add box geometry (collision and visual)
    half_size_list = [hs, hs, hs]
    if dynamic == True:
        # Collision geometry stays at builder origin; initial pose already positions the actor
        builder.add_box_collision(sapien.Pose([0, 0, 0]), half_size_list)

    # Create material
    material = sapien.render.RenderMaterial()
    material.set_base_color(color)
    builder.add_box_visual(sapien.Pose([0, 0, 0]), half_size_list, material=material)

    # Choose build method based on dynamic argument
    if dynamic == True:
        cube = builder.build_dynamic(name=name_prefix)
    else:
        cube = builder.build_kinematic(name=name_prefix)

    # Set cube attribute
    cube._cube_half_size = hs

    return cube


def build_board_with_hole(
    self,
    *,
    board_side=0.01,  # Square board side length
    hole_side=0.06,  # Square hole side length
    thickness=0.02,  # Board thickness
    position=None,  # Board position [x, y] or [x, y, z]
    rotation_quat=None,  # Rotation quaternion [w, x, y, z]
    name="board_with_hole",
):
    """
    Create a square board with a square hole
    Combine four rectangular strips: top, bottom, left, right

    Args:
        height: If provided, overwrite z coordinate in position
    """
    if position is None:
        position = [0.3, 0, 0]  # Default position, bottom on table

    # Board and hole half lengths
    board_half = board_side / 2
    hole_half = hole_side / 2
    thickness_half = thickness / 2

    # Use input position as board bottom, calculate board center position
    # Input position is bottom position, need to add thickness_half to get center position
    center_position = [position[0], position[1], position[2] + thickness_half]

    # Create actor builder
    builder = self.scene.create_actor_builder()

    # Set board initial position (use center position)
    if rotation_quat is None:
        rotation_quat = [1.0, 0.0, 0.0, 0.0]  # No rotation
    builder.set_initial_pose(sapien.Pose(p=center_position, q=rotation_quat))

    # Create material - brown board
    material = sapien.render.RenderMaterial()
    material.set_base_color([0.8, 0.6, 0.4, 1.0])  # Light brown

    # Four rectangular strips dimensions and positions
    # Top strip
    top_width = board_side  # Full board width
    top_height = board_half - hole_half  # From hole top to board top
    top_center_y = hole_half + top_height / 2
    builder.add_box_collision(
        sapien.Pose([0, top_center_y, 0]),
        [top_width / 2, top_height / 2, thickness_half],
    )
    builder.add_box_visual(
        sapien.Pose([0, top_center_y, 0]),
        [top_width / 2, top_height / 2, thickness_half],
        material=material,
    )

    # Bottom strip
    bottom_width = board_side  # Full board width
    bottom_height = board_half - hole_half  # From board bottom to hole bottom
    bottom_center_y = -(hole_half + bottom_height / 2)
    builder.add_box_collision(
        sapien.Pose([0, bottom_center_y, 0]),
        [bottom_width / 2, bottom_height / 2, thickness_half],
    )
    builder.add_box_visual(
        sapien.Pose([0, bottom_center_y, 0]),
        [bottom_width / 2, bottom_height / 2, thickness_half],
        material=material,
    )

    # Left strip - only within hole height range
    left_width = board_half - hole_half  # From board left to hole left
    left_height = hole_side  # Hole height
    left_center_x = -(hole_half + left_width / 2)
    builder.add_box_collision(
        sapien.Pose([left_center_x, 0, 0]),
        [left_width / 2, left_height / 2, thickness_half],
    )
    builder.add_box_visual(
        sapien.Pose([left_center_x, 0, 0]),
        [left_width / 2, left_height / 2, thickness_half],
        material=material,
    )

    # Right strip - only within hole height range
    right_width = board_half - hole_half  # From hole right to board right
    right_height = hole_side  # Hole height
    right_center_x = hole_half + right_width / 2
    builder.add_box_collision(
        sapien.Pose([right_center_x, 0, 0]),
        [right_width / 2, right_height / 2, thickness_half],
    )
    builder.add_box_visual(
        sapien.Pose([right_center_x, 0, 0]),
        [right_width / 2, right_height / 2, thickness_half],
        material=material,
    )

    # Add a black cube at hole center with same size as hole but half height (visual only, no collision)
    hole_cube_half_size_xy = hole_half  # cube half size same as hole
    hole_cube_half_height = thickness_half / 2  # cube height is half of board thickness

    # Create black material
    black_material = sapien.render.RenderMaterial()
    black_material.set_base_color([0.0, 0.0, 0.0, 1.0])  # Black

    # Add black cube (visual only, no collision)
    # Position: cube bottom at board bottom, so cube center at -thickness_half + hole_cube_half_height
    cube_center_z = -thickness_half + hole_cube_half_height
    builder.add_box_visual(
        sapien.Pose([0, 0, cube_center_z]),  # Black cube bottom at board bottom
        [hole_cube_half_size_xy, hole_cube_half_size_xy, hole_cube_half_height],
        material=black_material,
    )

    # Build actor
    board_actor = builder.build_kinematic(name=name)

    # Store board attributes
    board_actor._board_side = board_side
    board_actor._hole_side = hole_side
    board_actor._thickness = thickness

    return board_actor


def build_purple_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_PURPLE = (np.array([160, 32, 240, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_PURPLE),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_PURPLE),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_PURPLE),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_gray_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_GRAY = (np.array([128, 128, 128, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_GRAY),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GRAY),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GRAY),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_green_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_GREEN = (np.array([34, 139, 34, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_GREEN),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GREEN),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GREEN),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_red_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_RED = (np.array([200, 33, 33, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def _build_by_type(
    builder: ActorBuilder,
    name,
    body_type,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    if scene_idxs is not None:
        builder.set_scene_idxs(scene_idxs)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    else:
        raise ValueError(f"Unknown body type {body_type}")
    return actor


def _append_actor_obstacles(obb2d_list, actor, pad=0.0):
    try:
        # Special handling for board_with_hole
        if hasattr(actor, "_board_side") and hasattr(actor, "_hole_side"):
            # This is our board with hole, manually add its OBB
            board_side = actor._board_side
            hole_side = actor._hole_side

            # Get board world position
            actor_pos = actor.pose.p
            if isinstance(actor_pos, torch.Tensor):
                actor_pos = actor_pos[0].detach().cpu().numpy()

            board_center = np.array(actor_pos[:2], dtype=np.float64)
            board_half = board_side / 2
            hole_half = hole_side / 2

            # Add OBBs for four rectangular strips
            # Top strip
            if board_half > hole_half:  # Ensure enough space
                top_height = board_half - hole_half
                top_center = board_center + np.array([0, hole_half + top_height / 2])
                A_top = np.eye(2)  # No rotation
                h_top = np.array([board_half + pad, top_height / 2 + pad])
                obb2d_list.append((top_center, A_top, h_top))

                # Bottom strip
                bottom_center = board_center + np.array(
                    [0, -(hole_half + top_height / 2)]
                )
                obb2d_list.append((bottom_center, A_top, h_top))

                # Left strip
                left_width = board_half - hole_half
                left_center = board_center + np.array(
                    [-(hole_half + left_width / 2), 0]
                )
                h_left = np.array([left_width / 2 + pad, hole_half + pad])
                obb2d_list.append((left_center, A_top, h_left))

                # Right strip
                right_center = board_center + np.array([hole_half + left_width / 2, 0])
                obb2d_list.append((right_center, A_top, h_left))
            return

        obb = get_actor_obb(actor, to_world_frame=True, vis=False)
        obb2d = _trimesh_box_to_obb2d(obb, extra_pad=float(pad))
        obb2d_list.append(obb2d)
    except Exception:
        # Some objects (like site/marker) do not have physical mesh, ignore or use circle approximation below
        pass
