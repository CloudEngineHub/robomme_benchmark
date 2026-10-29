"""Method-independent layout sampling and physical geometry (no simulator imports)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from .diversity import sample_pan_loading_surface

METHODS = ("hook", "poke", "scoop", "ball", "cue")
HANDHELD_METHODS = METHODS[:3]
HALF = 0.02
TARGET_RADIUS = 0.04 * 2 / 3
TARGET_TOLERANCE = TOL = 0.025 * 2 / 3
CUE_TIP_RADIUS = 0.009
# A slowly contacted cube must remain visibly separate from the marker, not
# merely outside its small success tolerance. This is a cube-center region.
CUE_TARGET_NEIGHBORHOOD_M = TARGET_RADIUS + HALF * np.sqrt(2) + 0.010
CUE_STATIC_CLEARANCE_M = 0.015
MARGIN = 5
SLOT = 0.032
TABLE = np.array([[-0.715, -1.17], [0.44, 1.17]])
ROBOT = np.array([[-0.72, -0.10], [-0.52, 0.10]])
TASK_BOUNDS = CUBE_BOUNDS = TARGET_BOUNDS = np.array([[-0.36, -0.29], [0.18, 0.29]])
MAX_CUE_AXIS_ERROR_DEG = 18.0
MAX_CUE_CONTACT_OFFSET_M = 0.010
STOP_TOLERANCE_M = 0.001
BEARING_HALF_LENGTH_M = 0.018
ORIGINAL_TOOLS = {
    "hook": dict(
        tool_length_m=0.16,
        handle_half_size_m=[0.028, 0.014, 0.012],
        shaft_radius_m=0.005,
        hook_head_scale=0.80,
    ),
    "poke": dict(
        tool_length_m=0.20,
        handle_half_size_m=[0.028, 0.011, 0.011],
        shaft_radius_m=0.003,
        ball_radius_m=0.016,
    ),
    "scoop": dict(
        tool_length_m=0.18,
        handle_half_size_m=[0.028, 0.012, 0.012],
        shaft_radius_m=0.005,
        pan_length_m=0.11,
        pan_width_m=0.10,
    ),
}
DEFAULT_RANGES = {
    "ramp_length_m": (0.15, 0.30),
    "ramp_grade_deg": (10.0, 40.0),
    "ramp_width_m": (0.14, 0.29),
    "ball_radius_m": (0.018, 0.032),
    "ball_mass_kg": (0.10, 0.80),
    "cube_mass_kg": (0.025, 0.14),
    "cue_front_m": (0.11, 0.30),
    "cue_stroke_m": (0.04, 0.14),
    "pan_handle_length_m": (0.10, 0.15),
    "hook_length_m": (0.18, 0.40),
    "poke_length_m": (0.20, 0.40),
    "hook_head_scale": (0.75, 1.15),
    "poke_tip_radius_m": (0.013, 0.022),
    "pan_length_m": (0.065, 0.090),
    "pan_width_m": (0.060, 0.085),
}

# A pan requires the ramp for loading, and every ramp is paired with a ball.
# Keep exactly three available methods by reserving the ball slot when a pan
# is present, rather than adding a fourth method after layout generation.
COMBINATIONS = tuple(group for group in combinations(METHODS, 3)
                     if "scoop" not in group or "ball" in group)
PAN_LOADING_FACES = {
    "incline": ("inclined_face",),
    "wall": ("left_side", "right_side", "raised_end"),
    "side_wall": ("left_side", "right_side"),
}


def unit(angle):
    a = np.radians(angle)
    return np.array([np.cos(a), np.sin(a), 0.0])


def rotated(points, yaw, origin):
    a = np.radians(yaw)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.asarray(points) @ r.T + np.asarray(origin)[:2]


def rectangle(lo, hi, origin, yaw=0):
    return rotated(
        [[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]], [hi[0], lo[1]]], yaw, origin
    )


def bounds(points):
    p = np.asarray(points)
    return np.array([p.min(axis=0), p.max(axis=0)])


def overlap(a, b, gap=0):
    return bool(np.all(a[1] + gap > b[0]) and np.all(a[0] - gap < b[1]))


def project(points):
    p = np.asarray(points)
    x, y, z = p.T
    a = 1 / np.sqrt(5)
    depth = 1.1 * a - a * x - 2 * a * z
    return np.stack(
        [128 + 128 * y / depth, 128 + 128 * (2 * a * x - a * z - 0.2 * a) / depth],
        axis=1,
    ), depth


def visible(points, margin=MARGIN):
    uv, depth = project(points)
    return bool(
        np.all(depth > 0.01) and np.all(uv >= margin) and np.all(uv <= 255 - margin)
    )


def box_vertices(lo, hi):
    return np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )


def prism(poly, low=-0.002, high=0.04):
    return np.array([[*p, z] for p in poly for z in (low, high)])


def polygon(points):
    p = np.asarray(points)[:, :2]
    if len(p) == 2:
        lo, hi = p
        return np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    return p


def intersects(a, b, gap=0):
    """Separating-axis check, retaining the shape of diagonal corridors."""
    a, b = polygon(a), polygon(b)
    if not overlap(bounds(a), bounds(b), gap):
        return False
    for p in (a, b):
        for edge in np.roll(p, -1, axis=0) - p:
            if np.linalg.norm(edge) < 1e-8:
                continue
            axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            pa, pb = a @ axis, b @ axis
            if pa.max() + gap < pb.min() or pb.max() + gap < pa.min():
                return False
    return True


def clear(poly, occupied, gap=0.012):
    b = bounds(poly)
    return bool(
        np.all(b[0] > TABLE[0])
        and np.all(b[1] < TABLE[1])
        and not overlap(b, ROBOT, 0.018)
        and not any(intersects(poly, other, gap) for other in occupied)
    )


@dataclass(frozen=True)
class AlignedLayout:
    seed: int
    cube_xyz: tuple
    target_xyz: tuple
    target_heading_deg: float
    cube_mass_kg: float
    ramp_toe_xyz: tuple
    ramp_heading_deg: float
    ramp_length_m: float
    ramp_width_m: float
    ramp_grade_deg: float
    pan_face: str
    ball_radius: float
    ball_mass: float
    ball_rest_xyz: tuple
    cue_pivot_xyz: tuple
    cue_heading_deg: float
    cue_park_deg: float
    cue_front_m: float
    rim_label: str
    tool_poses: dict
    tool_params: dict
    handle_spacing_m: float
    pan_fraction: float = 0.73
    pan_route_xyz: tuple = ()
    cue_zone: str = ""
    pack_style: str = "hook beyond pan, poke alongside"
    pack_bounds_m: tuple = ()
    pack_area_m2: float = 0
    reference_pack_area_m2: float = 0
    cluster_bounds_m: tuple = ()
    cluster_area_m2: float = 0
    cue_sweep_polygons: tuple = ()
    sampling_order: tuple = ("cube_target", "cue", "handhelds_ball", "ramp")
    cube_yaw_deg: float | None = None
    pan_floor_friction: float | None = None
    ramp_slot_m: float | None = None
    parking_side: str = ""
    ramp_lateral_offset_m: float = 0
    diversity_profile: int = 0

    def __post_init__(self):
        # Derived parking metadata follows edited lengths and handle poses.
        # This also keeps saved-layout experiments from retaining stale boxes.
        footprints = [
            tool_poly(m, *self.tool_poses[m], self.tool_params[m])
            for m in HANDHELD_METHODS
        ]
        pack = bounds(np.concatenate(footprints))
        ball = rectangle(
            (-self.ball_radius, -self.ball_radius),
            (self.ball_radius, self.ball_radius),
            self.ball_rest_xyz,
        )
        cluster = bounds(np.concatenate([pack, ball]))
        for name, value in dict(
            pack_bounds_m=pack.tolist(),
            pack_area_m2=float(np.prod(pack[1] - pack[0])),
            cluster_bounds_m=cluster.tolist(),
            cluster_area_m2=float(np.prod(cluster[1] - cluster[0])),
        ).items():
            object.__setattr__(self, name, value)

    @property
    def cube(self):
        return np.array(self.cube_xyz)

    @property
    def target(self):
        return np.array(self.target_xyz)

    @property
    def toe(self):
        return np.array(self.ramp_toe_xyz)

    @property
    def cue_pivot(self):
        return np.array(self.cue_pivot_xyz)

    @property
    def ramp_direction(self):
        return unit(self.ramp_heading_deg)

    @property
    def across(self):
        return unit(self.ramp_heading_deg + 90)

    @property
    def ramp_angle(self):
        return np.radians(self.ramp_grade_deg)

    @property
    def ramp_run(self):
        return self.ramp_length_m * np.cos(self.ramp_angle)

    @property
    def ramp_height(self):
        return self.ramp_length_m * np.sin(self.ramp_angle)

    @property
    def cue_contact_radius(self):
        return self.cue_front_m + 0.08 + 0.009 + HALF * np.sqrt(2) + 0.003

    def point(self, t, lateral=0, height=None):
        p = self.toe - self.ramp_direction * self.ramp_run * t + self.across * lateral
        p[2] = self.ramp_height * t if height is None else height
        return p

    def release(self, t=0.78, lateral=0):
        normal = self.ramp_direction * np.sin(self.ramp_angle) + [
            0,
            0,
            np.cos(self.ramp_angle),
        ]
        return self.point(t, lateral) + normal * (self.ball_radius + 0.002)

    def ramp_vertices(self):
        return np.array(
            [
                self.point(t, w, z)
                for t in (0, 1)
                for w in (-self.ramp_width_m / 2, self.ramp_width_m / 2)
                for z in (-0.008, self.ramp_height * t)
            ]
        )

    def pan_site(self):
        if self.pan_face == "raised_end":
            return self.point(1, 0, 0), self.ramp_direction
        sign = 1 if self.pan_face == "left_side" else -1
        return self.point(
            self.pan_fraction, sign * self.ramp_width_m / 2, 0
        ), -sign * self.across


def tool_poly(method, p, yaw, params):
    local = bounds(np.concatenate(tool_part_bounds(method, [0, 0, 0], 0, params)))
    return rectangle(local[0], local[1], p, yaw)


def tool_part_bounds(method, p, yaw, params):
    length = params["tool_length_m"]
    r = params["shaft_radius_m"]
    handle = params["handle_half_size_m"]
    scale = params.get("hook_head_scale", 1.0)
    tip = params.get("ball_radius_m", 0.016)
    pan_half = params.get("pan_length_m", 0.07) / 2
    pan_width_half = params.get("pan_width_m", 0.065) / 2 + 0.002
    shaft_end = length - pan_half if method == "scoop" else length
    head = {
        "hook": (
            (length - 0.05 * scale, -0.006 * scale),
            (length + 0.006 * scale, 0.101 * scale),
        ),
        "poke": ((length - tip, -tip), (length + tip, tip)),
        "scoop": (
            (length - pan_half, -pan_width_half),
            (length + pan_half, pan_width_half),
        ),
    }[method]
    return [
        bounds(rectangle(lo, hi, p, yaw))
        for lo, hi in [
            ((-handle[0], -handle[1]), (handle[0], handle[1])),
            ((0, -r), (shaft_end, r)),
            head,
        ]
    ]


def alignment_audit(layout):
    d = (layout.target - layout.cube)[:2]
    d = d / np.linalg.norm(d)
    v = (layout.cube - layout.cue_pivot)[:2]
    along = float(v @ d)
    lateral = float(np.cross(d, v))
    angle = float(np.degrees(np.arctan2(abs(lateral), along)))
    cue_angle = float(
        abs((layout.cue_heading_deg - layout.target_heading_deg + 180) % 360 - 180)
    )
    tip_direction = unit(layout.cue_heading_deg)[:2]
    # Where the rod ray meets the cube's rear-face contact plane.
    hit_t = (along - HALF - CUE_TIP_RADIUS) / float(tip_direction @ d)
    hit = layout.cue_pivot[:2] + tip_direction * hit_t - layout.cube[:2]
    hit_offset = float(np.cross(d, hit))
    passed = bool(
        along > 0
        and angle <= MAX_CUE_AXIS_ERROR_DEG + 1e-6
        and cue_angle <= MAX_CUE_AXIS_ERROR_DEG + 1e-6
        and abs(hit_offset) <= MAX_CUE_CONTACT_OFFSET_M + 1e-6
    )
    if not passed:
        raise RuntimeError(f"Cue alignment failed: {angle}, {cue_angle}, {hit_offset}")
    return dict(
        passed=passed,
        mount_to_cube_axis_error_deg=angle,
        cue_axis_error_deg=cue_angle,
        mount_lateral_offset_m=lateral,
        rear_face_hit_offset_m=hit_offset,
        mount_behind_cube_m=along,
    )


@dataclass(frozen=True)
class PrecisionLayout(AlignedLayout):
    cue_stroke_m: float = 0.12
    cue_rear_m: float = 0.08
    cue_grip_offset_m: float = 0.04
    cue_yaw_damping: float = 0.05
    desired_release_fraction: float = 0.55
    layout_version: str = "precision-impact-v1"

    def pan_site(self):
        if self.pan_face == "inclined_face":
            return self.toe.copy(), -self.ramp_direction
        return super().pan_site()

    @property
    def cue_contact_radius(self):
        # Covers every rod/tip/handle point at every slide position and yaw,
        # plus the cube's circumscribed footprint and contact tolerance.
        tool_radius = max(
            self.cue_front_m + self.cue_stroke_m + CUE_TIP_RADIUS,
            np.hypot(self.cue_rear_m, .006),
            np.hypot(max(self.cue_grip_offset_m,
                         abs(self.cue_stroke_m - self.cue_grip_offset_m)) + .012, .012),
        )
        return tool_radius + HALF * np.sqrt(2) + .003


def reachable(point, minimum=0.23, maximum=0.81):
    return (
        minimum < np.linalg.norm(np.asarray(point)[:2] - [-0.615, 0])
        and np.linalg.norm(np.asarray(point) - [-0.615, 0, 0.333]) < maximum
    )


def pack_handhelds(rng, params, occupied, methods=METHODS):
    """Place the compact tool group in any visible, reachable unused area."""
    lengths = {m: params[m]["tool_length_m"] for m in ("hook", "poke", "scoop")}
    for _ in range(50):
        if rng.random() < 0.4:
            theta = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(0.75, 1.05)
            center = np.array(
                [
                    -0.13 + 0.31 * radius * np.cos(theta),
                    0.39 * radius * np.sin(theta),
                    0.018,
                ]
            )
        else:
            center = np.r_[rng.uniform([-0.40, -0.43], [0.20, 0.43]), 0.018]
        yaw = float(rng.uniform(-180, 180))
        d = unit(yaw)
        across = unit(yaw + 90)
        pitch = rng.uniform(0.080, 0.102)
        # The return tips must lie beyond the pan's actual leading edge.
        # A fixed offset would make larger sampled heads impossible to pack.
        hook_tip = (
            lengths["scoop"]
            + params["scoop"]["pan_length_m"] / 2
            + 0.05 * params["hook"]["hook_head_scale"]
            + rng.uniform(0.012, 0.030)
        )
        positions = dict(
            scoop=across * pitch,
            hook=d * (hook_tip - lengths["hook"]),
            poke=d * (hook_tip - lengths["poke"]) - across * pitch,
        )
        initial = [
            tool_poly(m, p, yaw, params[m])
            for m, p in positions.items()
            if m in methods
        ]
        box = bounds(np.concatenate(initial))
        shift = center - np.r_[(box[0] + box[1]) / 2, 0]
        all_poses = {m: [(p + shift).tolist(), yaw] for m, p in positions.items()}
        poses = {m: p for m, p in all_poses.items() if m in methods}
        polys = {m: tool_poly(m, *poses[m], params[m]) for m in poses}
        parts = {m: tool_part_bounds(m, *poses[m], params[m]) for m in poses}
        if any(
            not clear(p, occupied, 0.012) or not visible(prism(p, 0, 0.044))
            for p in polys.values()
        ):
            continue
        if any(
            not reachable(np.asarray(p) + [0, 0, 0.08], 0.28, 0.80)
            for p, _ in poses.values()
        ):
            continue
        if any(
            intersects(a, b, 0.006)
            for i, m in enumerate(poses)
            for n in list(poses)[i + 1 :]
            for a in parts[m]
            for b in parts[n]
        ):
            continue
        if any(
            intersects(rectangle((-0.035, -0.057), (0.035, 0.057), *poses[m]), p, 0.004)
            for m in poses
            for n in poses
            if m != n
            for p in parts[n]
        ):
            continue
        pack = bounds(np.concatenate(list(polys.values())))
        # Clearance, camera coverage and grasp reach decide placement; no
        # fixed rim band reserves space for methods that are absent.
        return all_poses, [p for m in poses for p in parts[m]], pitch, pack
    return None


def sample_precision_layout(
    seed,
    sector=0,
    max_attempts=8000,
    distance_band=(0.08, 0.32),
    methods=METHODS,
    ranges=None,
    release_fraction_band=(0.30, 0.72),
    pan_loading_surface=None,
):
    """Stratify task headings around 360 degrees; fit three separate tool areas."""
    rng = np.random.default_rng(seed)
    reject = {}
    ranges = dict(DEFAULT_RANGES if ranges is None else ranges)
    if pan_loading_surface is None:
        pan_loading_surface = sample_pan_loading_surface(seed)
    if pan_loading_surface not in PAN_LOADING_FACES:
        raise ValueError("Unknown pan loading surface")
    # Sample object dimensions once. Spatial retries never shrink tools, ramps
    # or masses to make a crowded placement fit.
    size = {key: float(rng.uniform(*band)) for key, band in ranges.items()}
    # Mass, travel distance, and the release-height quota stay fixed while
    # searching for a spatial arrangement; placement retries cannot ease them.
    distance = float(rng.uniform(*distance_band))
    desired_fraction = float(rng.uniform(*release_fraction_band))
    has_ramp = bool(set(methods) & {"ball", "scoop"})
    if "ball" in methods:
        # Couple the available potential energy to the journey while retaining
        # the full public size ranges. This is a noisy proposal heuristic, not
        # a success test; native free-roll and robot validation remain required.
        transfer = size["ball_mass_kg"] / (size["ball_mass_kg"] + size["cube_mass_kg"])
        proposed_height = (
            distance
            / (3.2 * transfer**2 * desired_fraction)
            * float(rng.uniform(0.75, 1.30))
        )
        minimum_grade, maximum_grade = np.radians(ranges["ramp_grade_deg"])
        minimum_length, maximum_length = ranges["ramp_length_m"]
        minimum_height = minimum_length * np.sin(minimum_grade)
        maximum_height = maximum_length * np.sin(maximum_grade)
        if proposed_height > maximum_height:
            proposed_height = maximum_height * float(rng.uniform(0.85, 1.0))
        height = float(np.clip(proposed_height, minimum_height, maximum_height))
        lower_length = max(minimum_length, height / np.sin(maximum_grade))
        upper_length = min(maximum_length, height / np.sin(minimum_grade))
        length = float(rng.uniform(lower_length, max(lower_length, upper_length)))
        size["ramp_length_m"] = length
        size["ramp_grade_deg"] = float(np.degrees(np.arcsin(height / length)))

    def fail(reason):
        reject[reason] = reject.get(reason, 0) + 1

    for attempt in range(max_attempts):
        heading = float((sector * 45 + rng.uniform(-22.5, 22.5) + 180) % 360 - 180)
        d = unit(heading)
        mid = np.r_[rng.uniform([-0.26, -0.23], [0.09, 0.23]), HALF]
        cube = mid - d * distance / 2
        target = mid + d * distance / 2
        if any(
            not np.all((p[:2] > TASK_BOUNDS[0]) & (p[:2] < TASK_BOUNDS[1]))
            for p in (cube, target)
        ):
            continue
        travel = rectangle((-0.045, -0.060), (distance + 0.055, 0.060), cube, heading)
        task = [
            travel,
            rectangle((-0.035, -0.035), (0.035, 0.035), cube),
            rectangle((-0.035, -0.035), (0.035, 0.035), target),
        ]
        sign = int(rng.choice([-1, 1]))
        cue = None
        if "cue" not in methods:
            front = size["cue_front_m"]
            stroke = 0.04
            cue = (
                cube - d * (front + 0.04),
                heading,
                front,
                heading + 35,
                stroke,
                0.011,
                stroke,
                stroke - 0.015,
                [],
                [],
            )
        for _ in range(0 if cue is not None else 40):
            gap = float(rng.uniform(0.006, 0.025))
            front = size["cue_front_m"]
            # Bound the mount's sideways displacement, rather than imposing
            # the same angular offset on both short and long cues. The latter
            # could place a long cue's pivot 8 cm off the cube–target line.
            alignment_angle = min(
                17.0, float(np.degrees(np.arcsin(min(1.0, 0.035 / (front + gap)))))
            )
            cue_heading = heading + sign * rng.uniform(0, alignment_angle)
            contact_point = (
                cube
                - d * (HALF + CUE_TIP_RADIUS)
                + unit(heading + 90) * rng.uniform(-0.006, 0.006)
            )
            pivot = contact_point - unit(cue_heading) * (front + gap)
            # Limit extension using the full all-angle envelope. A long cue
            # remains possible: its mount moves back instead of erasing the
            # required free slide between the reachable region and the goal.
            maximum_stroke = (np.linalg.norm((target - pivot)[:2])
                              - CUE_TARGET_NEIGHBORHOOD_M - CUE_STATIC_CLEARANCE_M
                              - front - CUE_TIP_RADIUS - HALF * np.sqrt(2) - .003)
            stroke = min(size["cue_stroke_m"], maximum_stroke - .001)
            if stroke < max(.035, ranges["cue_stroke_m"][0]):
                continue
            rear = stroke
            grip_offset = stroke - 0.015
            if not (-0.49 < pivot[0] < 0.18 and abs(pivot[1]) < 0.46):
                continue
            radius = front + stroke + CUE_TIP_RADIUS + HALF * np.sqrt(2) + 0.003
            if np.linalg.norm((target - pivot)[:2]) - CUE_TARGET_NEIGHBORHOOD_M < radius + CUE_STATIC_CLEARANCE_M:
                continue
            if not reachable(
                pivot - unit(cue_heading) * grip_offset + [0, 0, 0.095], 0.25, 0.81
            ):
                continue
            for delta in -sign * rng.permutation([15, 25, 40, 55]):
                park = cue_heading + delta
                rod = rectangle(
                    (-rear - 0.005, -0.012), (front + 0.011, 0.012), pivot, park
                )
                mount = rectangle((-0.028, -0.028), (0.028, 0.028), pivot)
                grip = rectangle(
                    (-grip_offset - 0.012, -0.014),
                    (-grip_offset + 0.012, 0.014),
                    pivot,
                    park,
                )
                feet = [rod, mount, grip]
                if any(not clear(p, task, 0.012) for p in feet):
                    continue
                if not (
                    visible(prism(rod, 0.01, 0.032))
                    and visible(prism(mount, 0, 0.05))
                    and visible(prism(grip, 0.02, 0.14))
                ):
                    continue
                sweep = [
                    rectangle((-rear - 0.013, -0.022), (front + 0.020, 0.022), pivot, a)
                    for a in np.linspace(park, cue_heading, int(abs(delta) / 2) + 2)
                ]
                sweep += [
                    rectangle(
                        (-rear - 0.01, -0.022),
                        (front + stroke + 0.020, 0.022),
                        pivot,
                        cue_heading,
                    )
                ]
                if any(not clear(p, [], 0.006) for p in sweep):
                    continue
                cue = (
                    pivot,
                    cue_heading,
                    front,
                    park,
                    stroke,
                    gap,
                    rear,
                    grip_offset,
                    feet,
                    sweep,
                )
                break
            if cue is not None:
                break
        if cue is None:
            fail("cue")
            continue
        (
            pivot,
            cue_heading,
            front,
            park,
            stroke,
            gap,
            rear,
            grip_offset,
            cue_feet,
            sweep,
        ) = cue
        for _ in range(18):
            ramp_heading = heading + rng.uniform(-60, 60)
            length = size["ramp_length_m"]
            grade = size["ramp_grade_deg"]
            width = size["ramp_width_m"]
            br = size["ball_radius_m"]
            cube_mass = size["cube_mass_kg"]
            ball_mass = size["ball_mass_kg"]
            toe = cube - d * (HALF + br) - unit(ramp_heading) * rng.uniform(0.07, 0.28)
            toe[2] = 0
            lateral = float(rng.uniform(-0.025, 0.025))
            toe += unit(ramp_heading + 90) * lateral
            run = length * np.cos(np.radians(grade))
            height = length * np.sin(np.radians(grade))
            if "ball" not in methods or ("scoop" in methods and pan_loading_surface in ("wall", "side_wall")):
                # For an explicit side-wall scoop the paired ball is present,
                # but the ramp need not aim its downhill direction at the cube.
                ramp_heading = float(rng.uniform(-180, 180))
                if rng.random() < 0.4:
                    rim_angle = rng.uniform(-np.pi, np.pi)
                    radius = rng.uniform(0.65, 1.1)
                    center = np.array(
                        [
                            -0.10 + 0.34 * radius * np.cos(rim_angle),
                            0.38 * radius * np.sin(rim_angle),
                            0.0,
                        ]
                    )
                else:
                    center = np.r_[rng.uniform([-0.36, -0.36], [0.20, 0.36]), 0.0]
                toe = center + unit(ramp_heading) * run / 2
            rp = rectangle((-run, -width / 2), (0, width / 2), toe, ramp_heading)
            vertices = np.array(
                [
                    toe
                    - unit(ramp_heading) * run * t
                    + unit(ramp_heading + 90) * w
                    + [0, 0, z]
                    for t in (0, 1)
                    for w in (-width / 2, width / 2)
                    for z in (-0.008, height * t)
                ]
            )
            if has_ramp and not visible(vertices):
                fail("ramp_view")
                continue
            if has_ramp and not clear(rp, task + cue_feet + sweep, 0.012):
                fail("ramp_overlap")
                continue
            if "ball" in methods and not reachable(
                toe - unit(ramp_heading) * run * 0.85 + [0, 0, height * 0.85 + 0.08],
                0.22,
                0.83,
            ):
                continue
            # Reserve the loading approach before parking loose tools. With a
            # fixed incline quota, parking first can repeatedly block the only
            # valid approach even though the rest of the table has ample room.
            pan_access = []
            pan_choice = None
            if "scoop" in methods:
                pan_length = size["pan_handle_length_m"]
                choices = []
                for face in PAN_LOADING_FACES[pan_loading_surface]:
                    for fraction in ((.60,) if face in ("inclined_face", "raised_end") else (.45, .60, .75, .90)):
                        direction, across = unit(ramp_heading), unit(ramp_heading + 90)
                        if face == "inclined_face":
                            site, normal = toe, -direction
                        elif face == "raised_end":
                            site, normal = toe - direction * run, direction
                        else:
                            if height * fraction < .043:
                                continue
                            side = 1 if face == "left_side" else -1
                            site = toe - direction * run * fraction + across * side * width / 2
                            normal = -side * across
                        if float((site - cube) @ normal) <= 0.02:
                            # Use the wall facing the cube, not a detour around
                            # the ramp to load against its opposite side.
                            continue
                        lateral_axis = np.array([-normal[1], normal[0], 0.0])
                        if face == "inclined_face" and abs(float((cube - site) @ lateral_axis)) + HALF * np.sqrt(2) + .006 > width / 2:
                            continue
                        yaw = np.degrees(np.arctan2(normal[1], normal[0]))
                        access = [rectangle((-.28, -.055), (-.003, .055), site, yaw),
                                  rectangle((-pan_length-.20, -.065), (-pan_length+.05, .065), site, yaw)]
                        if reachable(site - normal * (pan_length + .15) + [0, 0, .15], .25, .80) and all(
                            clear(p, cue_feet, .008) for p in access
                        ):
                            choices.append((face, fraction, access))
                if not choices:
                    fail("pan_access")
                    continue
                face, fraction, pan_access = choices[int(rng.integers(len(choices)))]
                pan_choice = (face, fraction)
            params = copy.deepcopy(ORIGINAL_TOOLS)
            pan_length = size["pan_handle_length_m"]
            params["scoop"].update(
                tool_length_m=pan_length,
                pan_length_m=size["pan_length_m"],
                pan_width_m=size["pan_width_m"],
                pan_rim_height_m=0.016,
                pan_rim_front_m=0.025,
                pan_lip_thickness_m=0.0,
            )
            params["hook"]["tool_length_m"] = size["hook_length_m"]
            params["hook"]["hook_head_scale"] = size["hook_head_scale"]
            params["poke"]["tool_length_m"] = size["poke_length_m"]
            params["poke"]["ball_radius_m"] = size["poke_tip_radius_m"]
            packed = pack_handhelds(
                rng,
                params,
                task + cue_feet + sweep + ([rp] if has_ramp else []) + pan_access,
                methods,
            )
            if packed is None:
                fail("tool_bay")
                continue
            poses, tools, pitch, pack = packed
            occupied = task + cue_feet + sweep + ([rp] if has_ramp else []) + tools + pan_access
            ball = None
            if "ball" not in methods:
                ball = np.array([-0.4, 0.4, br])
                foot = rectangle((-br, -br), (br, br), ball)
            for _ in range(0 if ball is not None else 100):
                point = np.r_[rng.uniform(pack[0] - 0.065, pack[1] + 0.065), br]
                foot = rectangle((-br, -br), (br, br), point)
                grip = rectangle((-0.045, -0.045), (0.045, 0.045), point)
                if (
                    clear(foot, occupied, 0.012)
                    and clear(grip, tools, 0.004)
                    and visible(prism(foot, 0, 2 * br))
                    and reachable(point + [0, 0, 0.08], 0.26, 0.80)
                ):
                    ball = point
                    break
            if ball is None:
                fail("ball_bay")
                continue
            label = (
                ("left" if pivot[1] > 0 else "right") + " side"
                if abs(pivot[1]) > 0.17
                else ("far rim" if pivot[0] > 0.035 else "near rim")
            )
            l = PrecisionLayout(
                seed=seed,
                cube_xyz=tuple(cube),
                target_xyz=tuple(target),
                target_heading_deg=heading,
                cube_mass_kg=cube_mass,
                ramp_toe_xyz=tuple(toe),
                ramp_heading_deg=ramp_heading,
                ramp_length_m=length,
                ramp_width_m=width,
                ramp_grade_deg=grade,
                pan_face="left_side",
                ball_radius=br,
                ball_mass=ball_mass,
                ball_rest_xyz=tuple(ball),
                cue_pivot_xyz=tuple(pivot),
                cue_heading_deg=cue_heading,
                cue_park_deg=park,
                cue_front_m=front,
                rim_label="workspace",
                tool_poses=poses,
                tool_params=params,
                handle_spacing_m=pitch,
                pan_fraction=0.60,
                cue_zone=label,
                cue_sweep_polygons=[p.tolist() for p in sweep],
                reference_pack_area_m2=float(np.prod(pack[1] - pack[0])),
                # A grippy floor retains the cube while lifting off an incline.
                # Use the same material for every pan, including distractors
                # and wall-loading scenes, so it cannot identify the method.
                pan_floor_friction=0.8,
                ramp_slot_m=0.018,
                parking_side="free workspace bay",
                ramp_lateral_offset_m=lateral,
                diversity_profile=sector,
                cue_stroke_m=stroke,
                cue_rear_m=rear,
                cue_grip_offset_m=grip_offset,
                layout_version="precision-impact-velocity-v2",
                desired_release_fraction=desired_fraction,
                sampling_order=("cube_target", "cue", "ramp", "handhelds_ball"),
            )
            blockers = cue_feet + tools + ([foot] if "ball" in methods else [])
            exit_point = l.toe - l.across * l.ramp_lateral_offset_m
            ball_path = rectangle(
                (0, -br - 0.01),
                (np.linalg.norm(l.cube - exit_point), br + 0.01),
                exit_point,
                ramp_heading,
            )
            if "ball" in methods and not clear(ball_path, blockers, 0.005):
                fail("ball_path")
                continue
            if "scoop" not in methods:
                if "cue" in methods:
                    alignment_audit(l)
                return l
            for face in rng.permutation(PAN_LOADING_FACES[pan_loading_surface]):
                for fraction in rng.permutation([0.45, 0.60, 0.75, 0.90]):
                    if pan_choice is not None and (face, fraction) != pan_choice:
                        continue
                    if (
                        face in ("left_side", "right_side")
                        and height * fraction < 0.043
                    ):
                        continue
                    trial = replace(l, pan_face=str(face), pan_fraction=float(fraction))
                    site, normal = trial.pan_site()
                    yaw = float(np.degrees(np.arctan2(normal[1], normal[0])))
                    corridor = rectangle((-0.28, -0.055), (-0.003, 0.055), site, yaw)
                    wrist = rectangle(
                        (-pan_length - 0.20, -0.065),
                        (-pan_length + 0.05, 0.065),
                        site,
                        yaw,
                    )
                    if not clear(corridor, blockers, 0.008):
                        fail("pan_corridor")
                        continue
                    if not clear(wrist, blockers, 0.008):
                        fail("pan_wrist")
                        continue
                    if not reachable(
                        site - normal * (pan_length + 0.15) + [0, 0, 0.15], 0.25, 0.80
                    ):
                        fail("pan_reach")
                        continue
                    staging = site - normal * 0.14
                    corners = rectangle(
                        (-run - 0.10, -width / 2 - 0.10),
                        (0.10, width / 2 + 0.10),
                        toe,
                        ramp_heading,
                    )
                    nodes = [cube, staging] + [np.r_[p, HALF] for p in corners]

                    def available(a, b):
                        delta = b - a
                        delta[2] = 0
                        length = np.linalg.norm(delta)
                        if length < 1e-5:
                            return False
                        direction = delta / length
                        angle = np.degrees(np.arctan2(delta[1], delta[0]))
                        path = rectangle(
                            (-0.035, -0.045), (length + 0.035, 0.045), a, angle
                        )
                        return clear(path, blockers + [rp], 0.008) and all(
                            reachable(
                                p - direction * (pan_length + 0.07) + [0, 0, 0.14],
                                0.24,
                                0.80,
                            )
                            for p in (a, b)
                        )

                    reached = {0: None}
                    queue = [0]
                    for a in queue:
                        for b in range(1, len(nodes)):
                            if b not in reached and available(nodes[a], nodes[b]):
                                reached[b] = a
                                queue.append(b)
                        if 1 in reached:
                            break
                    if 1 not in reached:
                        fail("pan_route")
                        continue
                    path = []
                    i = 1
                    while i is not None:
                        path.append(nodes[i].tolist())
                        i = reached[i]
                    if len(path) != 2:
                        fail("pan_direct_wall_route")
                        continue
                    trial = replace(trial, pan_route_xyz=list(reversed(path)))
                    if "cue" in methods:
                        alignment_audit(trial)

                    return trial
        fail("ramp_or_routes")
    raise RuntimeError(f"No precision layout in sector {sector}: {reject}")


def flush_layout(layout):
    """Keep the mount, tip and travel; remove only the unnecessary rear tail."""
    stroke = layout.cue_stroke_m
    if stroke < 0.035:
        raise ValueError("Stroke is too short for the cue grip")
    l = replace(
        layout,
        cue_rear_m=stroke,
        cue_grip_offset_m=stroke - 0.015,
        layout_version="precision-impact-target-neighborhood-v4",
    )
    sweep = [
        rectangle(
            (-stroke - 0.013, -0.022), (l.cue_front_m + 0.020, 0.022), l.cue_pivot, a
        ).tolist()
        for a in np.linspace(
            l.cue_park_deg,
            l.cue_heading_deg,
            int(abs(l.cue_park_deg - l.cue_heading_deg) / 2) + 2,
        )
    ]
    sweep.append(
        rectangle(
            (-stroke - 0.013, -0.022),
            (l.cue_front_m + stroke + 0.020, 0.022),
            l.cue_pivot,
            l.cue_heading_deg,
        ).tolist()
    )
    l = replace(l, cue_sweep_polygons=sweep)
    geometry_check(l)
    return l


def geometry_check(l):
    end = l.cue_stroke_m - l.cue_rear_m
    grip = l.cue_stroke_m - l.cue_grip_offset_m - 0.012
    margin = float(
        np.linalg.norm((l.target - l.cue_pivot)[:2]) - CUE_TARGET_NEIGHBORHOOD_M - l.cue_contact_radius
    )
    distance = float(np.linalg.norm((l.target - l.cube)[:2]))
    coast_gap = distance - l.cue_stroke_m - TOL
    if end < -BEARING_HALF_LENGTH_M + STOP_TOLERANCE_M:
        raise ValueError("Rear tail protrudes behind the bearing at full stroke")
    if end > BEARING_HALF_LENGTH_M - 0.003:
        raise ValueError("Rear cut leaves the bearing unsupported at full stroke")
    if grip < -BEARING_HALF_LENGTH_M + STOP_TOLERANCE_M:
        raise ValueError("Grip protrudes behind the bearing at full stroke")
    if l.cue_grip_offset_m + 0.012 > l.cue_rear_m:
        raise ValueError("Grip extends beyond the cut rod")
    if margin < CUE_STATIC_CLEARANCE_M:
        raise ValueError("Target neighborhood can be reached quasi-statically or lacks clearance")
    if coast_gap < 0.005:
        raise ValueError("Full forward travel is too long for the cube-target distance")
    return dict(
        passed=True,
        rear_end_at_full_stroke_m=end,
        grip_rear_at_full_stroke_m=grip,
        bearing_rear_face_m=-BEARING_HALF_LENGTH_M,
        static_target_clearance_m=margin,
        target_neighborhood_radius_m=CUE_TARGET_NEIGHBORHOOD_M,
        minimum_slow_cube_target_distance_m=margin + CUE_TARGET_NEIGHBORHOOD_M,
        target_neighborhood_outside_all_angle_static_reach=True,
        rear_length_m=l.cue_rear_m,
        stroke_m=l.cue_stroke_m,
        cube_target_distance_m=distance,
        minimum_coast_to_target_region_m=coast_gap,
        full_travel_less_than_cube_target_distance=True,
        no_tail_behind_bearing_at_full_stroke=True,
        target_outside_all_angle_static_reach=True,
    )


def ramp_gap(l):
    """Plane at the toe to the near cube surface, along the rolling direction."""
    yaw = np.radians(
        l.cube_yaw_deg if l.cube_yaw_deg is not None else l.target_heading_deg
    )
    along = l.ramp_direction
    extent = HALF * (
        abs(along @ np.array([np.cos(yaw), np.sin(yaw), 0]))
        + abs(along @ np.array([-np.sin(yaw), np.cos(yaw), 0]))
    )
    return float((l.cube - l.toe) @ along - extent)


def blockers(l, methods=("hook", "poke", "scoop", "ball", "cue")):
    feet = (
        [
            rectangle((-0.030, -0.030), (0.030, 0.030), l.cue_pivot),
            rectangle(
                (-l.cue_rear_m - 0.013, -0.022),
                (l.cue_front_m + 0.020, 0.022),
                l.cue_pivot,
                l.cue_park_deg,
            ),
        ]
        if "cue" in methods
        else []
    )
    if "ball" in methods:
        feet.append(rectangle((-0.045, -0.045), (0.045, 0.045), l.ball_rest_xyz))
    for method in l.tool_poses:
        if method not in methods:
            continue
        feet.extend(
            tool_part_bounds(method, *l.tool_poses[method], l.tool_params[method])
        )
    return feet


def route_pan(l, occupied, ramp):
    """Reroute within the assigned surface kind; never replace an incline by a wall."""
    pan_length = l.tool_params["scoop"]["tool_length_m"]
    surface = "incline" if l.pan_face == "inclined_face" else "wall"
    for face in dict.fromkeys(
        [l.pan_face, *PAN_LOADING_FACES[surface]]
    ):
        for fraction in dict.fromkeys([l.pan_fraction, 0.45, 0.60, 0.75, 0.90]):
            if face in ("left_side", "right_side") and l.ramp_height * fraction < 0.043:
                continue
            trial = replace(l, pan_face=face, pan_fraction=fraction)
            site, normal = trial.pan_site()
            yaw = np.degrees(np.arctan2(normal[1], normal[0]))
            approach = rectangle((-0.28, -0.060), (-0.003, 0.060), site, yaw)
            wrist = rectangle(
                (-pan_length - 0.20, -0.065), (-pan_length + 0.05, 0.065), site, yaw
            )
            if not clear(approach, occupied, 0.008) or not clear(
                wrist, occupied, 0.008
            ):
                continue
            if not reachable(
                site - normal * (pan_length + 0.15) + [0, 0, 0.15], 0.25, 0.80
            ):
                continue
            corners = rectangle(
                (-l.ramp_run - 0.10, -l.ramp_width_m / 2 - 0.10),
                (0.10, l.ramp_width_m / 2 + 0.10),
                l.toe,
                l.ramp_heading_deg,
            )
            nodes = [l.cube, site - normal * 0.14] + [np.r_[p, HALF] for p in corners]

            def available(a, b):
                delta = b - a
                delta[2] = 0
                length = np.linalg.norm(delta)
                if length < 1e-5:
                    return False
                direction = delta / length
                angle = np.degrees(np.arctan2(delta[1], delta[0]))
                path = rectangle((-0.035, -0.045), (length + 0.035, 0.045), a, angle)
                return clear(path, occupied + [ramp], 0.008) and all(
                    reachable(
                        p - direction * (pan_length + 0.07) + [0, 0, 0.14], 0.24, 0.80
                    )
                    for p in (a, b)
                )

            reached = {0: None}
            queue = [0]
            for a in queue:
                for b in range(1, len(nodes)):
                    if b not in reached and available(nodes[a], nodes[b]):
                        reached[b] = a
                        queue.append(b)
                if 1 in reached:
                    break
            if 1 not in reached:
                continue
            path = []
            i = 1
            while i is not None:
                path.append(nodes[i].tolist())
                i = reached[i]
            return replace(trial, pan_route_xyz=list(reversed(path)))
    raise ValueError("No clear pan route after moving the ramp")


def separate_ramp(
    layout,
    gap_m,
    length=None,
    grade=None,
    methods=("hook", "poke", "scoop", "ball", "cue"),
):
    if not 0.06 <= gap_m <= 0.32:
        raise ValueError("Ramp gap must be 6–32 cm")
    if length is not None and not 0.15 <= length <= 0.30:
        raise ValueError("Ramp length must be 15–30 cm")
    if grade is not None and not 10 <= grade <= 40:
        raise ValueError("Ramp angle must be 10–40 degrees")
    l = flush_layout(layout)
    l = replace(
        l,
        ramp_toe_xyz=tuple(l.toe - l.ramp_direction * (gap_m - ramp_gap(l))),
        ramp_length_m=l.ramp_length_m if length is None else length,
        ramp_grade_deg=l.ramp_grade_deg if grade is None else grade,
        layout_version="precision-impact-separated-ramp-v1",
    )
    if not visible(l.ramp_vertices()):
        raise ValueError("Ramp outside benchmark front camera")
    if not reachable(l.release(0.95, 0) + [0, 0, 0.17], 0.22, 0.83):
        raise ValueError("High ball release outside reach")
    occupied = blockers(l, methods)
    ramp = rectangle(
        (-l.ramp_run, -l.ramp_width_m / 2),
        (0, l.ramp_width_m / 2),
        l.toe,
        l.ramp_heading_deg,
    )
    distance = np.linalg.norm(l.target - l.cube)
    travel = rectangle(
        (-0.055, -0.060), (distance + 0.055, 0.060), l.cube, l.target_heading_deg
    )
    if not clear(
        ramp,
        occupied + [travel] + (l.cue_sweep_polygons if "cue" in methods else []),
        0.008,
    ):
        raise ValueError("Ramp blocks a tool or movement corridor")
    exit_point = l.toe - l.across * l.ramp_lateral_offset_m
    delta = l.cube - exit_point
    path = rectangle(
        (0, -l.ball_radius - 0.01),
        (np.linalg.norm(delta[:2]), l.ball_radius + 0.01),
        exit_point,
        np.degrees(np.arctan2(delta[1], delta[0])),
    )
    if not clear(path, occupied, 0.005):
        raise ValueError("Blocked ball runout")
    if "scoop" in methods:
        l = route_pan(l, occupied, ramp)
    alignment_audit(l)
    return l


def available_set(methods):
    methods = tuple(methods)
    if len(methods) != 3 or len(set(methods)) != 3 or not set(methods) <= set(METHODS):
        raise ValueError("Choose exactly three distinct MoveCube methods")
    return tuple(m for m in METHODS if m in methods)


def sample_methods(seed):
    rng = np.random.default_rng(seed)
    available = COMBINATIONS[int(rng.integers(len(COMBINATIONS)))]
    required = available[int(rng.integers(3))]
    return available, required


def sample_episode(
    seed,
    sector=0,
    gap_range=(0.08, 0.28),
    methods=None,
    ranges=None,
    distance_band=(0.08, 0.32),
    release_fraction_band=(0.30, 0.72),
    pan_loading_surface=None,
):
    """Sample availability before geometry; never constrain absent tools."""
    rng = np.random.default_rng(seed)
    methods = (
        available_set(methods)
        if methods is not None
        else COMBINATIONS[int(rng.integers(len(COMBINATIONS)))]
    )
    if "scoop" in methods and "ball" not in methods:
        raise ValueError("A scoop ramp must be paired with a ball")
    layout = sample_precision_layout(
        seed,
        sector=sector,
        methods=methods,
        ranges=ranges,
        distance_band=distance_band,
        release_fraction_band=release_fraction_band,
        pan_loading_surface=pan_loading_surface,
    )
    if "cue" in methods:
        layout = flush_layout(layout)
    return layout, methods, methods[int(rng.integers(3))]
