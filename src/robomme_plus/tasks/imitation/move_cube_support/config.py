"""Versioned, replayable scene parameters for the five-method evaluation."""

import copy
import math
from dataclasses import asdict

import numpy as np

from .diversity import (
    PHASES,
    RELEASE_BANDS,
    release_is_in_band,
)
from .diversity import (
    sampling_plan as make_sampling_plan,
)
from .layout import (
    COMBINATIONS,
    DEFAULT_RANGES,
    HALF,
    METHODS,
    PAN_LOADING_FACES,
    TASK_BOUNDS,
    PrecisionLayout,
    alignment_audit,
    available_set,
    geometry_check,
    sample_episode,
)

SCHEMA = "robomme-ood-move-cube/3"
LEGACY_SCHEMA = "robomme-ood-move-cube/2"


def phase_method_selection(seed, method=None):
    """Choose two three-method scenes before selecting the demonstrated method.

    At least two methods are shared: a unique common tool would reveal the
    answer without the demonstrated motion. The trios may otherwise differ.
    """
    rng = np.random.default_rng([seed, 610136])
    demonstration = COMBINATIONS[int(rng.integers(len(COMBINATIONS)))]
    candidates = [
        group for group in COMBINATIONS if len(set(group) & set(demonstration)) >= 2
    ]
    execution = candidates[int(rng.integers(len(candidates)))]
    common = [m for m in METHODS if m in demonstration and m in execution]
    if method is None:
        # Pairing makes ball much more frequent among shared tools. These
        # weights keep seed-only required-method frequencies near 20% each
        # across the seven trios, while choosing layouts before the method.
        weights = np.array([{"scoop": 2.7, "ball": .49}.get(m, 1.) for m in common])
        method = str(rng.choice(common, p=weights / weights.sum()))
    if method not in METHODS:
        raise ValueError(f"MoveCube method must be one of {METHODS}")
    if method not in common:
        if method not in demonstration:
            candidates = [group for group in COMBINATIONS if method in group]
            demonstration = candidates[int(rng.integers(len(candidates)))]
        candidates = [
            group
            for group in COMBINATIONS
            if method in group and len(set(group) & set(demonstration)) >= 2
        ]
        execution = candidates[int(rng.integers(len(candidates)))]
    return dict(demonstration=demonstration, execution=execution), method


def method_selection(seed, method=None):
    """Compatibility helper for the public variant selector."""
    phases, method = phase_method_selection(seed, method)
    return phases["demonstration"], method


def phase_available_methods(parameters):
    """Read phase-specific trios; old list records mean the same trio twice."""
    value = parameters.get("available_methods", ())
    if isinstance(value, dict):
        if set(value) != {"demonstration", "execution"}:
            raise ValueError("MoveCube needs available methods for both phases")
        return {phase: available_set(group) for phase, group in value.items()}
    group = available_set(value)
    return {phase: group for phase in ("demonstration", "execution")}


def sample_config(
    seed, method=None, ranges=None, sampling_plan=None, mass_scope="episode"
):
    """Geometry is independent of the required method within the shared choices."""
    available, method = phase_method_selection(seed, method)
    ranges = validate_ranges(ranges)
    plan = (
        copy.deepcopy(sampling_plan)
        if sampling_plan is not None
        else make_sampling_plan(seed, ranges["cube_mass_kg"], mass_scope=mass_scope)
    )
    validate_sampling_plan(plan, ranges)
    if mass_scope not in ("episode", "phase"):
        raise ValueError("mass_scope must be episode or phase")
    if mass_scope == "episode":
        if plan["demonstration"]["cube_mass_kg"] != plan["execution"]["cube_mass_kg"]:
            raise ValueError(
                "Episode-level mass requires the same mass stratum in both phases"
            )
        mass_rng = np.random.default_rng([seed, 610150])
        cube_mass = float(mass_rng.uniform(*plan["demonstration"]["cube_mass_kg"]))
        ball_mass = float(mass_rng.uniform(*ranges["ball_mass_kg"]))
    layouts = {}
    for index, phase in enumerate(("demonstration", "execution")):
        phase_ranges = dict(ranges)
        phase_ranges["cube_mass_kg"] = plan[phase]["cube_mass_kg"]
        if mass_scope == "episode":
            phase_ranges["cube_mass_kg"] = [cube_mass, cube_mass]
            phase_ranges["ball_mass_kg"] = [ball_mass, ball_mass]
        # A fixed set of large dimensions may have no spatial solution,
        # especially for the paired pan/ball/cue trio. Draw fresh dimensions
        # within the same ranges and quotas instead of shrinking the tools.
        for attempt in range(8):
            try:
                layout, _, _ = sample_episode(
                    seed + index * 104729 + attempt * 1000003,
                    sector=(seed + index * 3) % 8,
                    methods=available[phase],
                    ranges=phase_ranges,
                    distance_band=plan[phase]["cube_target_distance_m"],
                    release_fraction_band=plan[phase]["ball_release"]["fraction_range"],
                    pan_loading_surface=plan[phase].get("pan_loading_surface"),
                )
                break
            except RuntimeError as error:
                if not str(error).startswith("No precision layout") or attempt == 7:
                    raise
        layouts[phase] = asdict(layout)
    return dict(
        schema=SCHEMA,
        task="MoveCube",
        seed=seed,
        method=method,
        available_methods={phase: list(group) for phase, group in available.items()},
        layouts=layouts,
        oracle_parameters={},
        parameter_ranges=ranges,
        sampling_plan=plan,
        mass_scope=mass_scope,
    )


def validate_config(parameters, method=None, seed=None):
    """Reject stale geometry, absent methods, nonfinite values and static shortcuts."""
    if not isinstance(parameters, dict) or parameters.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError(
            "MoveCube requires five-method scene_parameters with schema " + SCHEMA
        )
    result = copy.deepcopy(parameters)
    if result.get("task") != "MoveCube" or result.get("method") not in METHODS:
        raise ValueError("Invalid MoveCube scene identity")
    if seed is not None and result.get("seed") != seed:
        raise ValueError("MoveCube scene seed does not match episode seed")
    if method is not None and result["method"] != method:
        raise ValueError("MoveCube scene method does not match episode variant")
    phase_methods = phase_available_methods(result)
    if any(result["method"] not in group for group in phase_methods.values()):
        raise ValueError("Demonstrated method is absent from the scene")
    if len(set.intersection(*(set(group) for group in phase_methods.values()))) < 2:
        raise ValueError(
            "A shared distractor is required to avoid a unique-common-tool shortcut"
        )
    if result["schema"] == SCHEMA and any(
        "scoop" in group and "ball" not in group for group in phase_methods.values()
    ):
        raise ValueError("A scoop ramp must be paired with a ball in each phase")

    def finite(value):
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                finite(v)
        elif isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError("Nonfinite MoveCube scene parameter")

    finite(result)
    if "sampling_plan" in result:
        validate_sampling_plan(
            result["sampling_plan"], validate_ranges(result.get("parameter_ranges"))
        )
    if set(result.get("layouts", {})) != {"demonstration", "execution"}:
        raise ValueError("MoveCube needs demonstration and execution layouts")
    if "mass_scope" in result and result["mass_scope"] not in ("episode", "phase"):
        raise ValueError("mass_scope must be episode or phase")
    if result.get("mass_scope") == "episode":
        a, b = (result["layouts"][phase] for phase in PHASES)
        if any(a.get(key) != b.get(key) for key in ("cube_mass_kg", "ball_mass")):
            raise ValueError("Episode-level masses must match between phases")
    for phase, data in result["layouts"].items():
        available = phase_methods[phase]
        required_handhelds = set(available) & set(METHODS[:3])
        if not required_handhelds <= set(
            data.get("tool_params", {})
        ) or not required_handhelds <= set(data.get("tool_poses", {})):
            raise ValueError(
                "An available handheld method is missing its tool geometry or pose"
            )
        layout = PrecisionLayout(**data)
        distance = np.linalg.norm((layout.target - layout.cube)[:2])
        if "sampling_plan" in result:
            quota = result["sampling_plan"][phase]
            surface = quota.get("pan_loading_surface")
            if "scoop" in available and surface is not None and layout.pan_face not in PAN_LOADING_FACES[surface]:
                raise ValueError(f"{phase} pan loading surface violates sampling quota")
            for value, band in (
                (layout.cube_mass_kg, quota["cube_mass_kg"]),
                (distance, quota["cube_target_distance_m"]),
            ):
                if not band[0] - 1e-8 <= value <= band[1] + 1e-8:
                    raise ValueError(
                        f"{phase} physical parameters violate sampling quota"
                    )
            params = result.get("oracle_parameters", {}).get(phase, {})
            if (
                result["method"] == "ball"
                and "fraction" in params
                and not release_is_in_band(quota["ball_release"], params["fraction"])
            ):
                raise ValueError("Ball reference violates its release-height quota")
        if not 0.08 - 1e-6 <= distance <= 0.32 + 1e-6:
            raise ValueError("Cube-target distance must be 8–32 cm")
        for point in (layout.cube, layout.target):
            if not np.all(
                (point[:2] >= TASK_BOUNDS[0]) & (point[:2] <= TASK_BOUNDS[1])
            ):
                raise ValueError("Cube and target must be inside the middle workspace")
        if abs(layout.cube[2] - HALF) > 1e-6:
            raise ValueError("Cube must start on the table")
        if not (
            0.15 <= layout.ramp_length_m <= 0.30
            and 10 <= layout.ramp_grade_deg <= 40
            and 0.10 <= layout.ramp_width_m <= 0.30
        ):
            raise ValueError("Invalid ramp dimensions")
        if not (
            0.015 <= layout.ball_radius <= 0.035
            and 0.02 <= layout.ball_mass <= 1.0
            and 0.02 <= layout.cube_mass_kg <= 0.2
        ):
            raise ValueError("Invalid ball/cube mass or radius")
        if "cue" in available:
            geometry_check(layout)
            alignment_audit(layout)
        for name, params in layout.tool_params.items():
            if name not in METHODS[:3] or not 0.10 <= params["tool_length_m"] <= 0.40:
                raise ValueError("Invalid handheld tool geometry")
    return result


def distance_band(seed, phase_index):
    # Stratification prevents rejection sampling from collapsing 8–32 cm to
    # whichever short distances happen to fit most often.
    plan = make_sampling_plan(seed, DEFAULT_RANGES["cube_mass_kg"])
    return tuple(plan[PHASES[phase_index]]["cube_target_distance_m"])


def validate_sampling_plan(plan, ranges):
    if not isinstance(plan, dict) or set(plan) != set(PHASES):
        raise ValueError("Sampling quotas are required for both phases")
    for phase in PHASES:
        quota = plan[phase]
        if not isinstance(quota, dict):
            raise ValueError(f"Invalid {phase} sampling quota")
        if "pan_loading_surface" in quota and (
            not isinstance(quota["pan_loading_surface"], str)
            or quota["pan_loading_surface"] not in PAN_LOADING_FACES
        ):
            raise ValueError("Invalid pan loading surface quota")
        for name, allowed in (
            ("cube_mass_kg", ranges["cube_mass_kg"]),
            ("cube_target_distance_m", (0.08, 0.32)),
        ):
            band = quota.get(name)
            if (
                not isinstance(band, (list, tuple))
                or len(band) != 2
                or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in band
                )
                or not allowed[0] <= band[0] <= band[1] <= allowed[1]
            ):
                raise ValueError(f"Invalid {phase} {name} sampling quota")
        release = quota.get("ball_release")
        if (
            not isinstance(release, dict)
            or release.get("regime") not in RELEASE_BANDS
            or list(RELEASE_BANDS[release["regime"]]) != release.get("fraction_range")
        ):
            raise ValueError("Invalid ball release regime")


def validate_ranges(ranges=None):
    result = dict(DEFAULT_RANGES)
    if ranges is not None:
        unknown = set(ranges) - set(result)
        if unknown:
            raise ValueError(f"Unknown MoveCube parameter ranges: {sorted(unknown)}")
        result.update(ranges)
    for name, band in result.items():
        if (
            len(band) != 2
            or not all(math.isfinite(v) for v in band)
            or not 0 < band[0] <= band[1]
        ):
            raise ValueError(f"Invalid parameter range {name}: {band}")
    return {key: list(value) for key, value in result.items()}
