"""Offline reference search, isolated from the demonstration being recorded.

Every accepted parameter set is checked with a fresh robot rollout. Restored
states and directly placed balls are used only to propose parameters in this
private search environment; they never become evaluation frames or actions.
"""

from __future__ import annotations

import copy
import json
import math
import signal
import threading
from contextlib import contextmanager

import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

from .config import phase_available_methods
from .criteria import TARGET_TOLERANCE_M
from .diversity import release_is_in_band, release_plan, validate_reference_loading_surface
from .layout import HALF, unit, CUE_TARGET_NEIGHBORHOOD_M
from .oracle import ball_release, cue_stroke, hold, joint_step, prepare_flush_cue, solve
from .plan_retiming import reference_path
from .search_budget import (
    LayoutSearchExhausted, ReferenceSearchBudget,
    MAX_LAYOUT_ROBOT_TRIALS, MAX_LAYOUT_SEARCH_SECONDS,
)


PRIVATE_PLANNING_TIMEOUT_S = 10.0
MAX_REFERENCE_WAYPOINT_SECONDS = 60.0
# Private search can spend more time on one phase, provided both phases and
# the four seconds of review padding fit the three-minute video requirement.
MAX_SCOOP_REFERENCE_SECONDS = 110.0
MAX_SCOOP_PAIR_SECONDS = 175.0


@contextmanager
def planning_deadline(seconds=PRIVATE_PLANNING_TIMEOUT_S):
    """Bound private path search, never simulation or trajectory execution."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    # Respect an enclosing deadline instead of replacing someone else's timer.
    if previous_timer[0] > 0:
        yield
        return

    def expired(signum, frame):
        raise TimeoutError("Private reference path planning exceeded its deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class ReferencePlanner(PandaArmMotionPlanningSolver):
    def _legacy_steps(self, steps):
        # Frozen oracle parameters specify these short holds in 100 Hz ticks.
        # Keep their physical duration when planning new 20 Hz trajectories.
        return math.ceil(steps * self.base_env.control_freq / 100)

    def open_gripper(self, t=6, gripper_state=None):
        return super().open_gripper(t=self._legacy_steps(t), gripper_state=gripper_state)

    def close_gripper(self, t=6, gripper_state=None):
        return super().close_gripper(t=self._legacy_steps(t), gripper_state=gripper_state)

    def follow_path(self, result, refine_steps=0):
        # Near-singular retiming can produce minutes of almost motionless
        # joint commands for a single waypoint. Reject that private proposal
        # before simulating it; do not turn it into an enormous eval budget.
        seconds = len(result["position"]) / self.base_env.control_freq
        if seconds > MAX_REFERENCE_WAYPOINT_SECONDS:
            raise RuntimeError(f"Private reference waypoint lasts {seconds:.1f}s; resample the path")
        result = reference_path(self.base_env, result)
        return super().follow_path(result, refine_steps=self._legacy_steps(refine_steps))

    def _bounded_plan(self, name, pose, dry_run=False, refine_steps=0):
        try:
            with planning_deadline():
                result = getattr(super(), name)(pose, dry_run=True, refine_steps=refine_steps)
        except TimeoutError:
            print(f"Private reference {name} timed out at {pose.p.tolist()}", flush=True)
            return -1
        if dry_run or isinstance(result, int):
            return result
        return self.follow_path(result, refine_steps=refine_steps) if len(result["position"]) else None

    def move_to_pose_with_screw(self, pose, dry_run=False, refine_steps=0):
        return self._bounded_plan("move_to_pose_with_screw", pose, dry_run, refine_steps)

    def move_to_pose_with_RRTConnect(self, pose, dry_run=False, refine_steps=0):
        return self._bounded_plan("move_to_pose_with_RRTConnect", pose, dry_run, refine_steps)

    def move_to_pose_with_RRTStar(self, *args, **kwargs):
        return self.move_to_pose_with_RRTConnect(*args, **kwargs)


def planner_for(env):
    return ReferencePlanner(
        env,
        vis=False,
        print_env_info=False,
        base_pose=env.agent.robot.pose,
        **env.reference_planner_options(),
    )


def reset_reference(env):
    env.reset(seed=env.seed)
    if getattr(env, "_reference_phase", "demonstration") == "execution":
        # Preserve the real pair's actor geometry and ordering during private
        # pan calibration. Tiny contact differences from duplicating a phase
        # can change whether the beveled lip enters beneath the cube.
        env._resetting = True
        try:
            env._park_phase("demonstration")
            env._select_phase("execution")
            env._reset_active_scene()
            q = env.agent.robot.get_qpos()[0, :7].cpu().numpy()
            joint_step(env, q, np.zeros(7), 1)
        finally:
            env._resetting = False
    hold(env, 0.6, 1)
    env.clear_evidence()
    env._limit_failed = False
    env.timestep = 0


def snapshot(env):
    return copy.deepcopy(env.get_state_dict()), copy.deepcopy(env.agent.get_state())


def restore(env, state):
    env.set_state_dict(copy.deepcopy(state[0]))
    env.agent.set_state(copy.deepcopy(state[1]))
    env.clear_evidence()
    env._limit_failed = False
    env.timestep = 0


def measurement(env):
    result = env.result()
    delta = np.asarray(result["final_cube"]) - env.layout.target
    return dict(
        error_m=result["final_error_m"],
        along_error_m=float(delta @ unit(env.layout.target_heading_deg)),
        lateral_error_m=float(delta @ unit(env.layout.target_heading_deg + 90)),
        success=result["success"],
        violations=result["criteria"]["violations"],
        cue_impact_speed_mps=result["strict_evidence"]["cue"][
            "impact_contact_speed_mps"
        ],
    )


def ball_candidates(env):
    reset_reference(env)
    initial = snapshot(env)
    layout = env.layout
    plan = release_plan(env.scene_parameters, env._phase)
    top_release = plan["regime"] == "top"
    limit = layout.ramp_width_m / 2 - layout.ball_radius - 0.006
    candidates = []
    probes = []
    cache = {}

    def probe(fraction, lane):
        key = (float(fraction), float(lane))
        if key not in cache:
            restore(env, initial)
            env.ball.set_pose(sapien.Pose(ball_release(layout, *key)))
            env.ball.set_linear_velocity([0, 0, 0])
            env.ball.set_angular_velocity([0, 0, 0])
            hold(env, 3.0, 1)
            cache[key] = measurement(env)
        return cache[key]

    # Reject clearly underpowered scenes before the dense height search. These
    # are physical probes, not an analytical claim of global infeasibility.
    ceiling = plan["fraction_range"][1]
    preflight = [
        probe(
            ceiling,
            float(np.clip(-layout.ramp_lateral_offset_m + shift, -limit, limit)),
        )
        for shift in (0.0, -0.003, 0.003, -0.006, 0.006, -0.01, 0.01)
    ]
    if all(m["along_error_m"] < -0.04 for m in preflight):
        print("Ball height-band probes all undershoot by more than 4 cm", flush=True)
        return []
    for shift in (0.0, -0.003, 0.003, -0.006, 0.006, -0.01, 0.01):
        lane = float(np.clip(-layout.ramp_lateral_offset_m + shift, -limit, limit))
        trials = []
        fractions = sorted(
            set(
                [0.15, 1.0]
                + list(np.linspace(*plan["fraction_range"], 5))
                + list(plan["fraction_range"])
                + [layout.desired_release_fraction]
            )
        )
        for fraction in fractions:
            m = probe(fraction, lane)
            trials.append((fraction, m))
            probes.append((fraction, lane, m))
        best_fraction, best = min(trials, key=lambda trial: trial[1]["error_m"])
        print(
            f"Ball search lane {lane:.4f}: closest {best['error_m']:.4f} m "
            f"at fraction {best_fraction:.3f}; required {plan['fraction_range']}",
            flush=True,
        )
        # Interior cases need both undershoot and overshoot. Top cases instead
        # need a successful top release and an insufficient lower release.
        # Across the batch, always using the top therefore cannot succeed.
        weak = [
            x
            for x in trials
            if x[1]["along_error_m"] < -TARGET_TOLERANCE_M and not x[1]["violations"]
        ]
        strong = [
            x
            for x in trials
            if x[1]["along_error_m"] > TARGET_TOLERANCE_M and not x[1]["violations"]
        ]
        if not weak or (not top_release and not strong):
            continue
        for fraction, m in trials:
            if (
                release_is_in_band(plan, fraction)
                and m["error_m"] < TARGET_TOLERANCE_M
                and not m["violations"]
            ):
                candidates.append(
                    (m["error_m"], dict(fraction=float(fraction), lateral_m=lane))
                )
        pairs = [(a, b) for a in weak for b in strong if a[0] < b[0]]
        if pairs:
            a, b = min(pairs, key=lambda pair: pair[1][0] - pair[0][0])
            for fraction in np.linspace(a[0], b[0], 9)[1:-1]:
                if not release_is_in_band(plan, fraction):
                    continue
                m = probe(fraction, lane)
                if m["error_m"] < TARGET_TOLERANCE_M and not m["violations"]:
                    candidates.append(
                        (m["error_m"], dict(fraction=float(fraction), lateral_m=lane))
                    )
        if candidates and not top_release:
            break
    if not candidates or min(error for error, _ in candidates) > 0.008:
        # A few millimetres of lateral placement can change the impact normal.
        # Refine promising two-dimensional placements instead of rejecting a
        # scene solely because its correct lane falls between the coarse lanes.
        from scipy.optimize import minimize, minimize_scalar

        promising = sorted(
            [
                p
                for p in probes
                if release_is_in_band(plan, p[0]) and not p[2]["violations"]
            ],
            key=lambda p: p[2]["error_m"],
        )

        def objective(values):
            m = probe(float(values[0]), float(values[1]))
            return m["error_m"] + (1.0 if m["violations"] else 0.0)

        for fraction, lane, best in promising[:2]:
            if best["error_m"] > 0.06:
                break
            lateral_bounds = (max(-limit, lane - 0.004), min(limit, lane + 0.004))
            if top_release:
                fit = minimize_scalar(
                    lambda x: objective([1.0, x]),
                    bounds=lateral_bounds,
                    method="bounded",
                    options={"xatol": 0.00015, "maxiter": 16},
                )
                release, lateral = 1.0, float(fit.x)
            else:
                fit = minimize(
                    objective,
                    [fraction, lane],
                    method="Powell",
                    bounds=[plan["fraction_range"], lateral_bounds],
                    options={"xtol": 0.00015, "ftol": 0.0001, "maxfev": 60},
                )
                release, lateral = map(float, fit.x)
            hit = probe(release, lateral)
            weak = probe(0.15, lateral)
            strong = None if top_release else probe(1.0, lateral)
            legal = (
                not hit["violations"]
                and not weak["violations"]
                and (strong is None or not strong["violations"])
            )
            bracketed = weak["along_error_m"] < -TARGET_TOLERANCE_M and (
                strong is None or strong["along_error_m"] > TARGET_TOLERANCE_M
            )
            print(
                f"Ball refined fraction {release:.4f}, lane {lateral:.5f}: {hit['error_m']:.4f} m",
                flush=True,
            )
            if legal and bracketed and hit["error_m"] < TARGET_TOLERANCE_M:
                candidates.append(
                    (hit["error_m"], dict(fraction=release, lateral_m=lateral))
                )
                break
    return [params for _, params in sorted(candidates, key=lambda x: x[0])]


def cue_candidates(env):
    layout = env.layout
    face = layout.cube - unit(layout.target_heading_deg) * (HALF + 0.009)
    center = (
        np.degrees(
            np.arctan2((face - layout.cue_pivot)[1], (face - layout.cue_pivot)[0])
        )
        - layout.cue_heading_deg
    )
    center = (center + 180) % 360 - 180
    candidates = []
    for offset in (
        center,
        center - 1.0,
        center + 1.0,
        center - 2.0,
        center + 2.0,
        center - 4.0,
        center + 4.0,
    ):
        reset_reference(env)
        planner = planner_for(env)
        env._reference_planner = planner
        try:
            full_path = prepare_flush_cue(
                env, float(offset), stabilize=True, retract=True
            )
            hold(env, 0.25, -1)
            initial = snapshot(env)
            restore(env, initial)
            hold(env, 0.15, -1)
            env.clear_evidence()
            cue_stroke(env, full_path, 6.0, settle=1.5)
            slow = measurement(env)
            if (
                slow["success"]
                or slow["error_m"] <= CUE_TARGET_NEIGHBORHOOD_M
                or slow["along_error_m"] >= -TARGET_TOLERANCE_M
                or env.peak_cube_speed >= 0.08
            ):
                print(
                    f"Cue slow full stroke was not a clean undershoot: {slow}",
                    flush=True,
                )
                continue
            # A shorter impulse can avoid dragging the cube sideways after the
            # first impact. Every candidate still has to coast to the goal,
            # and a faster strike at the same amplitude must overshoot it.
            for fraction in (1.0, 0.75, 0.5):
                path = full_path[: round((len(full_path) - 1) * fraction) + 1]
                trials = []

                def probe(duration, scale=1.0):
                    restore(env, initial)
                    hold(env, 0.15, -1)
                    env.clear_evidence()
                    cue_stroke(env, path, duration, settle=1.5, velocity_scale=scale)
                    measured = measurement(env)
                    trials.append((duration, scale, measured))
                    return measured

                durations = (
                    0.04,
                    0.05,
                    0.06,
                    0.07,
                    0.08,
                    0.09,
                    0.10,
                    0.12,
                    0.14,
                    0.18,
                    0.22,
                    0.30,
                    0.40,
                )
                # At 20 Hz several requested durations produce the exact
                # same control sequence. Probe each realizable stroke once
                # so the bounded search can try more distinct aims.
                durations = dict.fromkeys(
                    max(1, round(0.04 * env.control_freq), round(t * env.control_freq))
                    / env.control_freq
                    for t in durations
                )
                for duration in durations:
                    probe(duration)

                def overshoots():
                    return [
                        m
                        for _, _, m in trials
                        if m["along_error_m"] > TARGET_TOLERANCE_M
                        and not m["violations"]
                    ]

                if not (any(t[2]["success"] for t in trials) and overshoots()):
                    nearest = sorted(
                        (t for t in trials if not t[2]["violations"]),
                        key=lambda t: t[2]["error_m"],
                    )[:2]
                    for duration, _, _ in nearest:
                        for scale in (0.60, 0.85, 1.10, 1.35, 1.60):
                            probe(duration, scale)
                closest = min(trials, key=lambda t: t[2]["error_m"])
                print(
                    f"Cue aim {offset:.2f}, stroke {fraction:.2f}: closest {closest[2]}",
                    flush=True,
                )
                for duration, scale, measured in trials:
                    stronger = [
                        m
                        for m in overshoots()
                        if m["cue_impact_speed_mps"]
                        > measured["cue_impact_speed_mps"] + 0.04
                    ]
                    if (
                        measured["success"]
                        and stronger
                        and measured["cue_impact_speed_mps"]
                        > slow["cue_impact_speed_mps"] + 0.04
                    ):
                        candidates.append(
                            (
                                measured["error_m"],
                                dict(
                                    aim_offset_deg=float(offset),
                                    duration_s=duration,
                                    velocity_scale=scale,
                                    stroke_fraction=fraction,
                                    stabilize=True,
                                    retract_before_strike=True,
                                    private_calibration={
                                        "slow_full_stroke": slow,
                                        "correct": measured,
                                        "excessive": min(
                                            stronger, key=lambda m: m["along_error_m"]
                                        ),
                                    },
                                ),
                            )
                        )
                if candidates:
                    break
                legal_trials = [m for _, _, m in trials if not m["violations"]]
                if (
                    fraction == 1.0
                    and legal_trials
                    and max(m["along_error_m"] for m in legal_trials) < -0.06
                ):
                    # The full stroke is substantially underpowered at this
                    # aim. Try another aim before shorter strokes.
                    break
            if candidates:
                break
        except LayoutSearchExhausted:
            raise
        except RuntimeError as error:
            print(f"Cue search aim {offset:.2f} rejected: {error}", flush=True)
        finally:
            env._reference_planner = None
            planner.close()
    return [params for _, params in sorted(candidates, key=lambda x: x[0])]


def find_reference(
    config, phase, *, control_mode="pd_joint_pos_vel", trajectory_output=None,
    max_seconds=None, max_search_trials=MAX_LAYOUT_ROBOT_TRIALS,
    max_search_seconds=MAX_LAYOUT_SEARCH_SECONDS,
):
    """Validate a robot rollout within one layout's private search budget.

    ``max_seconds`` caps simulated pan motion; ``max_search_seconds`` caps
    elapsed calibration time, including probes, resets and all robot trials.
    """
    from ..move_cube import MoveCube

    search_budget = ReferenceSearchBudget(max_search_trials, max_search_seconds)
    clone_config = copy.deepcopy(config)
    if config["method"] != "scoop":
        clone_config["layouts"] = {
            name: copy.deepcopy(config["layouts"][phase])
            for name in ("demonstration", "execution")
        }
        available = phase_available_methods(config)[phase]
        clone_config["available_methods"] = {
            name: list(available) for name in ("demonstration", "execution")
        }
        if "sampling_plan" in config:
            clone_config["sampling_plan"] = {
                name: copy.deepcopy(config["sampling_plan"][phase])
                for name in ("demonstration", "execution")
            }
    # Calibrate this scene from scratch, rather than carrying a different
    # phase's already-frozen action parameters into the private clone.
    clone_config["oracle_parameters"] = {}
    env = MoveCube(
        seed=config["seed"],
        _scene_parameters=clone_config,
        obs_mode="state",
        control_mode=control_mode,
    )
    if config["method"] == "scoop":
        env._reference_phase = phase
    try:
        # Includes ball/cue probes, before the later full-rollout recorder is
        # installed. The budget survives every private reset and refinement.
        search_step = env.step

        def budgeted_step(action):
            search_budget.check_time()
            return search_step(action)

        env.step = budgeted_step
        search_budget.check_time()
        method = config["method"]
        phase_budget = min(MAX_SCOOP_REFERENCE_SECONDS, max_seconds) if max_seconds is not None else MAX_SCOOP_REFERENCE_SECONDS
        if method == "scoop" and phase_budget <= 0:
            raise RuntimeError("No pan reference time remains for this phase")
        if method == "ball":
            candidates = ball_candidates(env)
        elif method == "cue":
            candidates = cue_candidates(env)
        elif method == "scoop":
            # MoveCube initially selects demonstration. Execution can have a
            # different ramp face and height: build strategies from its own
            # immutable layout without adding another physics reset.
            layout = env._layouts[phase]
            candidates = [
                dict(flip=flip, scoop_options={
                    "motion_version": 2, "approach_mode": "straight",
                    "follow_incline": layout.pan_face == "inclined_face",
                    "loading_pitch": 0.14, "stroke_speed": 0.50,
                    "loading_preload_m": 0.001, "lift_before_withdraw_m": 0.04,
                    "carry_height_m": 0.10, "carry_pitch": 0.0, "drop_offset": 0.08,
                    **options,
                })
                for options in (
                    {},
                    {"carry_height_m": max(0.14, layout.ramp_height + 0.045),
                     "carry_pitch": -0.10, "lift_before_withdraw_m": 0.065},
                    {"loading_preload_m": 0.0015, "carry_height_m": 0.12},
                )
                for flip in (0, 1)
            ]
        elif method == "hook":
            candidates = [dict(flip=flip) for flip in (0, 1)]
        else:
            candidates = [
                dict(
                    flip=flip,
                    poke_side=side,
                    contact_speed=0.20,
                    contact_precision=0.003,
                )
                for side in (0, 1, -1)
                for flip in (0, 1)
            ]

        # New handheld references use selective 2x plans. Frozen parameters
        # without this key retain their original speed when reconstructed.
        if method in ("hook", "poke", "scoop"):
            candidates = [dict(candidate, plan_stride=2) for candidate in candidates]

        # A permanent contact violation cannot be repaired later. Abort that
        # private trial immediately rather than completing a known-invalid
        # minute-long trajectory. Incomplete positive evidence is not a failure.
        original_step = env.step
        captured_actions = []

        def checked_step(action):
            if method == "scoop" and len(captured_actions) >= round(
                phase_budget * env.control_freq
            ):
                raise RuntimeError(f"Pan reference exceeded its {phase_budget:.1f} second phase budget")
            result = original_step(action)
            captured_actions.append(np.asarray(action).copy())
            if bool(result[-1].get("fail", False)):
                raise RuntimeError(
                    f"Latched reference violation: {env.result()['criteria']}"
                )
            return result

        env.step = checked_step
        # Try distinct loading/carry geometries before resampling the scene;
        # many near-identical curling retries cannot repair an unsuitable load.
        pending = list(candidates[:10] if method == "scoop" else candidates[:12])
        placement_refinements = 0
        carry_refinement = False
        grip_refinements = 0
        gentle_refinement = False
        invalid_pan_approaches = set()
        failed_pan_prefixes = {}
        while pending:
            search_budget.check_time()
            parameters = pending.pop(0)
            approach = (
                parameters.get("flip", 0),
                parameters.get("scoop_options", {}).get("routing_yaw_deg"),
                parameters.get("scoop_options", {}).get("approach_mode", "routed"),
                parameters.get("scoop_options", {}).get("motion_version", 1),
                parameters.get("scoop_options", {}).get("gentle_contact", False),
            )
            if method == "scoop" and (
                approach in invalid_pan_approaches or failed_pan_prefixes.get(approach, 0) >= 2
            ):
                # Carry height, pitch, insertion and unloading options do not
                # change the grasp or the preceding table route. A permanently
                # failed prefix cannot be repaired by those later options.
                # Allow one repeat for a stochastic planning failure, then
                # change the grasp/route instead of replaying it ten times.
                print(f"Skipping repeated failed pan approach: {approach}", flush=True)
                continue
            search_budget.start_trial()
            print(
                f"Reference trial {search_budget.trials}/{search_budget.max_trials} "
                f"for {method} {phase}; layout search limit {search_budget.max_seconds:g}s",
                flush=True,
            )
            captured_actions.clear()
            reset_reference(env)
            env._reference_pan_landing = None
            env._reference_pan_carry_strategy = None
            env._reference_scoop_stage = "grasp"
            planner = planner_for(env)
            try:
                result = solve(env, planner, parameters)
                search_budget.check_time()
                if result["success"]:
                    validate_reference_loading_surface(config, phase, result)
                    if method == "scoop" and getattr(env, "_reference_pan_carry_strategy", None):
                        parameters.setdefault("scoop_options", {}).update(
                            env._reference_pan_carry_strategy
                        )
                    if trajectory_output is not None:
                        np.savez_compressed(
                            trajectory_output,
                            actions=np.asarray(captured_actions),
                            parameters_json=json.dumps(parameters, sort_keys=True),
                            layout_json=json.dumps(
                                config["layouts"][phase], sort_keys=True
                            ),
                        )
                    return parameters
                print(
                    f"Reference miss {method} {phase}: {parameters}: {result['criteria']}",
                    flush=True,
                )
                if (
                    method == "scoop"
                    and not carry_refinement
                    and not result["criteria"]["violations"]
                    and "pan_carry_did_not_approach_target_by_6cm"
                    in result["criteria"]["missing_requirements"]
                    and result["final_error_m"] <= TARGET_TOLERANCE_M
                ):
                    adjusted = copy.deepcopy(parameters)
                    adjusted.setdefault("scoop_options", {})["carry_over_target"] = True
                    pending.insert(0, adjusted)
                    carry_refinement = True
                    print(f"Refining supported pan approach: {adjusted}", flush=True)
            except LayoutSearchExhausted:
                raise
            except RuntimeError as error:
                print(
                    f"Reference rejected {method} {phase}: {parameters}: {error}",
                    flush=True,
                )
                if (method == "scoop" and not gentle_refinement
                        and not parameters.get("scoop_options", {}).get("gentle_contact", False)
                        and env._reference_scoop_stage == "loading"
                        and any(text in str(error) for text in ("jammed", "centered", "grasp lost"))):
                    # A tight loading contact can need slower insertion while
                    # the subsequent supported carry keeps its efficient path.
                    adjusted = copy.deepcopy(parameters)
                    adjusted.setdefault("scoop_options", {})["gentle_contact"] = True
                    pending.insert(0, adjusted)
                    gentle_refinement = True
                if (
                    method == "scoop"
                    and env._reference_scoop_stage in ("grasp", "routing", "straight approach")
                ):
                    failed_pan_prefixes[approach] = failed_pan_prefixes.get(approach, 0) + 1
                    if env._failed():
                        invalid_pan_approaches.add(approach)
                if (
                    method == "scoop"
                    and grip_refinements < 2
                    and parameters.get("flip", 0) in (0, 1)
                    and any(
                        text in str(error)
                        for text in (
                            "Cartesian",
                            "waypoint planning",
                            "continuous arm configuration",
                        )
                    )
                ):
                    # A quarter-turn grasp changes the arm configuration while
                    # leaving the desired pan motion and scene unchanged.
                    adjusted = copy.deepcopy(parameters)
                    adjusted["flip"] = parameters.get("flip", 0) + 0.5
                    pending.insert(0, adjusted)
                    grip_refinements += 1
                    print(
                        f"Retrying pan with a quarter-turn grip: {adjusted}", flush=True
                    )
            finally:
                planner.close()
            if method == "scoop" and getattr(env, "_reference_pan_carry_strategy", None):
                # A landing-offset correction is valid for this unload
                # direction; a new IK-margin winner can point the pan 180
                # degrees away and invalidate that physical calibration.
                parameters.setdefault("scoop_options", {}).update(
                    env._reference_pan_carry_strategy
                )
            landing = env._reference_pan_landing
            if (
                method == "scoop"
                and placement_refinements < 3
                and landing is not None
                and landing["legal"]
                and 0.008 < abs(landing["along_error_m"]) < 0.10
            ):
                # Re-run the complete robot motion with a corrected landing
                # offset; the observed miss never becomes accepted evidence.
                adjusted = copy.deepcopy(parameters)
                options = adjusted.setdefault("scoop_options", {})
                options["drop_offset"] = float(
                    np.clip(
                        options.get("drop_offset", 0.055) + landing["along_error_m"],
                        -0.02,
                        0.14,
                    )
                )
                pending.insert(0, adjusted)
                placement_refinements += 1
                print(f"Refining pan landing: {adjusted}", flush=True)
        raise RuntimeError(
            f"No validated {method} reference for seed {config['seed']}, phase {phase}"
        )
    finally:
        env.close()
