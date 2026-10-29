"""Explicit method-specific MoveCube acceptance, independent of rendering.

Native handheld progress requirements are retained. Pan bowl contact is allowed
on every face; airborne tilt unloading and the target-area exclusion are required.
"""

VERSION = "three-method-behavior-v11"
import math
from .pan_unload import MAX_TABLE_PUSH_M
from .behavior import CONTACT_MOTION_TOLERANCE_M, MAX_DISTRACTOR_DISPLACEMENT_M

METHODS = ("hook", "poke", "scoop", "ball", "cue")
TARGET_TOLERANCE_M = 0.05 / 3
MIN_SUPPORTED_LIFT_M = 0.04
MIN_SUPPORTED_LIFT_SECONDS = 0.15
MIN_CARRY_M = 0.06
MIN_HELD_TOOL_PROGRESS_M = 0.06
MIN_CUE_COAST_M = 0.04
# Per continuous contact: separation starts a fresh allowance. A single contact
# over the limit stays a violation. Shaft/handle brushes do not earn progress.
MAX_NON_HEAD_CONTACT_SECONDS = 0.50


def comparison_measurement(measurement):
    """Label directional misses beyond the actual target acceptance region.

    This only names failed controls; it never changes the success threshold.
    Sideways error must itself stay within tolerance to call it an overshoot.
    """
    m = dict(measurement)
    if (
        m["valid"]
        and m["error_m"] > TARGET_TOLERANCE_M
        and abs(m["lateral_error_m"]) <= TARGET_TOLERANCE_M
    ):
        if m["along_error_m"] > TARGET_TOLERANCE_M:
            m["classification"] = "overshoot"
        elif m["along_error_m"] < -TARGET_TOLERANCE_M:
            m["classification"] = "undershoot"
    return m


def cue_impact_components(slide_speed, yaw_speed, tip_radius, axis_yaw, target_yaw):
    """Project incoming axial and rotational tip motion toward the target.

    Rotation is permitted for aiming and side strikes. It must not supply more
    target-directed impact speed than the forward stroke itself.
    """
    angle = target_yaw - axis_yaw
    axial = max(0.0, slide_speed * math.cos(angle))
    rotational = abs(yaw_speed * tip_radius * math.sin(angle))
    return axial, rotational, rotational > axial + 0.02


def verdict(method, available, state):
    """Return latched violations separately from incomplete positive evidence."""
    bad = []
    missing = []

    def reject(test, reason):
        if test:
            bad.append(reason)

    def need(test, reason):
        if not test:
            missing.append(reason)

    if method not in available:
        return dict(
            success=False,
            failed=True,
            violations=["required_method_not_available"],
            missing_requirements=[],
            version=VERSION,
        )
    reject(state["robot_cube_contact"], "robot_or_gripper_touched_cube")
    reject(state["cube_grasped"], "robot_grasped_cube")
    reject(state["target_moved"], "target_moved")
    reject(state["fixture_moved"], "fixed_fixture_moved")
    reject(state["discontinuity"], "cube_state_discontinuity")
    reject(bool(set(state["tools_used"]) - {method, "cue"}), "another_tool_was_used")
    reject(any(distance > MAX_DISTRACTOR_DISPLACEMENT_M
               for tool, distance in state.get("tool_max_displacement_m", {}).items()
               if tool not in {method, "cue"}), "another_tool_was_moved")
    reject(state.get("wrong_tool_progress_m", 0.0) > CONTACT_MOTION_TOLERANCE_M,
           "cube_moved_by_another_tool")
    reject(not state["cue"]["limits_respected"], "cue_exceeded_joint_limits")
    reject(
        method != "scoop" and state["max_cube_rise_m"] > 0.025,
        "cube_lifted_for_non_scoop_method",
    )
    need(state["final_error_m"] <= TARGET_TOLERANCE_M, "cube_outside_target")
    need(state["on_table"], "cube_not_resting_on_table")
    need(state["stable_target_s"] >= 0.4, "target_not_stably_reached_for_0.4_seconds")
    need(state["robot_static"], "robot_not_static")
    need(not state["cube_held_now"], "cube_not_released")
    if method in ("hook", "poke", "scoop"):
        e = state["handheld"][method]
        reject(e["non_head_longest_contact_s"] > MAX_NON_HEAD_CONTACT_SECONDS,
               "non_head_contact_too_long")
        need(
            not set(state["current_contacts"]) &
            {method + suffix for suffix in ("_head", "_shaft", "_handle")},
            "cube_still_touching_handheld_tool",
        )
        if method == "scoop":
            reject(state.get("pan_touched_target", False), "pan_touched_target_area")
            unload = state.get("pan_unload", {})
            reject(
                unload.get("table_push_distance_m", 0.0) > MAX_TABLE_PUSH_M,
                "pan_pushed_cube_on_table_after_carry",
            )
            need(unload.get("tilted_release", False), "no_airborne_pan_tilt_release")
            need(e["wall_contact_steps"] >= 2, "no_scoop_loading_against_ramp")
            need(
                e["max_supported_rise_m"] >= MIN_SUPPORTED_LIFT_M,
                "cube_not_lifted_4cm_while_supported_by_pan",
            )
            need(
                e["longest_supported_s"] >= MIN_SUPPORTED_LIFT_SECONDS,
                "no_sustained_supported_lift",
            )
            need(
                e["max_supported_carry_m"] > MIN_CARRY_M,
                "cube_not_carried_6cm_while_lifted",
            )
            need(
                e["max_supported_target_progress_m"] > MIN_CARRY_M,
                "pan_carry_did_not_approach_target_by_6cm",
            )
        else:
            if method == "hook":
                reject(e.get("near_target_wrong_face_motion_m", 0.0) > CONTACT_MOTION_TOLERANCE_M,
                       "hook_final_approach_used_non_inner_face")
                need(e.get("near_target_inner_progress_m", 0.0) > 0.01,
                     "no_inner_hook_final_approach")
            need(
                e["correct_contact_steps"] >= 3,
                "insufficient_poke_head_contact" if method == "poke"
                else "insufficient_held_correct_head_contact",
            )
            need(
                e["correct_progress_m"] > MIN_HELD_TOOL_PROGRESS_M,
                "less_than_6cm_of_correct_method_progress",
            )
    elif method == "ball":
        e = state["ball"]
        need(e["grasped"], "ball_not_picked_up_by_robot")
        need(e["released_on_ramp"], "ball_not_released_on_ramp")
        reject(e["launched"], "ball_launched_instead_of_released")
        reject(e["held_hit"], "held_ball_pushed_cube")
        need(e["ramp_after_release_s"] >= 0.04, "no_free_roll_on_ramp_after_release")
        need(e["descent_after_release_m"] > 0.005, "ball_did_not_descend_ramp")
        need(e["free_before_hit_s"] >= 0.05, "ball_not_free_before_impact")
        need(e["rolling_rotation_rad"] > 0.2, "ball_did_not_roll")
        need(e["free_hit"], "no_free_ball_cube_impact")
        need(e["impact_speed_mps"] > 0.08, "ball_impact_too_slow")
        need(
            e["flat_runout_m"] >= state["required_flat_runout_m"],
            "insufficient_table_runout_before_ball_impact",
        )
    else:
        e = state["cue"]
        # Any cue face and either axial or rotational striking are valid.
        # Incidental slow contacts do not erase an earlier dynamic strike.
        need(e.get("dynamic_hit", False), "no_dynamic_cue_strike")
        need(e.get("impact_contact_speed_mps", 0.0) > 0.08, "cue_strike_not_dynamic")
        need(e["limits_respected"], "cue_exceeded_joint_limits")
        need(e["static_target_unreachable"], "cue_can_reach_target_quasi_statically")
        need(
            e["coast_after_last_contact_m"] > MIN_CUE_COAST_M,
            "cube_did_not_coast_4cm_after_cue_contact",
        )
        need("cue" not in state["current_contacts"], "cue_still_touching_cube")
    return dict(
        success=not bad and not missing,
        failed=bool(bad),
        violations=bad,
        missing_requirements=missing,
        version=VERSION,
    )
