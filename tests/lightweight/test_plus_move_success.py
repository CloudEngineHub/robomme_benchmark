"""Adversarial regressions: reaching the goal with a shortcut must fail."""

import copy
import unittest
from robomme_plus.tasks.imitation.move_cube_support.criteria import (
    verdict,
    METHODS,
    cue_impact_components,
    comparison_measurement,
    MAX_NON_HEAD_CONTACT_SECONDS,
)
import math


def successful(method):
    handheld = {
        m: dict(
            unheld_cube_contact=False,
            illegal_surface_contact=False,
            shaft_contact_s=0.0,
            shaft_contact_run_steps=0,
            shaft_longest_contact_s=0.0,
            non_head_contact_steps=0,
            non_head_contact_run_steps=0,
            non_head_longest_contact_s=0.0,
            wall_contact_steps=4,
            max_supported_rise_m=0.08,
            longest_supported_s=0.5,
            max_supported_carry_m=0.12,
            max_supported_target_progress_m=0.10,
            incorrect_contact_motion_m=0.0,
            near_target_inner_progress_m=0.05,
            near_target_wrong_face_motion_m=0.0,
            correct_contact_steps=80,
            correct_progress_m=0.12,
        )
        for m in METHODS[:3]
    }
    return dict(
        robot_cube_contact=False,
        cube_grasped=False,
        target_moved=False,
        fixture_moved=False,
        discontinuity=False,
        tools_used=[method],
        tool_max_displacement_m={},
        wrong_tool_progress_m=0.0,
        cube_contacts=["scoop_head", "ramp"]
        if method == "scoop"
        else [method + "_head"]
        if method in ("hook", "poke")
        else [method],
        max_cube_rise_m=0.08 if method == "scoop" else 0.001,
        final_error_m=0.003,
        on_table=True,
        stable_target_s=0.5,
        robot_static=True,
        cube_held_now=False,
        current_contacts=[],
        pan_unload=dict(
            tilted_release=True,
            max_horizontal_translation_m=0.001,
            table_push_distance_m=0.0,
            post_release_contact=False,
        ),
        required_flat_runout_m=0.08,
        handheld=handheld,
        ball=dict(
            grasped=True,
            released_on_ramp=True,
            launched=False,
            held_hit=False,
            ramp_after_release_s=0.3,
            robot_recontact_after_release=False,
            descent_after_release_m=0.08,
            free_before_hit_s=0.6,
            rolling_rotation_rad=2.0,
            free_hit=True,
            impact_speed_mps=0.7,
            flat_runout_m=0.18,
        ),
        cue=dict(
            dynamic_hit=True,
            impact_contact_speed_mps=0.6,
            non_tip_contact=False,
            swing_hit=False,
            held_tip_hit=True,
            impact_forward_speed_mps=0.6,
            limits_respected=True,
            static_target_unreachable=True,
            coast_after_last_contact_m=0.09,
        ),
    )


class SuccessContractTest(unittest.TestCase):
    def check_rejected(self, method, s, reason):
        r = verdict(method, METHODS, s)
        self.assertFalse(r["success"])
        self.assertIn(reason, r["violations"] + r["missing_requirements"])

    def test_legitimate_evidence_for_each_method(self):
        for m in METHODS:
            with self.subTest(method=m):
                self.assertTrue(verdict(m, METHODS, successful(m))["success"])

    def test_goal_position_alone_never_substitutes_for_required_method(self):
        for used in METHODS:
            for required in METHODS:
                if used == required:
                    continue
                with self.subTest(used=used, required=required):
                    state = successful(used)
                    state["wrong_tool_progress_m"] = .12
                    self.check_rejected(required, state, "cube_moved_by_another_tool")

    def test_global_shortcuts_fail_even_after_goal_success(self):
        for m in METHODS:
            for key, reason in [
                ("robot_cube_contact", "robot_or_gripper_touched_cube"),
                ("cube_grasped", "robot_grasped_cube"),
                ("target_moved", "target_moved"),
                ("fixture_moved", "fixed_fixture_moved"),
                ("discontinuity", "cube_state_discontinuity"),
            ]:
                with self.subTest(method=m, shortcut=key):
                    s = successful(m)
                    s[key] = True
                    self.check_rejected(m, s, reason)

    def test_using_distractor_before_correct_tool_is_permanent_failure(self):
        s = successful("hook")
        s["tools_used"].append("poke")
        r = verdict("hook", METHODS, s)
        self.assertTrue(r["failed"])
        self.assertIn("another_tool_was_used", r["violations"])
        s["current_contacts"] = []
        self.assertFalse(verdict("hook", METHODS, s)["success"])

    def test_incidental_distractor_contact_is_allowed_but_propulsion_is_not(self):
        s = successful("hook")
        s["cube_contacts"].extend(["cue", "poke_head"])
        s["tools_used"].append("cue")  # legacy cue movement must not be use
        self.assertTrue(verdict("hook", METHODS, s)["success"])
        s["wrong_tool_progress_m"] = 0.01
        self.check_rejected("hook", s, "cube_moved_by_another_tool")

    def test_non_scoop_lifting_is_forbidden(self):
        for m in ("hook", "poke", "ball", "cue"):
            s = successful(m)
            s["max_cube_rise_m"] = 0.04
            self.check_rejected(m, s, "cube_lifted_for_non_scoop_method")

    def test_pan_push_is_not_scoop(self):
        s = successful("scoop")
        s["max_cube_rise_m"] = 0.0
        s["handheld"]["scoop"].update(
            max_supported_rise_m=0.0,
            longest_supported_s=0.0,
            max_supported_carry_m=0.0,
            max_supported_target_progress_m=0.0,
        )
        self.check_rejected("scoop", s, "cube_not_lifted_4cm_while_supported_by_pan")

    def test_pan_lift_and_carry_cannot_license_a_final_push(self):
        for field, value, reason in [
            ("table_push_distance_m", 0.02, "pan_pushed_cube_on_table_after_carry"),
            ("tilted_release", False, "no_airborne_pan_tilt_release"),
        ]:
            with self.subTest(shortcut=field):
                s = successful("scoop")
                s["pan_unload"][field] = value
                self.check_rejected("scoop", s, reason)

    def test_old_pan_evidence_is_not_sufficient_for_new_unloading_rule(self):
        s = successful("scoop")
        del s["pan_unload"]
        self.check_rejected("scoop", s, "no_airborne_pan_tilt_release")

    def test_pan_motion_during_airborne_tilt_release_is_allowed(self):
        s = successful("scoop")
        s["pan_unload"]["max_horizontal_translation_m"] = 0.08
        self.assertTrue(verdict("scoop", METHODS, s)["success"])
        # Moving in the air does not waive the direct-target-contact exclusion.
        s["pan_touched_target"] = True
        self.check_rejected("scoop", s, "pan_touched_target_area")

    def test_pan_shaft_or_handle_contact_prevents_release_acceptance(self):
        for part in ("head", "shaft", "handle"):
            with self.subTest(part=part):
                s = successful("scoop")
                s["current_contacts"] = ["scoop_" + part]
                r = verdict("scoop", METHODS, s)
                self.assertFalse(r["failed"])
                self.assertIn("cube_still_touching_handheld_tool", r["missing_requirements"])

    def test_pan_underside_cannot_cover_target_at_table_height(self):
        s = successful("scoop")
        s["pan_touched_target"] = True
        self.check_rejected("scoop", s, "pan_touched_target_area")

    def test_hook_contact_is_allowed_but_final_approach_must_use_inner_faces(self):
        s = successful("hook")
        s["handheld"]["hook"].update(unheld_cube_contact=True, illegal_surface_contact=True,
                                      underside_contact_steps=10, underside_contact_motion_m=.03)
        self.assertTrue(verdict("hook", METHODS, s)["success"])
        s["handheld"]["hook"]["near_target_wrong_face_motion_m"] = .003
        self.check_rejected("hook", s, "hook_final_approach_used_non_inner_face")
        s["handheld"]["hook"]["near_target_wrong_face_motion_m"] = 0
        s["handheld"]["hook"]["near_target_inner_progress_m"] = 0
        self.check_rejected("hook", s, "no_inner_hook_final_approach")

    def test_pan_touch_is_allowed_independently_of_instantaneous_grasp(self):
        s = successful("scoop")
        s["handheld"]["scoop"].update(unheld_cube_contact=True, illegal_surface_contact=True)
        s["pan_unload"]["post_release_contact"] = True
        self.assertTrue(verdict("scoop", METHODS, s)["success"])
        s["handheld"]["scoop"]["max_supported_rise_m"] = 0
        self.check_rejected("scoop", s, "cube_not_lifted_4cm_while_supported_by_pan")

    def test_pan_brief_hop_is_not_supported_carry(self):
        s = successful("scoop")
        s["handheld"]["scoop"]["longest_supported_s"] = 0.02
        self.check_rejected("scoop", s, "no_sustained_supported_lift")

    def test_pan_lift_in_place_is_not_carry(self):
        s = successful("scoop")
        s["handheld"]["scoop"]["max_supported_carry_m"] = 0.001
        self.check_rejected("scoop", s, "cube_not_carried_6cm_while_lifted")

    def test_pan_sideways_wiggle_after_pushing_to_target_is_not_carry_to_target(self):
        s = successful("scoop")
        s["handheld"]["scoop"]["max_supported_target_progress_m"] = 0.0
        self.check_rejected("scoop", s, "pan_carry_did_not_approach_target_by_6cm")

    def test_hook_cannot_count_a_token_correct_contact_then_push(self):
        s = successful("hook")
        s["handheld"]["hook"].update(
            correct_progress_m=0.006, near_target_wrong_face_motion_m=0.12
        )
        self.check_rejected("hook", s, "hook_final_approach_used_non_inner_face")
        s["handheld"]["hook"]["near_target_wrong_face_motion_m"] = 0.0
        self.check_rejected("hook", s, "less_than_6cm_of_correct_method_progress")

    def test_poke_head_progress_does_not_require_grasp_detector(self):
        s = successful("poke")
        s["handheld"]["poke"].update(unheld_cube_contact=True,
                                     incorrect_contact_motion_m=0.04)
        self.assertTrue(verdict("poke", METHODS, s)["success"])
        s["handheld"]["poke"]["correct_contact_steps"] = 0
        self.check_rejected("poke", s, "insufficient_poke_head_contact")

    def test_large_distractor_movement_fails_without_lifting(self):
        for method in METHODS:
            for distractor in set(METHODS) - {method, "cue"}:
                with self.subTest(method=method, distractor=distractor):
                    s = successful(method)
                    s["tool_max_displacement_m"][distractor] = 0.049
                    self.assertTrue(verdict(method, METHODS, s)["success"])
                    s["tool_max_displacement_m"][distractor] = 0.051
                    self.check_rejected(method, s, "another_tool_was_moved")

    def test_required_tool_and_passive_cue_may_move(self):
        for method in METHODS:
            s = successful(method)
            s["tool_max_displacement_m"] = {method: 0.4, "cue": 0.2}
            self.assertTrue(verdict(method, METHODS, s)["success"])

    def test_repeated_brief_non_head_contacts_are_allowed_after_separation(self):
        for method in METHODS[:3]:
            for duration in (0.004, 0.102, 0.3, 0.5):
                with self.subTest(method=method, duration=duration):
                    s = successful(method)
                    s["cube_contacts"].extend([method + "_shaft", method + "_handle"])
                    s["handheld"][method].update(
                        illegal_surface_contact=True,
                        non_head_contact_steps=round(duration * 3 * 500),
                        non_head_longest_contact_s=duration,
                    )
                    self.assertTrue(verdict(method, METHODS, s)["success"])

    def test_long_non_head_contact_fails_even_after_separation(self):
        for method in METHODS[:3]:
            with self.subTest(method=method):
                s = successful(method)
                s["handheld"][method]["non_head_longest_contact_s"] = MAX_NON_HEAD_CONTACT_SECONDS + 0.002
                self.check_rejected(method, s, "non_head_contact_too_long")

    def test_brush_may_continue_but_every_tool_part_must_release_before_success(self):
        for method in METHODS[:3]:
            for part in ("head", "shaft", "handle"):
                with self.subTest(method=method, part=part):
                    s = successful(method)
                    s["handheld"][method]["non_head_longest_contact_s"] = 0.004
                    s["current_contacts"] = [method + "_" + part]
                    r = verdict(method, METHODS, s)
                    self.assertFalse(r["failed"])
                    self.assertFalse(r["success"])
                    self.assertIn("cube_still_touching_handheld_tool", r["missing_requirements"])

    def test_poke_brush_cannot_replace_round_head_progress(self):
        s = successful("poke")
        s["handheld"]["poke"].update(non_head_longest_contact_s=0.004,
                                      correct_progress_m=0.0027)
        self.check_rejected("poke", s, "less_than_6cm_of_correct_method_progress")

    def test_ball_shortcuts_fail(self):
        cases = [
            ("released_on_ramp", False, "ball_not_released_on_ramp"),
            ("launched", True, "ball_launched_instead_of_released"),
            ("held_hit", True, "held_ball_pushed_cube"),
            ("ramp_after_release_s", 0.0, "no_free_roll_on_ramp_after_release"),
            ("free_before_hit_s", 0.01, "ball_not_free_before_impact"),
            ("rolling_rotation_rad", 0.0, "ball_did_not_roll"),
            ("descent_after_release_m", 0.0, "ball_did_not_descend_ramp"),
            ("flat_runout_m", 0.01, "insufficient_table_runout_before_ball_impact"),
        ]
        for key, value, reason in cases:
            with self.subTest(shortcut=key):
                s = successful("ball")
                s["ball"][key] = value
                self.check_rejected("ball", s, reason)

    def test_ball_recontact_is_allowed_with_legitimate_free_rolling_evidence(self):
        state = successful("ball")
        state["ball"]["robot_recontact_after_release"] = True
        self.assertTrue(verdict("ball", METHODS, state)["success"])
        state["ball"]["free_before_hit_s"] = .01
        result = verdict("ball", METHODS, state)
        self.assertFalse(result["success"])
        self.assertFalse(result["failed"])
        self.assertIn("ball_not_free_before_impact", result["missing_requirements"])

    def test_cue_shortcuts_fail(self):
        cases = [
            ("dynamic_hit", False, "no_dynamic_cue_strike"),
            ("impact_contact_speed_mps", 0.02, "cue_strike_not_dynamic"),
            (
                "static_target_unreachable",
                False,
                "cue_can_reach_target_quasi_statically",
            ),
            (
                "coast_after_last_contact_m",
                0.01,
                "cube_did_not_coast_4cm_after_cue_contact",
            ),
            ("limits_respected", False, "cue_exceeded_joint_limits"),
        ]
        for key, value, reason in cases:
            with self.subTest(shortcut=key):
                s = successful("cue")
                s["cue"][key] = value
                self.check_rejected("cue", s, reason)

    def test_cue_any_face_rotation_and_unheld_touch_are_not_hard_failures(self):
        s = successful("cue")
        s["cue"].update(non_tip_contact=True, swing_hit=True, held_tip_hit=False,
                        impact_forward_speed_mps=-.1)
        self.assertTrue(verdict("cue", METHODS, s)["success"])
        s["final_error_m"] = .22
        r = verdict("cue", METHODS, s)
        self.assertFalse(r["success"])
        self.assertFalse(r["failed"])  # incomplete; may continue until timeout

    def test_transient_goal_crossing_or_held_cube_is_not_success(self):
        for key, value, reason in [
            ("stable_target_s", 0.02, "target_not_stably_reached_for_0.4_seconds"),
            ("cube_held_now", True, "cube_not_released"),
            ("on_table", False, "cube_not_resting_on_table"),
            ("robot_static", False, "robot_not_static"),
        ]:
            s = successful("poke")
            s[key] = value
            self.check_rejected("poke", s, reason)

    def test_absent_required_tool_fails(self):
        r = verdict("cue", ("hook", "poke", "ball"), successful("cue"))
        self.assertFalse(r["success"])
        self.assertIn("required_method_not_available", r["violations"])

    def test_cue_aiming_motion_is_not_confused_with_rotational_propulsion(self):
        self.assertFalse(cue_impact_components(0.6, 4.0, 0.2, 0, 0)[2])
        self.assertFalse(cue_impact_components(0.6, 4.0, 0.2, 0, math.radians(12))[2])
        self.assertTrue(cue_impact_components(0.02, 4.0, 0.2, 0, math.pi / 2)[2])

    def test_failure_labels_cannot_turn_a_miss_into_a_success(self):
        m = dict(
            valid=True,
            along_error_m=0.021,
            lateral_error_m=0.011,
            error_m=0.024,
            classification="invalid_or_miss",
        )
        self.assertEqual(comparison_measurement(m)["classification"], "overshoot")
        m["lateral_error_m"] = 0.04
        self.assertEqual(comparison_measurement(m)["classification"], "invalid_or_miss")
        self.assertFalse(
            verdict("ball", METHODS, {**successful("ball"), "final_error_m": 0.024})[
                "success"
            ]
        )


if __name__ == "__main__":
    unittest.main()
