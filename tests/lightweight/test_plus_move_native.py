"""Native initialization/action smoke tests; require the ManiSkill environment."""

import unittest
from unittest.mock import patch
import numpy as np
import sapien
from robomme_plus.tasks.imitation.move_cube import MoveCube
from robomme_plus.tasks.imitation.move_cube_support.scene import REST_QPOS, Scene, pos
from robomme_plus.tasks.imitation.move_cube_support.criteria import verdict
from robomme_plus.tasks.imitation.move_cube_support.evidence import StrictScene


class NativeMoveCube(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MoveCube(seed=0, obs_mode="state")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=0)

    def test_new_seed_resamples_both_layouts_and_same_seed_replays(self):
        before = self.env.scene_parameters["layouts"]
        self.env.reset(seed=2)
        after = self.env.scene_parameters["layouts"]
        self.assertNotEqual(before, after)
        self.env.reset(seed=2)
        self.assertEqual(after, self.env.scene_parameters["layouts"])

    def test_benchmark_home_and_native_cameras_survive_settling(self):
        np.testing.assert_allclose(
            self.env.agent.robot.get_qpos()[0].cpu(), REST_QPOS, atol=1e-6
        )
        for name in ("base_camera", "hand_camera"):
            camera = self.env._sensors[name].config
            self.assertEqual((camera.width, camera.height), (256, 256))
        self.assertEqual(self.env.sim_freq, 500)
        self.assertEqual(self.env.control_freq, 20)

    def test_only_three_available_methods_and_no_label_in_observation(self):
        e = self.env
        present = set(e.handheld)
        if e.ball is not None:
            present.add("ball")
        if e.cue is not None:
            present.add("cue")
        self.assertEqual(e.ramp is not None, e.ball is not None)
        self.assertEqual(present, set(e.available_methods))
        before = e._get_obs_extra({})
        required = e._required_method
        try:
            for method in e.available_methods:
                e._required_method = method
                for key, value in e._get_obs_extra({}).items():
                    np.testing.assert_array_equal(value.cpu(), before[key].cpu())
        finally:
            e._required_method = required

    def test_position_actions_and_velocity_commands_are_preserved(self):
        e = self.env
        q = np.array(REST_QPOS[:7])
        eight = np.r_[q, 1.0]
        action = e.normalize_policy_action(eight)
        np.testing.assert_array_equal(action[7:14], np.zeros(7))
        np.testing.assert_allclose(action[:7], q)
        fifteen = np.r_[q, np.full(7, 0.3), -1.0]
        np.testing.assert_allclose(e.normalize_policy_action(fifteen), fifteen)
        with self.assertRaises(ValueError):
            e.normalize_policy_action(np.r_[q, np.nan])
        with self.assertRaises(ValueError):
            e.normalize_policy_action(np.zeros(9))
        for _ in range(20):
            e.step(eight)
        self.assertFalse(e._failed())

    def test_both_phase_groups_spawn_their_own_trio(self):
        e = self.env
        self.assertNotEqual(*e._available_by_phase.values())
        for phase in ("demonstration", "execution"):
            e._park_phase("execution" if phase == "demonstration" else "demonstration")
            e._select_phase(phase)
            e._reset_active_scene()
            present = set(e.handheld)
            if e.ball is not None:
                present.add("ball")
            if e.cue is not None:
                present.add("cue")
            self.assertEqual(e.ramp is not None, e.ball is not None)
            self.assertEqual(present, set(e._available_by_phase[phase]))
            self.assertEqual(len(present), 3)
            self.assertIn(e.variant, present)

    def test_failure_latches_and_successful_demo_is_required_for_transition(self):
        e = self.env
        e.strict["robot_cube_contact"] = True
        for _ in range(3):
            result = e.step(np.r_[REST_QPOS[:7], 1.0])
            self.assertTrue(result[-1]["fail"].item())
            self.assertFalse(result[-1]["success"].item())
        with self.assertRaisesRegex(RuntimeError, "unsuccessful demonstration"):
            e._switch_to_execution(e, None)
        e.reset(seed=0)
        self.assertFalse(e._failed())


class HandheldContactSamples:
    def sample(self, part=None, movement=0.0):
        # Replace only the physics contact collector; retain the full strict
        # evidence update, required-method filter, and task verdict.
        e = self.env
        method = e.variant
        parts = set() if part is None else {part} if isinstance(part, str) else set(part)
        def contact_snapshot():
            e.time += 1 / e.sim_freq
            e.current_contacts = {method + "_" + p for p in parts}
            for p in parts & {"shaft", "handle"}:
                e.evidence[method][p + "_contact_steps"] += 1
            if movement:
                p = pos(e.cube)
                p[0] += movement
                e.cube.set_pose(sapien.Pose(p))
        with patch.object(Scene, "_after_simulation_step", side_effect=contact_snapshot):
            e._after_simulation_step()


class NativePokeContactEvidence(HandheldContactSamples, unittest.TestCase):
    """Feed contact snapshots into the real scene's evidence/termination path."""

    @classmethod
    def setUpClass(cls):
        cls.env = MoveCube(seed=1, variant="poke", obs_mode="state")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=1)

    def test_separate_contacts_each_receive_the_full_allowance(self):
        e = self.env
        for contact in range(3):
            for _ in range(250):
                self.sample("shaft")
            self.assertEqual(e.strict["handheld"]["poke"]["shaft_contact_run_steps"], 250)
            self.assertFalse(e._failed())
            self.sample()  # Separation gives the next contact a fresh timer.
            evidence = e.strict["handheld"]["poke"]
            self.assertEqual(evidence["shaft_contact_run_steps"], 0)
            self.assertEqual(evidence["shaft_contact_s"], (contact + 1) * .5)
            self.assertEqual(evidence["shaft_longest_contact_s"], .5)
            self.assertFalse(e._failed())
        self.assertFalse(any(x["kind"] == "non_head_contact_too_long"
                             for x in e.strict_events))

    def test_long_contact_failure_survives_separation_and_resets_with_evidence(self):
        e = self.env
        for _ in range(250):
            self.sample("shaft")
        self.assertEqual(e.strict["handheld"]["poke"]["shaft_contact_s"], 0.50)
        self.assertEqual(e.strict["handheld"]["poke"]["shaft_longest_contact_s"], 0.50)
        self.assertFalse(e._failed())
        self.sample("shaft")
        self.assertTrue(e._failed())
        self.sample()
        self.assertEqual(e.strict["handheld"]["poke"]["shaft_contact_run_steps"], 0)
        self.assertEqual(e.strict["handheld"]["poke"]["shaft_longest_contact_s"], .502)
        self.assertTrue(e._failed())
        self.assertIn("non_head_contact_too_long",
                      verdict("poke", e.available_methods, e.strict)["violations"])
        self.assertEqual(sum(x["kind"] == "non_head_contact_too_long"
                             for x in e.strict_events), 1)
        e.clear_evidence()
        self.assertEqual(e.strict["handheld"]["poke"]["shaft_contact_s"], 0)
        self.assertEqual(e.strict["handheld"]["poke"]["shaft_contact_run_steps"], 0)
        self.assertEqual(e.strict["handheld"]["poke"]["shaft_longest_contact_s"], 0)
        self.assertFalse(e._failed())

    def test_brief_required_shaft_contact_is_not_wrong_tool_propulsion(self):
        e = self.env
        self.sample("shaft", movement=0.003)
        self.assertFalse(e._failed())
        self.assertEqual(e.strict["wrong_tool_progress_m"], 0)
        self.assertEqual(e.strict["handheld"]["poke"]["correct_progress_m"], 0)
        # The exception belongs only to the demonstrated poke method.
        required = e._required_method
        try:
            e._required_method = next(m for m in e.available_methods if m != "poke")
            self.sample("shaft", movement=0.003)
            self.assertIn("cube_moved_by_another_tool",
                          verdict(e._required_method, e.available_methods, e.strict)["violations"])
        finally:
            e._required_method = required

    def test_handle_contact_receives_the_same_allowance_as_the_shaft(self):
        e = self.env
        self.sample("handle", movement=.003)
        for _ in range(249):
            self.sample("handle")
        self.assertFalse(e._failed())
        self.assertEqual(e.strict["wrong_tool_progress_m"], 0)
        self.assertEqual(e.strict["handheld"]["poke"]["non_head_longest_contact_s"], .5)
        self.sample("handle")
        self.assertIn("non_head_contact_too_long",
                      verdict("poke", e.available_methods, e.strict)["violations"])

    def test_mixed_head_and_handle_brush_cannot_earn_poke_progress(self):
        e = self.env
        self.sample({"head", "handle"}, movement=.003)
        self.assertFalse(e._failed())
        self.assertEqual(e.strict["handheld"]["poke"]["correct_contact_steps"], 0)
        self.assertEqual(e.strict["handheld"]["poke"]["correct_progress_m"], 0)

    def test_head_progress_counts_without_a_grasp_detection(self):
        e = self.env
        heading = np.deg2rad(e.layout.target_heading_deg)
        p = pos(e.cube)
        p[:2] += 0.003 * np.array([np.cos(heading), np.sin(heading)])
        e.cube.set_pose(sapien.Pose(p))
        self.sample("head")
        evidence = e.strict["handheld"]["poke"]
        self.assertTrue(evidence["unheld_cube_contact"])
        self.assertEqual(evidence["correct_contact_steps"], 1)
        self.assertAlmostEqual(evidence["correct_progress_m"], 0.003, places=6)
        self.assertEqual(evidence["incorrect_contact_motion_m"], 0)
        self.assertFalse(e._failed())
        # Even one gripper/cube contact remains a hard failure.
        e.events["robot_cube"] = True
        self.sample("head")
        e.events["robot_cube"] = False
        self.sample()
        self.assertIn("robot_or_gripper_touched_cube",
                      verdict("poke", e.available_methods, e.strict)["violations"])

    def test_large_distractor_excursion_latches_even_after_returning(self):
        e = self.env
        candidates = {m: parts["head"] for m, parts in e.handheld.items() if m != "poke"}
        if e.ball is not None:
            candidates["ball"] = e.ball
        self.assertTrue(candidates)
        for method, actor in candidates.items():
            with self.subTest(method=method):
                e.clear_evidence()
                origin = pos(actor)
                quat = actor.pose.q[0].cpu().numpy()
                actor.set_pose(sapien.Pose(origin + [0.02, 0, 0], quat))
                self.sample()
                self.assertFalse(e._failed())
                actor.set_pose(sapien.Pose(origin + [0.06, 0, 0], quat))
                self.sample()
                self.assertNotIn(method, e.strict["tools_used"])
                self.assertIn("another_tool_was_moved",
                              verdict("poke", e.available_methods, e.strict)["violations"])
                actor.set_pose(sapien.Pose(origin, quat))
                self.sample()
                self.assertAlmostEqual(e.strict["tool_max_displacement_m"][method], .06, places=6)
                self.assertTrue(e._failed())
                e.clear_evidence()
                self.assertFalse(e._failed())
                self.assertEqual(e.strict["tool_max_displacement_m"], {})


class NativeOtherHandheldContactEvidence(HandheldContactSamples, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {
            method: MoveCube(seed=1, variant=method, obs_mode="state")
            for method in ("hook", "scoop")
        }

    @classmethod
    def tearDownClass(cls):
        for env in cls.envs.values():
            env.close()

    def setUp(self):
        for env in self.envs.values():
            env.reset(seed=1)

    def test_repeated_shaft_and_handle_contacts_then_one_overlong_contact(self):
        for method, self.env in self.envs.items():
            e = self.env
            for part in ("shaft", "handle"):
                with self.subTest(method=method, part=part):
                    e.clear_evidence()
                    for _ in range(3):
                        for _ in range(250):
                            self.sample(part)
                        self.assertFalse(e._failed())
                        self.sample()
                    evidence = e.strict["handheld"][method]
                    self.assertEqual(evidence["non_head_contact_steps"], 750)
                    self.assertEqual(evidence["non_head_contact_run_steps"], 0)
                    self.assertEqual(evidence["non_head_longest_contact_s"], .5)
                    for _ in range(251):
                        self.sample(part)
                    self.assertTrue(e._failed())
                    self.sample()
                    self.assertEqual(evidence["non_head_contact_run_steps"], 0)
                    self.assertEqual(evidence["non_head_longest_contact_s"], .502)
                    self.assertTrue(e._failed())
                    self.assertEqual(e._failure_code, "non_head_contact_too_long")
                    e.clear_evidence()
                    self.assertFalse(e._failed())
                    evidence = e.strict["handheld"][method]
                    self.assertEqual(evidence["non_head_contact_steps"], 0)
                    self.assertEqual(evidence["non_head_longest_contact_s"], 0)

    def test_switching_or_overlapping_parts_does_not_restart_or_double_timer(self):
        for method, self.env in self.envs.items():
            with self.subTest(method=method):
                e = self.env
                for _ in range(100):
                    self.sample("shaft")
                for _ in range(100):
                    self.sample({"shaft", "handle"})
                for _ in range(50):
                    self.sample("handle")
                self.assertFalse(e._failed())
                self.assertEqual(e.strict["handheld"][method]["non_head_longest_contact_s"], .5)
                self.sample("handle")
                self.assertTrue(e._failed())

    def test_brush_near_target_is_not_wrong_tool_or_method_progress(self):
        for method, self.env in self.envs.items():
            e = self.env
            for part in ("shaft", "handle"):
                with self.subTest(method=method, part=part):
                    cube = pos(e.cube)
                    cube[:2] = e.layout.target[:2] + [.04, 0]
                    e.cube.set_pose(sapien.Pose(cube))
                    e.clear_evidence()
                    self.sample(part, movement=.003)
                    self.assertFalse(e._failed())
                    out = e.strict["handheld"][method]
                    self.assertEqual(out["near_target_wrong_face_motion_m"], 0)
                    self.assertEqual(out["correct_progress_m"], 0)
                    self.assertEqual(out["max_supported_carry_m"], 0)
                    self.assertEqual(e.strict["wrong_tool_progress_m"], 0)
                    required = e._required_method
                    try:
                        e._required_method = next(m for m in e.available_methods if m != method)
                        self.sample(part, movement=.003)
                        self.assertIn("cube_moved_by_another_tool",
                                      verdict(e._required_method, e.available_methods, e.strict)["violations"])
                    finally:
                        e._required_method = required

    def test_pan_non_head_contact_uses_brush_rule_but_still_blocks_release(self):
        self.env = e = self.envs["scoop"]
        e._strict_pan_last = e.time
        with patch.object(e._pan_unload, "update", wraps=e._pan_unload.update) as update:
            self.sample("handle")
            self.assertTrue(update.call_args.kwargs["contact"])
            self.assertFalse(update.call_args.kwargs["head_contact"])
            self.sample({"head", "handle"})
            self.assertTrue(update.call_args.kwargs["head_contact"])
        self.assertEqual(e._strict_pan_last, -1)
        self.assertFalse(e.strict["pan_unload"]["tilted_release"])

    def test_brush_does_not_mask_wrong_hook_head_face_during_final_approach(self):
        self.env = e = self.envs["hook"]
        cube = pos(e.cube)
        cube[:2] = e.layout.target[:2] + [.04, 0]
        e.cube.set_pose(sapien.Pose(cube))
        for inner in (True, False):
            with self.subTest(inner=inner):
                e.clear_evidence()
                with patch.object(StrictScene, "_hook_inner_only", return_value=inner):
                    self.sample({"head", "shaft"}, movement=.003)
                result = verdict("hook", e.available_methods, e.strict)
                self.assertEqual(result["failed"], not inner)
                self.assertEqual(e.strict["handheld"]["hook"]["correct_progress_m"], 0)
                if not inner:
                    self.assertIn("hook_final_approach_used_non_inner_face", result["violations"])


class NativeBallImpactEvidence(unittest.TestCase):
    def test_impact_records_incoming_speed_and_resets_between_trials(self):
        e = MoveCube(seed=2, variant="ball", obs_mode="state")
        self.addCleanup(e.close)
        e.ball._bodies[0].disable_gravity = True
        e.cube._bodies[0].disable_gravity = True
        for speed in (.09, .07):
            with self.subTest(speed=speed):
                e.cube.set_pose(sapien.Pose([0, 0, .4]))
                e.ball.set_pose(sapien.Pose([-(e.layout.ball_radius + .02 + .002), 0, .4]))
                e.ball.set_linear_velocity([speed, 0, 0])
                e.cube.set_linear_velocity([0, 0, 0])
                e.ball.set_angular_velocity([0, 0, 0])
                e.cube.set_angular_velocity([0, 0, 0])
                e.clear_evidence()
                for _ in range(50):
                    incoming = np.linalg.norm(e.ball._bodies[0].linear_velocity)
                    e.scene.step()
                    StrictScene._after_simulation_step(e)
                    if "ball" in e.current_contacts:
                        break
                else:
                    self.fail("Ball did not reach cube")
                self.assertAlmostEqual(e.strict["ball"]["impact_speed_mps"], incoming, places=7)
                self.assertEqual(e.strict["ball"]["impact_speed_mps"] > .08, speed > .08)
                if speed > .08:
                    self.assertLess(np.linalg.norm(e.ball._bodies[0].linear_velocity), .08)


class NativePanReleaseEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MoveCube(seed=2, variant="scoop", obs_mode="state")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_cube_on_shaft_or_handle_has_not_separated_from_pan(self):
        e = self.env
        for part in ("shaft", "handle"):
            with self.subTest(part=part):
                e.clear_evidence()
                tracker = e._pan_unload
                head = np.array([0, 0, 0.10])
                for pitch, local_x in ((0.2, 0), (0.5, 0.055)):
                    c, s = np.cos(pitch), np.sin(pitch)
                    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                    tracker.update(dt=.002, carried=True, held=True, contact=True,
                                   table_contact=False, head=head, rotation=rotation,
                                   cube=head + rotation @ [local_x, 0, .023],
                                   target=[.15, 0, .02])
                # Feed non-bowl contacts for longer than the 40 ms release
                # confirmation. The evidence pipeline must treat them as pan
                # contact, rather than falsely confirming airborne separation.
                e.current_contacts = {"scoop_" + part}
                with patch.object(Scene, "_after_simulation_step", return_value=None):
                    for _ in range(25):
                        e._after_simulation_step()
                self.assertEqual(tracker.free_seconds, 0)
                self.assertFalse(tracker.separated)
                self.assertFalse(e.strict["pan_unload"]["tilted_release"])


if __name__ == "__main__":
    unittest.main()
