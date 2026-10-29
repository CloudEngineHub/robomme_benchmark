"""Real native scene resets and public per-trial budget/observation behavior."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from robomme_plus.registry import TASKS
from robomme_plus.tasks.imitation.move_cube import MoveCube
from robomme_plus.tasks.imitation.move_cube_support.scene import REST_QPOS, pos
from robomme_plus.tasks.imitation.move_cube_support.trials import RETRY_MESSAGE
from robomme_plus.evaluation.demonstration import DemonstrationWrapper
from robomme_plus.evaluation.failure import FailAwareWrapper
from robomme_plus.evaluation import batches
from tests.lightweight.test_plus_move_success import successful


class InternalMoveCubeTrials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native_env = gym.make(TASKS["MoveCube"].gym_id, seed=2, variant="ball",
                                 obs_mode="rgb+depth+segmentation")
        cls.native = cls.native_env.unwrapped

    @classmethod
    def tearDownClass(cls):
        cls.native_env.close()

    def setUp(self):
        e = self.native
        self.native_env.reset(seed=2)
        e._required_method = "ball"
        e._park_phase("demonstration")
        e._select_phase("execution")
        e._reset_active_scene()
        e.timestep = 2
        e.on_execution_start()
        self.wrapper = DemonstrationWrapper(self.native_env, episode_metadata={"max_steps": 3, "oracle_policy_step": 1})
        self.wrapper.execution_clock.start()
        self.action = np.r_[REST_QPOS[:7], 1.]

    def test_timeout_restores_same_scene_without_episode_reset_and_allows_only_two(self):
        e = self.native
        geometry = copy.deepcopy(e.scene_parameters)
        actors = (e.cube, e.ball, e.cue, e.goal_site)
        with patch.object(e, "reset", side_effect=AssertionError("must not reset episode")):
            self.wrapper.step(self.action)
            self.wrapper.step(self.action)
            obs, reward, done, truncated, info = self.wrapper.step(self.action)
            self.assertFalse(done or truncated)
            self.assertEqual(info["message"], RETRY_MESSAGE)
            self.assertTrue(info["trial_reset"])
            self.assertEqual(info["trial_index"], 2)
            self.assertEqual(info["trials_remaining"], 1)
            self.assertEqual(info["execution_control_steps"], 0)
            self.assertEqual(info["remaining_steps"], 3)
            self.assertEqual(info["execution_total_control_steps"], 3)
            self.assertEqual(int(e.elapsed_steps[0]), 0)
            self.assertEqual(e._phase, "execution")
            self.assertEqual(e.timestep, 2)
            self.assertEqual(e.scene_parameters, geometry)
            self.assertEqual((e.cube, e.ball, e.cue, e.goal_site), actors)
            np.testing.assert_allclose(pos(e.cube), e.layout.cube, atol=.001)
            np.testing.assert_allclose(pos(e.ball), e.layout.ball_rest_xyz, atol=.001)
            np.testing.assert_allclose(e.cue.get_qpos()[0, 1].item(), 0., atol=.001)
            np.testing.assert_allclose(e.agent.robot.get_qpos()[0].cpu(), REST_QPOS, atol=1e-6)
            self.assertEqual(len(obs["front_rgb_list"]), 1)
            self.assertEqual(len(e.result()["previous_trials"]), 1)
            for _ in range(3):
                result = self.wrapper.step(self.action)
            self.assertTrue(result[3])
            self.assertEqual(result[4]["status"], "timeout")
            self.assertNotIn("trial_reset", result[4])
            self.assertEqual(result[4]["execution_total_control_steps"], 6)
            before = int(e.elapsed_steps[0])
            self.wrapper.step(self.action)
            self.assertEqual(int(e.elapsed_steps[0]), before)

    def test_failed_trial_evidence_is_preserved_and_second_violation_is_terminal(self):
        for method in ("ball", "cue"):
            with self.subTest(method=method):
                self.setUp()
                e = self.native
                e._required_method = method
                e.strict["robot_cube_contact"] = True
                result = self.wrapper.step(self.action)
                self.assertFalse(result[2] or result[3])
                self.assertTrue(result[4]["trial_reset"])
                self.assertNotIn("failure_reason", result[4])
                self.assertIsNone(e._failure_code)
                self.assertFalse(e.strict["robot_cube_contact"])
                self.assertTrue(e.result()["previous_trials"][0]["result"]["strict_evidence"]["robot_cube_contact"])
                e.strict["robot_cube_contact"] = True
                result = self.wrapper.step(self.action)
                self.assertTrue(result[2])
                self.assertEqual(result[4]["status"], "fail")
                self.assertEqual(result[4]["failure_reason"],
                                 "The robot or gripper directly touched the cube.")
                self.assertEqual(e._execution_trial, 2)

    def test_success_on_either_trial_ends_successfully_without_another_reset(self):
        for number in (1, 2):
            with self.subTest(trial=number):
                self.setUp()
                e = self.native
                if number == 2:
                    e.strict["robot_cube_contact"] = True
                    self.wrapper.step(self.action)
                e.strict = successful("ball")
                with patch.object(e, "_after_simulation_step"):
                    result = self.wrapper.step(self.action)
                self.assertTrue(result[2])
                self.assertEqual(result[4]["status"], "success")
                self.assertNotIn("failure_reason", result[4])
                self.assertEqual(e._execution_trial, number)
                self.assertNotIn("trial_reset", result[4])

    def test_other_methods_and_demonstration_do_not_get_a_retry(self):
        e = self.native
        for method in ("hook", "poke", "scoop"):
            e._required_method = method
            self.assertIsNone(e.retry_execution_trial("fail", 1))
        e._required_method = "ball"
        e._policy_execution_active = False
        self.assertIsNone(e.retry_execution_trial("fail", 1))
        e._policy_execution_active = True
        e._phase = "demonstration"
        self.assertIsNone(e.retry_execution_trial("fail", 1))

    def test_oracle_failure_never_retries_even_through_policy_wrapper(self):
        for method in ("ball", "cue"):
            with self.subTest(method=method):
                self.setUp()
                e = self.native
                e._required_method = method
                e.scene_parameters.setdefault("oracle_parameters", {})["execution"] = {}

                def failed_oracle(env, planner, parameters):
                    e.strict["robot_cube_contact"] = True
                    result = env.step(self.action)
                    self.assertTrue(result[2])
                    self.assertEqual(result[4]["status"], "fail")
                    self.assertNotIn("trial_reset", result[4])

                with patch("robomme_plus.tasks.imitation.move_cube_support.oracle.solve", side_effect=failed_oracle) as solve:
                    with self.assertRaisesRegex(RuntimeError, "reference failed"):
                        e._solve_reference(self.wrapper, None)
                    solve.assert_called_once()
                self.assertEqual(e._execution_trial, 1)
                self.assertEqual(e._execution_trial_history, [])
                self.assertTrue(e._policy_execution_active)

    def test_penalty_timeout_also_gets_one_fresh_budget(self):
        env = FailAwareWrapper(self.wrapper)
        env._last_obs, env._last_info = self.wrapper.observe_current()
        with patch.object(self.wrapper, "step", side_effect=RuntimeError("IK failure")):
            result = env.step(self.action)
            self.assertFalse(result[2] or result[3])
            self.assertEqual(result[4]["execution_control_steps"], 0)
            self.assertEqual(result[4]["execution_penalty_steps"], 0)
            self.assertEqual(result[4]["execution_total_control_steps"], 100)
            self.assertTrue(result[4]["trial_reset"])
            result = env.step(self.action)
            self.assertTrue(result[3])
            self.assertEqual(result[4]["execution_total_control_steps"], 200)
            self.assertNotIn("trial_reset", result[4])

    def test_settled_miss_ends_shot_but_not_while_ball_is_moving(self):
        e = self.native
        e._strict_ball_separated = True
        e.ball.set_linear_velocity([.1, 0, 0])
        for _ in range(500):
            e._update_execution_trial()
        self.assertFalse(e._trial_missed)
        e.ball.set_linear_velocity([0, 0, 0])
        for _ in range(380):
            e._update_execution_trial()
        self.assertTrue(e._trial_missed)
        self.assertTrue(e._failed())
        result = self.wrapper.step(self.action)
        self.assertTrue(result[4]["trial_reset"])
        self.assertEqual(e._execution_trial_history[0]["reason"], "settled_shot_missed_target")


class CompoundActionRetry(unittest.TestCase):
    def test_retry_returns_to_policy_before_remaining_waypoint_steps(self):
        calls = []
        def step(action):
            calls.append(action)
            return {}, 0., False, False, {"trial_reset": len(calls) == 2}
        planner = SimpleNamespace(env=SimpleNamespace(step=step))
        def action():
            for i in range(8):
                planner.env.step(i)
        collected = batches._collect_dense_steps(planner, action)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(len(collected), 2)
        self.assertTrue(batches.interrupts_action(collected[-1]))
        self.assertFalse(collected[-1][2] or collected[-1][3])

    def test_waypoint_skips_gripper_after_reset_and_returns_notice(self):
        from robomme_plus.control.waypoint import MultiStepDemonstrationWrapper
        from unittest.mock import Mock
        class Native(gym.Env):
            task_name = "MoveCube"
            spec = SimpleNamespace(id="test")
            count = 0
            def step(self, action):
                self.count += 1
                return {"frame": self.count}, 0., False, False, {"trial_reset": self.count == 2}
        native = Native()
        waypoint = MultiStepDemonstrationWrapper(native)
        planner = SimpleNamespace(env=native, close_gripper=Mock(), open_gripper=Mock())
        def move(pose):
            for _ in range(10):
                planner.env.step(None)
        planner.move_to_pose_with_screw = move
        with patch.object(waypoint, "_at_pose", return_value=False), patch.object(waypoint, "_get_planner", return_value=planner):
            result = waypoint.step([0, 0, .2, 0, 0, 0, -1])
        self.assertEqual(native.count, 2)
        planner.close_gripper.assert_not_called()
        self.assertTrue(result[4]["trial_reset"])
        self.assertFalse(result[2] or result[3])


if __name__ == "__main__":
    unittest.main()
