"""Action errors consume budget without advancing physics or ending early."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import gymnasium as gym
import numpy as np

from robomme_plus.evaluation.clock import ExecutionClock
from robomme_plus.evaluation.failure import FailAwareWrapper


class RecordingEnv(gym.Env):
    def __init__(self, limit=250):
        self.execution_clock = ExecutionClock({"max_steps": limit, "oracle_policy_step": 30})
        self.action_space = gym.spaces.Box(-1, 1, shape=(8,))
        self.observation_space = gym.spaces.Dict({})
        self.task_name = "BinFill"
        self.reset()

    def observation(self):
        return {"front_rgb_list": [np.full((2, 2, 3), self.physics_steps)],
                "eef_state_list": [np.array([self.physics_steps, 0, 0, 0, 0, 0])]}

    def reset(self, **kwargs):
        self.physics_steps = self.calls = 0
        self.status = "ongoing"
        self.render_error = False
        self.execution_clock.start()
        return self.observation(), {"task_goal": ["test goal"], "status": self.status,
                                    **self.execution_clock.information()}

    def observe_current(self):
        if self.render_error:
            raise RuntimeError("render unavailable")
        return self.observation(), {"task_goal": ["test goal"], "status": self.status}

    def step(self, action):
        self.calls += 1
        if action == "raise":
            raise RuntimeError("planning exhausted")
        if action == "error_tuple":
            return {}, 0, True, False, {"status": "error", "error_message": "IK failed"}
        if action == "partial":
            for _ in range(3):
                self.physics_steps += 1
                self.execution_clock.advance()
            raise RuntimeError("error after partial motion")
        if action == "partial_failure":
            self.physics_steps += 1
            self.execution_clock.advance()
            self.status = "fail"
            raise RuntimeError("error after task failure")
        self.physics_steps += 1
        expired = self.execution_clock.advance()
        self.status = action if action in ("success", "fail") else "timeout" if expired else "ongoing"
        return (self.observation(), 1, self.status in ("success", "fail"),
                self.status == "timeout", {"status": self.status, **self.execution_clock.information()})


class ErrorRecoveryTest(unittest.TestCase):
    def make_env(self, limit=250):
        native = RecordingEnv(limit)
        env = FailAwareWrapper(native)
        env.reset()
        return native, env

    def test_exception_and_returned_error_are_recoverable_without_physics(self):
        for action in ("raise", "error_tuple"):
            with self.subTest(action=action):
                native, env = self.make_env()
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(native.physics_steps, 0)
                self.assertEqual(native.calls, 1)
                self.assertFalse(terminated or truncated)
                self.assertEqual(reward, 0)
                self.assertEqual(info["status"], "ongoing")
                self.assertTrue(info["action_error"])
                self.assertEqual(info["error_penalty_steps"], 100)
                self.assertEqual(info["execution_control_steps"], 100)
                self.assertEqual(info["execution_penalty_steps"], 100)
                self.assertEqual(info["remaining_steps"], 150)
                self.assertTrue(info["error_message"])
                self.assertEqual(len(obs["front_rgb_list"]), 1)
                self.assertEqual(info["task_goal"], ["test goal"])

    def test_valid_action_after_error_executes_and_clears_old_error(self):
        native, env = self.make_env()
        env.step("raise")
        obs, _, terminated, truncated, info = env.step("valid")
        self.assertEqual(native.physics_steps, 1)
        self.assertEqual(obs["eef_state_list"][0][0], 1)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["execution_control_steps"], 101)
        self.assertEqual(info["execution_penalty_steps"], 100)
        for name in FailAwareWrapper._ERROR_FIELDS:
            self.assertNotIn(name, info)

    def test_repeated_errors_eventually_timeout_and_terminal_is_latched(self):
        native, env = self.make_env()
        for expected in (100, 200, 300):
            result = env.step("raise")
            _, _, terminated, truncated, info = result
            self.assertEqual(info["execution_control_steps"], expected)
            self.assertFalse(terminated)
            self.assertEqual(truncated, expected >= 250)
        self.assertEqual(info["status"], "timeout")
        self.assertEqual(info["remaining_steps"], 0)
        self.assertEqual(native.physics_steps, 0)
        self.assertIs(env.step("valid"), result)
        self.assertEqual(native.calls, 3)
        self.assertEqual(native.execution_clock.steps, 300)

    def test_partial_execution_is_observed_and_charged_in_addition_to_penalty(self):
        native, env = self.make_env()
        obs, _, _, _, info = env.step("partial")
        self.assertEqual(native.physics_steps, 3)
        self.assertEqual(native.calls, 1)
        self.assertEqual(obs["eef_state_list"][0][0], 3)
        self.assertEqual(info["execution_control_steps"], 103)
        self.assertEqual(info["execution_penalty_steps"], 100)

    def test_physical_step_can_exhaust_budget_after_a_penalty(self):
        native, env = self.make_env(limit=101)
        self.assertFalse(env.step("raise")[3])
        _, _, terminated, truncated, info = env.step("valid")
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["status"], "timeout")
        self.assertEqual(native.physics_steps, 1)
        self.assertEqual(info["execution_control_steps"], 101)

    def test_error_does_not_erase_task_failure_during_partial_execution(self):
        _, env = self.make_env(limit=100)
        _, _, terminated, truncated, info = env.step("partial_failure")
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["status"], "fail")
        self.assertEqual(info["execution_control_steps"], 101)

    def test_success_and_task_failure_still_terminate(self):
        for status in ("success", "fail"):
            native, env = self.make_env()
            result = env.step(status)
            self.assertTrue(result[2])
            self.assertEqual(result[4]["status"], status)
            self.assertIs(env.step("raise"), result)
            self.assertEqual(native.execution_clock.penalty_steps, 0)
            self.assertEqual(native.calls, 1)

    def test_reset_clears_timeout_and_penalties(self):
        native, env = self.make_env(limit=100)
        self.assertTrue(env.step("raise")[3])
        _, info = env.reset()
        self.assertEqual(info["execution_control_steps"], 0)
        self.assertEqual(info["execution_penalty_steps"], 0)
        self.assertEqual(env.step("valid")[4]["status"], "ongoing")
        self.assertEqual(native.physics_steps, 1)

    def test_render_error_falls_back_to_one_frame_and_still_times_out(self):
        native, env = self.make_env(limit=100)
        env._last_obs["front_rgb_list"] *= 4  # Initial demo/history must not be replayed.
        native.render_error = True
        obs, _, _, truncated, info = env.step("raise")
        self.assertEqual(len(obs["front_rgb_list"]), 1)
        self.assertEqual(info["observation_error"], "render unavailable")
        self.assertTrue(truncated)
        self.assertEqual(native.physics_steps, 0)

    def test_ee_pose_ik_error_reaches_the_shared_recovery_handler(self):
        import torch
        from robomme_plus.control.ee_pose import EndeffectorDemonstrationWrapper

        native = RecordingEnv()
        ee = EndeffectorDemonstrationWrapper(native)
        env = FailAwareWrapper(ee)
        env.reset()
        ee._ee_pose_planner = SimpleNamespace(
            robot=SimpleNamespace(get_qpos=lambda: torch.zeros((1, 9))),
            planner=SimpleNamespace(transform_goal_to_wrt_base=lambda pose: pose,
                                    IK=Mock(return_value=("IK Failed", []))),
        )
        obs, _, terminated, truncated, info = env.step([99, 99, 99, 0, 0, 0, 1])
        self.assertFalse(terminated or truncated)
        self.assertIn("IK failed", info["error_message"])
        self.assertEqual(info["execution_control_steps"], 100)
        self.assertEqual(native.physics_steps, 0)
        self.assertEqual(native.calls, 0)
        self.assertEqual(len(obs["front_rgb_list"]), 1)

    def test_penalty_does_not_count_during_demonstration(self):
        clock = ExecutionClock({"max_steps": 100, "oracle_policy_step": 10})
        clock.advance()
        clock.penalize(100)
        self.assertEqual(clock.steps, 0)
        self.assertFalse(clock.expired)


if __name__ == "__main__":
    unittest.main()
