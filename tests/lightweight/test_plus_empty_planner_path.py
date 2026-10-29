import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import torch

from robomme_plus.control.planner import _EmptyPathMixin, _FailAwareMixin
from robomme_plus.control.waypoint import MultiStepDemonstrationWrapper
from robomme_plus.evaluation.clock import ExecutionClock
from robomme_plus.evaluation.failure import FailAwareWrapper


class EmptyPathTest(unittest.TestCase):
    def test_empty_path_does_not_step_and_nonempty_path_delegates(self):
        class Parent:
            def follow_path(self, result, refine_steps=0):
                return result, refine_steps
        class Planner(_EmptyPathMixin, Parent):
            pass
        planner = Planner()
        self.assertIsNone(planner.follow_path({"position": np.empty((0, 7))}, 3))
        path = {"position": np.zeros((2, 7))}
        self.assertEqual(planner.follow_path(path, 3), (path, 3))


class WaypointEnv(gym.Env):
    def __init__(self, task, limit):
        self.task_name, self.spec = task, SimpleNamespace(id=task)
        self.execution_clock = ExecutionClock(dict(max_steps=limit, oracle_policy_step=10))
        self.agent = SimpleNamespace(
            tcp=SimpleNamespace(pose=SimpleNamespace(
                p=torch.zeros((1, 3)), q=torch.tensor([[1., 0., 0., 0.]]))),
            robot=SimpleNamespace(get_qpos=lambda: torch.zeros((1, 7 if task in (
                'PatternLock', 'RouteStick') else 9))),
        )

    def reset(self, **kwargs):
        self.execution_clock.start()
        self.calls = 0
        return self.observe_current()

    def observe_current(self):
        return {'frame': self.calls}, {'status': 'ongoing', **self.execution_clock.information()}

    def step(self, action):
        self.calls += 1
        expired = self.execution_clock.advance()
        obs, info = self.observe_current()
        info['status'] = 'timeout' if expired else 'ongoing'
        return obs, 0., False, expired, info


class WaypointRecoveryTest(unittest.TestCase):
    def make_env(self, task='ButtonUnmask', limit=500, screw=None, rrt=None):
        base = WaypointEnv(task, limit)

        class Parent:
            move_to_pose_with_screw = Mock(side_effect=screw) if isinstance(screw, Exception) else Mock(return_value=screw)

        class Planner(_FailAwareMixin, Parent):
            pass

        planner = Planner()
        planner.env, planner.gripper_state = base, 1
        planner.move_to_pose_with_RRTStar = Mock(return_value=rrt)
        waypoint = MultiStepDemonstrationWrapper(base)
        waypoint._get_planner = lambda: planner
        env = FailAwareWrapper(waypoint)
        env.reset()
        return base, planner, env

    def test_empty_plans_return_a_frame_and_consume_one_step_for_arms_and_sticks(self):
        for task, grippers in [('ButtonUnmask', [0]), ('PatternLock', [-1, 0, 1]), ('RouteStick', [-1, 0, 1])]:
            for gripper in grippers:
                with self.subTest(task=task, gripper=gripper):
                    base, _, env = self.make_env(task, limit=1)
                    obs, _, done, truncated, info = env.step([0, 0, 0, .0003, 0, 0, gripper])
                    self.assertEqual(obs['frame'], [1])
                    self.assertEqual(base.calls, 1)
                    self.assertFalse(done)
                    self.assertTrue(truncated)
                    self.assertEqual(info['execution_penalty_steps'], 0)
                    self.assertIsNotNone(env._terminal_result)

    def test_parameterization_failure_reaches_rrt_but_unrelated_errors_do_not(self):
        for message, expected_rrt in [('Fail to parameterize path', 1), ('render failed', 0)]:
            with self.subTest(message=message):
                base, planner, env = self.make_env(screw=RuntimeError(message))
                result = env.step([.01, 0, 0, 0, 0, 0, 0])
                self.assertEqual(planner.move_to_pose_with_RRTStar.call_count, expected_rrt)
                self.assertEqual(base.calls, expected_rrt)
                self.assertEqual(result[4]['execution_penalty_steps'], 0 if expected_rrt else 100)

    def test_failed_tiny_adjustments_hold_but_larger_moves_still_report_errors(self):
        for distance in (.00002, .0001, .001, .002):
            with self.subTest(distance=distance):
                base, _, env = self.make_env(screw=-1, rrt=-1)
                result = env.step([distance, 0, 0, 0, 0, 0, 0])
                self.assertEqual(base.calls, int(distance <= .001))
                self.assertEqual(result[4]['execution_penalty_steps'], 0 if distance <= .001 else 100)
