"""Private planning deadlines leave robot execution and existing timers alone."""
import signal
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import sapien

from robomme_plus.tasks.imitation.move_cube_support.reference import (
    PandaArmMotionPlanningSolver, ReferencePlanner, planning_deadline,
)


class ReferencePlanningDeadline(unittest.TestCase):
    def test_cue_strike_and_settle_use_physical_time_at_both_rates(self):
        from robomme_plus.tasks.imitation.move_cube_support import oracle

        path = np.array([np.zeros(7), np.full(7, 0.02)])
        for hz in (20, 100):
            base = SimpleNamespace(control_freq=hz, result=lambda: {"success": True})
            env = SimpleNamespace(unwrapped=base)
            for duration in (0.04, 0.3):
                with patch.object(oracle, 'joint_step') as step:
                    self.assertTrue(oracle.cue_stroke(env, path, duration, settle=0.5)['success'])
                # The first command holds the path's initial position; later
                # commands traverse it in the requested physical duration.
                stroke_steps = len(step.call_args_list) - round(0.5 * hz) - 1
                self.assertLessEqual(abs(stroke_steps / hz - duration), 1 / hz)
                np.testing.assert_array_equal(step.call_args_list[0].args[1], path[0])
                np.testing.assert_array_equal(step.call_args_list[stroke_steps].args[1], path[-1])
                self.assertTrue(np.isfinite([call.args[2] for call in step.call_args_list]).all())

    def test_gripper_and_refinement_dwell_keep_their_physical_duration(self):
        for hz, close_steps, refine_steps in ((20, 8, 3), (100, 40, 15)):
            planner = object.__new__(ReferencePlanner)
            planner.base_env = SimpleNamespace(control_freq=hz)
            with patch.object(PandaArmMotionPlanningSolver, 'open_gripper') as opened, \
                 patch.object(PandaArmMotionPlanningSolver, 'close_gripper') as closed, \
                 patch.object(PandaArmMotionPlanningSolver, 'follow_path') as followed:
                planner.open_gripper(t=40)
                planner.close_gripper(t=40)
                opened.assert_called_once_with(t=close_steps, gripper_state=None)
                closed.assert_called_once_with(t=close_steps, gripper_state=None)
                planner.follow_path({'position': np.zeros((10, 7))}, refine_steps=15)
                followed.assert_called_once_with({'position': unittest.mock.ANY}, refine_steps=refine_steps)

    def test_degenerate_long_waypoint_is_rejected_before_simulation(self):
        planner = object.__new__(ReferencePlanner)
        planner.base_env = SimpleNamespace(control_freq=100)
        with patch.object(PandaArmMotionPlanningSolver, 'follow_path', return_value='executed') as follow:
            with self.assertRaisesRegex(RuntimeError, 'waypoint lasts'):
                planner.follow_path({'position': np.zeros((6001, 7))})
            follow.assert_not_called()
            self.assertEqual(planner.follow_path({'position': np.zeros((500, 7))}), 'executed')

    def test_expiration_restores_handler_and_cancels_timer(self):
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with planning_deadline(.01):
                time.sleep(.1)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_existing_timer_is_not_replaced(self):
        signal.setitimer(signal.ITIMER_REAL, 60.)
        try:
            with planning_deadline(.01):
                self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 59.)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)

    def test_timeout_returns_failure_without_executing(self):
        planner = object.__new__(ReferencePlanner)
        planner.follow_path = Mock()
        with patch.object(PandaArmMotionPlanningSolver, 'move_to_pose_with_screw', side_effect=TimeoutError):
            self.assertEqual(planner.move_to_pose_with_screw(sapien.Pose()), -1)
        planner.follow_path.assert_not_called()

    def test_robot_execution_is_outside_timer_and_dry_run_does_not_execute(self):
        planner = object.__new__(ReferencePlanner)
        path = {'position': np.ones((2, 7))}

        def follow(result, refine_steps):
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))
            self.assertIs(result, path)
            self.assertEqual(refine_steps, 7)
            return 'executed'

        planner.follow_path = Mock(side_effect=follow)
        with patch.object(PandaArmMotionPlanningSolver, 'move_to_pose_with_RRTConnect', return_value=path):
            self.assertIs(planner.move_to_pose_with_RRTStar(sapien.Pose(), dry_run=True), path)
            planner.follow_path.assert_not_called()
            self.assertEqual(planner.move_to_pose_with_RRTConnect(sapien.Pose(), refine_steps=7), 'executed')


if __name__ == '__main__':
    unittest.main()
