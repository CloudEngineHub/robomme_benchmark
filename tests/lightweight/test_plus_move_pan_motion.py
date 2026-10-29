"""Continuous pan transport preserves reachability and motion limits."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mplib
import numpy as np
import torch

from robomme_plus.tasks.imitation.move_cube_support import oracle, reference


class PanMotionTests(unittest.TestCase):
    def test_lift_readiness_uses_cube_footprint_and_rim_clearance(self):
        self.assertTrue(oracle.pan_footprint_ready([.018, 0., .022], np.eye(3), .08, .065))
        self.assertFalse(oracle.pan_footprint_ready([.030, 0., .022], np.eye(3), .08, .065))
        self.assertFalse(oracle.pan_footprint_ready([0., .015, .022], np.eye(3), .08, .065))
        self.assertFalse(oracle.pan_footprint_ready([0., 0., -.022], np.eye(3), .08, .065))
        # A rotated cube needs more width than an axis-aligned one.
        r = oracle.euler2mat(0, 0, np.pi / 4)
        self.assertFalse(oracle.pan_footprint_ready([0., .004, .022], r, .08, .065))



    def test_private_search_also_stops_at_the_phase_budget(self):
        env = SimpleNamespace(
            layout=SimpleNamespace(ramp_height=.12, pan_face='left_side'),
            step=Mock(return_value=(None, 0, False, False, {})),
            control_freq=1, close=Mock(), _failed=Mock(return_value=False),
        )
        env._layouts = {'execution': env.layout}
        original_step = env.step
        tried = []
        def solve(base, planner, params):
            tried.append(params)
            base.step(np.zeros(15))
            base.step(np.zeros(15))  # Must be rejected before reaching physics.
            raise AssertionError('Budget was not enforced')
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'reset_reference'), \
             patch.object(reference, 'planner_for', return_value=Mock()), \
             patch.object(reference, 'solve', side_effect=solve), \
             patch.object(reference, 'MAX_SCOOP_REFERENCE_SECONDS', 1):
            with self.assertRaisesRegex(RuntimeError, 'No validated scoop reference'):
                reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution')
        self.assertEqual(original_step.call_count, len(tried))

    def test_smallest_safe_turn_wins_over_larger_joint_margin(self):
        options = [(.10, np.pi, np.ones(3), 0.),
                   (.03, 0., np.zeros(3), 0.),
                   (.08, np.pi / 4, np.zeros(3), 0.)]
        self.assertEqual(oracle.choose_pan_carry(options, 0., direct=True)[1], 0.)
        self.assertEqual(oracle.choose_pan_carry(options, 0., direct=False)[1], np.pi)
        options[1] = (.001, 0., np.zeros(3), 0.)
        self.assertEqual(oracle.choose_pan_carry(options, 0., direct=True)[1], np.pi / 4)

    def test_reachable_zero_turn_can_translate_before_turning(self):
        options = [(.10, np.pi / 2, np.ones(3), 0.),
                   (.03, 0., np.zeros(3), .45)]
        self.assertEqual(oracle.choose_pan_carry(options, 0., direct=True)[1], 0.)

    def test_gentle_contact_does_not_slow_the_supported_carry(self):
        base = SimpleNamespace(_pan_motion_version=2, _pan_gentle_contact=True,
                               _reference_scoop_stage='loading')
        with patch.object(oracle, 'move_pan_direct') as move:
            oracle.move_loaded_pan(base, Mock(), np.zeros(3), np.eye(3))
            self.assertEqual(move.call_args.kwargs['speed'], .20)
            base._reference_scoop_stage = 'carrying'
            oracle.move_loaded_pan(base, Mock(), np.zeros(3), np.eye(3))
            self.assertEqual(move.call_args.kwargs['speed'], .35)

    def setup_motion(self):
        robot = SimpleNamespace(
            get_qpos=lambda: torch.zeros((1, 9)),
            get_qlimits=lambda: torch.tensor([[[-2., 2.]] * 9]),
        )
        base = SimpleNamespace(_imitation_head=object(), agent=SimpleNamespace(robot=robot), control_freq=100)
        inner = SimpleNamespace(joint_vel_limits=np.full(7, .7), joint_acc_limits=np.full(7, .8),
                                transform_goal_to_wrt_base=lambda goal: goal)
        def ik(goal, q, **kwargs):
            nxt = q.copy()
            nxt[:3] = goal[:3]
            return 'Success', [nxt]
        inner.IK = ik
        inner.TOPP = lambda path, step: mplib.Planner.TOPP(inner, path, step)
        planner = SimpleNamespace(env=base, planner=inner, follow_path=Mock())
        return base, planner

    def execute(self, base, planner, point=None, **kwargs):
        with patch.object(oracle, 'ensure_pan_grasp'), \
             patch.object(oracle, 'pos', return_value=np.zeros(3)), \
             patch.object(oracle, 'rotation', return_value=np.eye(3)), \
             patch.object(oracle, 'relative_tcp', return_value=(np.zeros(3), np.eye(3))):
            oracle.move_pan_direct(base, planner, np.array([.1, 0., 0.]) if point is None else point, np.eye(3), **kwargs)

    def test_cartesian_knots_are_one_bounded_continuous_motion(self):
        base, planner = self.setup_motion()
        self.execute(base, planner)
        planner.follow_path.assert_called_once()
        path = planner.follow_path.call_args.args[0]
        np.testing.assert_allclose(path['position'][0], np.zeros(7), atol=1e-7)
        np.testing.assert_allclose(path['position'][-1], [.1, 0., 0., 0., 0., 0., 0.], atol=1e-7)
        self.assertTrue(np.all(path['velocity'][1:-1, 0] > 0))
        self.assertLessEqual(np.max(np.abs(path['velocity'])), .35 + 1e-6)
        np.testing.assert_allclose(planner.planner.joint_vel_limits, .7)
        np.testing.assert_allclose(planner.planner.joint_acc_limits, .8)

    def test_failed_retiming_restores_limits_and_never_moves(self):
        base, planner = self.setup_motion()
        planner.planner.TOPP = Mock(side_effect=RuntimeError('retiming failed'))
        with self.assertRaisesRegex(RuntimeError, 'retiming failed'):
            self.execute(base, planner)
        planner.follow_path.assert_not_called()
        np.testing.assert_allclose(planner.planner.joint_vel_limits, .7)
        np.testing.assert_allclose(planner.planner.joint_acc_limits, .8)

    def test_already_satisfied_ik_keeps_contact_without_retiming_constant_path(self):
        base, planner = self.setup_motion()
        planner.planner.IK = lambda goal, q, **kwargs: ('Success', [q.copy()])
        planner.planner.TOPP = Mock(side_effect=AssertionError('Constant path'))
        self.execute(base, planner, point=np.array([.0002, 0., 0.]), position_tolerance=.00015)
        path = planner.follow_path.call_args.args[0]
        self.assertEqual(len(path['position']), 6)
        np.testing.assert_allclose(path['position'], 0)
        np.testing.assert_allclose(path['velocity'], 0)

    def test_short_contact_stroke_has_bounded_fallback_if_topp_is_singular(self):
        base, planner = self.setup_motion()
        planner.planner.TOPP = Mock(side_effect=RuntimeError('Fail to parameterize path'))
        self.execute(base, planner, point=np.array([.005, 0., 0.]))
        path = planner.follow_path.call_args.args[0]
        np.testing.assert_allclose(path['position'][-1], [.005, 0., 0., 0., 0., 0., 0.])
        self.assertLessEqual(np.max(np.abs(path['velocity'])), .35)
        acceleration = np.diff(path['velocity'], axis=0) * base.control_freq
        self.assertLessEqual(np.max(np.abs(acceleration)), .35 + 1e-6)
        np.testing.assert_allclose(planner.planner.joint_acc_limits, .8)

    def test_ik_branch_jump_rejected_before_motion(self):
        base, planner = self.setup_motion()
        planner.planner.IK = lambda goal, q, **kwargs: ('Success', [q + 1.])
        with self.assertRaisesRegex(RuntimeError, 'branch change'):
            self.execute(base, planner)
        planner.follow_path.assert_not_called()


if __name__ == '__main__':
    unittest.main()
