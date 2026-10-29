"""Oracle motion uses the official criterion rather than legacy contact bans."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robomme_plus.tasks.imitation.move_cube_support import oracle, reference


class ReferenceContactCriteria(unittest.TestCase):
    def test_execution_candidates_use_execution_ramp_geometry(self):
        env = SimpleNamespace(layout=SimpleNamespace(ramp_height=.08, pan_face='left_side'),
                              step=Mock(), close=Mock(), _failed=Mock(return_value=False))
        env._layouts = {'execution': SimpleNamespace(ramp_height=.25, pan_face='inclined_face')}
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'reset_reference'), \
             patch.object(reference, 'planner_for', return_value=Mock()), \
             patch.object(reference, 'validate_reference_loading_surface'), \
             patch.object(reference, 'solve', return_value={'success': True}):
            params = reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution')
        options = params['scoop_options']
        self.assertTrue(options['follow_incline'])
        self.assertEqual(options['approach_mode'], 'straight')
        self.assertEqual(options['lift_before_withdraw_m'], .04)
        self.assertAlmostEqual(options['carry_height_m'], .10)

    def test_pan_search_does_not_repeat_failed_routing_for_every_carry_option(self):
        env = SimpleNamespace(layout=SimpleNamespace(ramp_height=.12, pan_face='left_side'),
                              step=Mock(), close=Mock(), _failed=Mock(return_value=False))
        env._layouts = {'execution': env.layout}
        tried = []
        def solve(base, planner, parameters):
            tried.append(parameters['flip'])
            base._reference_scoop_stage = 'routing'
            raise RuntimeError('Pan route overshot the loading site')
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'reset_reference'), \
             patch.object(reference, 'planner_for', return_value=Mock()), \
             patch.object(reference, 'solve', side_effect=solve):
            with self.assertRaisesRegex(RuntimeError, 'No validated scoop reference'):
                reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution')
        self.assertEqual(tried, [0, 1, 0, 1])

    def test_pan_recording_keeps_the_calibrated_unload_direction(self):
        choices = oracle.pan_carry_candidates(np.radians(-98.9), 81.1, 0.45)
        self.assertEqual(len(choices), 1)
        self.assertAlmostEqual(choices[0][0], 0.45)
        self.assertAlmostEqual(np.degrees(choices[0][1]), 81.1)
        # Keep the chosen rotation path even when the angles cross +/-180.
        self.assertAlmostEqual(oracle.pan_carry_candidates(0, 225, 0)[0][1],
                               np.radians(225))
        for yaw, fraction in ((float('nan'), 0), (0, -0.1), (0, 1.1)):
            with self.assertRaises(ValueError):
                oracle.pan_carry_candidates(0, yaw, fraction)

    def test_pan_search_saves_the_successful_carry_strategy(self):
        env = SimpleNamespace(
            layout=SimpleNamespace(ramp_height=.12, pan_face='left_side'),
            step=Mock(), close=Mock(), _failed=Mock(return_value=False),
        )
        env._layouts = {'execution': env.layout}
        def solve(base, planner, parameters):
            base._reference_pan_carry_strategy = {
                'carry_yaw_deg': 81.1, 'carry_turn_fraction': .45}
            return {'success': True}
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'reset_reference'), \
             patch.object(reference, 'planner_for', return_value=Mock()), \
             patch.object(reference, 'validate_reference_loading_surface'), \
             patch.object(reference, 'solve', side_effect=solve):
            result = reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution')
        self.assertEqual(result['scoop_options']['carry_yaw_deg'], 81.1)
        self.assertEqual(result['scoop_options']['carry_turn_fraction'], .45)

    def test_pan_search_skips_only_permanently_illegal_unchanged_approaches(self):
        # Exercise the candidate loop: changing carry options cannot fix a
        # collision during routing, but a new grasp must still be attempted.
        for stage, permanent, expected_trials in (
            ('routing', True, 2), ('grasp', True, 2),
            ('loading', True, 3), ('routing', False, 3),
        ):
            with self.subTest(stage=stage, permanent=permanent):
                env = SimpleNamespace(
                    layout=SimpleNamespace(ramp_height=.12, pan_face='left_side'),
                    step=Mock(), close=Mock(), _failed=Mock(return_value=permanent),
                )
                env._layouts = {'execution': env.layout}
                tried = []
                def solve(base, planner, parameters):
                    tried.append(parameters)
                    base._reference_scoop_stage = stage
                    if len(tried) <= 2:
                        raise RuntimeError('Approach rejected')
                    return {'success': True}
                with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
                     patch.object(reference, 'reset_reference'), \
                     patch.object(reference, 'planner_for', return_value=Mock()), \
                     patch.object(reference, 'validate_reference_loading_surface'), \
                     patch.object(reference, 'solve', side_effect=solve):
                    if expected_trials == 2:
                        with self.assertRaisesRegex(RuntimeError, 'No validated scoop reference'):
                            reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution')
                    else:
                        result = reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution')
                        self.assertEqual(result['flip'], 0)
                self.assertEqual(len(tried), expected_trials)
                self.assertEqual([p['flip'] for p in tried[:2]], [0, 1])
                env.close.assert_called_once()

    def environment(self, failed):
        return SimpleNamespace(
            _imitation_head=object(),
            _imitation_evidence={'execution': {'shaft_contact_steps': 1, 'handle_contact_steps': 1}},
            _failed=Mock(return_value=failed),
            result=lambda: {'criteria': {'failed': failed, 'violations': ['robot_or_gripper_touched_cube'] if failed else []}},
        )

    def test_contacts_allowed_by_current_criterion_do_not_abort_reference(self):
        env = self.environment(False)
        with patch.object(oracle, 'relative_tcp', return_value=(np.zeros(3), np.eye(3))), patch.object(oracle, 'move_pose') as move:
            oracle.move_head(env, None, np.zeros(3), np.eye(3), corrections=2)
        self.assertEqual(move.call_count, 2)
        self.assertEqual(env._failed.call_count, 2)

    def test_current_criterion_violation_still_aborts_immediately(self):
        env = self.environment(True)
        with patch.object(oracle, 'relative_tcp', return_value=(np.zeros(3), np.eye(3))), patch.object(oracle, 'move_pose') as move:
            with self.assertRaisesRegex(RuntimeError, 'robot_or_gripper_touched_cube'):
                oracle.move_head(env, None, np.zeros(3), np.eye(3), corrections=2)
        move.assert_called_once()

    def test_pan_grasp_check_uses_the_same_current_criterion(self):
        env = self.environment(False)
        env._imitation_handle = object()
        env.agent = SimpleNamespace(is_grasping=lambda handle: np.array(True))
        oracle.ensure_pan_grasp(SimpleNamespace(unwrapped=env))
        env._failed.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'Illegal pan motion'):
            oracle.ensure_pan_grasp(SimpleNamespace(unwrapped=env))

    def test_poke_separates_after_small_overshoot_but_rejects_large_miss(self):
        for distance, should_fail in ((.008, False), (.0155, False), (.025, True)):
            with self.subTest(distance=distance):
                cube, head = object(), object()
                base = SimpleNamespace(cube=cube, _imitation_head=head,
                    _imitation_tool={'ball_radius_m': .015},
                    layout=SimpleNamespace(target_heading_deg=0.))
                env = SimpleNamespace(unwrapped=base)
                def position(actor):
                    return np.array([distance if actor is cube else 0., 0., .02])
                with patch.object(oracle, 'pos', side_effect=position), patch.object(oracle, 'move_head') as move, patch.object(oracle, 'hold') as hold:
                    if should_fail:
                        with self.assertRaisesRegex(RuntimeError, 'overshot'):
                            oracle.move_contact(env, None, 'poke', np.array([0., 0., .02]), finish_error=.003)
                        hold.assert_not_called()
                    else:
                        oracle.move_contact(env, None, 'poke', np.array([0., 0., .02]), finish_error=.003)
                        self.assertEqual(move.call_count, 3)  # Approach, descend, separate.
                        hold.assert_called_once()


if __name__ == '__main__':
    unittest.main()
