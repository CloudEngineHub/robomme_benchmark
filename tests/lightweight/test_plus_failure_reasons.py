"""Rule failures expose their cause, but never hidden answers or stale feedback."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from robomme_plus.evaluation import batches
from robomme_plus.evaluation.failure import FailAwareWrapper
from robomme_plus.scoring.failures import clear_failure, failure_message, record_failure
from robomme_plus.scoring.predicates import sequential_task_check
from robomme_plus.scoring.subgoals import task
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.tasks.counting.utils.controllers import CountingController
from robomme_plus.tasks.imitation.utils.geometry import move_failed, move_failure_code
from robomme_plus.tasks.permanence.utils.progress import ColorPickupProgress
from tests.lightweight.test_plus_error_recovery import RecordingEnv


class FailureReasonsTest(unittest.TestCase):
    def test_container_feedback_preserves_duplicate_color_and_order_rules(self):
        progress = ColorPickupProgress(
            ["red", "blue"], {"red": 2, "blue": 1},
            {"hidden_a": "red", "hidden_b": "red", "hidden_c": "blue"},
        )
        self.assertIsNone(progress.invalid_reason(0, ["hidden_a"]))
        self.assertIsNone(progress.invalid_reason(0, ["hidden_b"]))
        self.assertEqual(progress.invalid_reason(0, ["hidden_c"]), "wrong_container")
        self.assertEqual(progress.invalid_reason(0, ["distractor"]), "wrong_container")
        self.assertEqual(progress.invalid_reason(0, ["hidden_a", "hidden_b"]), "multiple_containers")
        self.assertTrue(progress.observe(0, ["hidden_b"]))
        self.assertEqual(progress.invalid_reason(1, ["hidden_b"]), "repeated_container")
        self.assertFalse(progress.invalid(1, ["hidden_a"]))

    def test_failed_predicate_is_evaluated_once_and_reports_only_the_triggered_rule(self):
        env = SimpleNamespace(timestep=0)
        check = Mock(return_value=[torch.tensor([False]), np.array([True])])
        complete = Mock(return_value=True)
        entry = task('pick hidden red cube at <42, 91>', complete, None, failure=check, failure_reason=['wrong_cube', 'premature_button'])
        done, _, failed, _ = sequential_task_check(env, [entry], True)
        assert failed and (not done)
        check.assert_called_once_with()
        complete.assert_not_called()
        assert env.timestep == 0
        assert failure_message(env) == 'Pressed the button before completing the required actions.'

    def test_ongoing_predicate_does_not_record_a_reason(self):
        env = SimpleNamespace(timestep=0)
        entry = task('pick', lambda: False, None, failure=lambda: [False, False], failure_reason=['wrong_object', 'premature_button'])
        assert not sequential_task_check(env, [entry], True)[2]
        assert not hasattr(env, '_failure_code')

    def test_exception_reports_check_failure_without_exception_contents(self):
        env = SimpleNamespace(timestep=0)
        entry = task('hidden target', lambda: False, None, failure=Mock(side_effect=ValueError('goal=red_cube_19; x=0.3')), failure_reason='wrong_cube')
        assert sequential_task_check(env, [entry], True)[2]
        assert failure_message(env) == 'The simulator could not complete a task-rule check.'

    def test_first_known_cause_is_latched_and_unknown_values_are_not_exposed(self):
        env = SimpleNamespace()
        record_failure(env, {'private_goal': 'red cube at 0.3'})
        record_failure(env, 'private_goal: red cube at 0.3')
        assert failure_message(env) == 'A task rule was violated.'
        record_failure(env, 'wrong_container')
        record_failure(env, 'premature_button')
        assert failure_message(env) == 'Picked up the wrong container for the current step.'
        clear_failure(env)
        assert env._failure_code is None

    def test_public_info_only_has_reason_for_terminal_simulator_failure(self):
        cases = ['ongoing', 'success', 'timeout', 'fail']
        for status in cases:
            with self.subTest(status=status):
                native = RecordingEnv()
                env = FailAwareWrapper(native)
                env.reset()
                record_failure(native, 'too_many_placements')
                info = {'status': status, 'failure_reason': 'private target=blue_cube_9'}
                result = env._remember((native.observation(), 0, status in ('success', 'fail'), status == 'timeout', info))
                if status == 'fail':
                    assert result[4]['failure_reason'] == 'Completed more pick-and-place cycles than requested.'
                    assert env.step('valid') is result
                else:
                    assert 'failure_reason' not in result[4]
                env.reset()
                assert native._failure_code is None
                assert 'failure_reason' not in env.step('valid')[4]

    def test_recoverable_errors_and_error_penalty_timeout_have_no_failure_reason(self):
        env = FailAwareWrapper(RecordingEnv(limit=200))
        env.reset()
        for status in ('ongoing', 'timeout'):
            info = env.step('raise')[4]
            assert info['status'] == status
            assert 'failure_reason' not in info

    def test_partial_action_error_preserves_a_real_task_failure(self):
        native = RecordingEnv()
        env = FailAwareWrapper(native)
        env.reset()
        record_failure(native, 'wrong_placement')
        _, _, terminated, _, info = env.step('partial_failure')
        assert terminated and info['status'] == 'fail'
        assert info['failure_reason'] == 'Placed the cube outside the required target.'

    def test_native_info_and_dense_step_batches_preserve_the_public_reason(self):
        cases = [False, True]
        for failed in cases:
            with self.subTest(failed=failed):
                native = object.__new__(MemoryTask)
                record_failure(native, 'wrong_object')
                raw = {'success': torch.tensor([False]), 'fail': torch.tensor([failed])}
                with patch.object(BaseEnv, 'get_info', return_value=raw):
                    info = native.get_info()
                batch = batches.to_step_batch([
                    ({}, 0, False, False, {"success": False, "fail": False}),
                    ({}, 0, failed, False, info),
                ])
                info = batches.flatten_info(batch[-1])
                if failed:
                    assert info['failure_reason'] == 'Picked up the wrong object.'
                else:
                    assert 'failure_reason' not in info

    def test_counting_feedback_distinguishes_the_actual_violation(self):
        cases = [
            ('wrong_target_order_or_extra_swing', {'count': 6}, 'Completed more swings than requested.'),
            ('wrong_target_order_or_extra_swing', {'count': 1}, 'Visited the swing targets in the wrong order.'),
            ('stopped_on_wrong_pass_or_off_target', {'visits': 2}, 'Stopped the cube on the wrong pass.'),
            ('stopped_on_wrong_pass_or_off_target', {'visits': 3}, 'Stopped the cube away from the target.'),
            ('too_many_placements', {}, 'Completed more pick-and-place cycles than requested.'),
            ('distractor_in_bin', {}, 'Put an unrequested cube into the bin.'),
        ]
        for reason, fields, expected in cases:
            with self.subTest(reason=reason, fields=fields, expected=expected):
                native = SimpleNamespace(device='cpu', task_list=[{}], num_repeats=3, stop_time=3)
                controller = SimpleNamespace(env=native, ready=False, goal='private goal', failure=reason, success=False, **fields)
                flags = CountingController.flags(controller)
                assert flags['fail'].item()
                assert failure_message(native) == expected

    def test_move_cube_reason_agrees_with_the_existing_failure_rule(self):
        cases = [
            ('hook', {'cube_grasped': True}, 'robot_grasped_cube'),
            ('poke', {'robot_contact_steps': 1}, 'robot_or_gripper_touched_cube'),
            ('poke', {'shaft_contact_steps': 1}, 'illegal_tool_surface_touched_cube'),
            ('scoop', {'illegal_pan_contact_steps': 1}, 'illegal_tool_surface_touched_cube'),
            ('hook', {'cube_lifted': True}, 'cube_lifted_for_non_scoop_method'),
            ('scoop', {'cube_lifted': True}, None),
        ]
        for method, evidence, code in cases:
            with self.subTest(method=method, evidence=evidence, code=code):
                assert move_failure_code(method, evidence) == code
                assert move_failed(method, evidence) == (code is not None)

    def test_pattern_failure_reports_segment_deviation_or_wrong_stop(self):
        from robomme_plus.tasks.imitation.pattern_lock import _TaskScene
        from robomme_plus.tasks.imitation.utils.pattern_dwell import PatternDwell

        for deviation in (True, False):
            with self.subTest(deviation=deviation):
                scorer = PatternDwell([[0, 0], [.09, 0], [.18, 0]], [0, 2])
                for step in range(7):
                    scorer.update(step, [0, 0, .07])
                tip, step = ([0, .03, .07], 7) if deviation else ([.09, 0, .07], 13)
                if not deviation:
                    for prior_step in range(7, step):
                        scorer.update(prior_step, tip)
                native = SimpleNamespace(
                    _pattern_dwell=scorer,
                    _pattern_grid_phase='execution',
                    _limit_diagnostics={'pattern_stops': {}},
                    buttons_grid=[object() for _ in range(3)],
                    elapsed_steps=torch.tensor([step]),
                    use_demonstrationwrapper=True,
                    demonstration_record_traj=False,
                    task_list=[dict(
                        name='endpoint', func=lambda: False,
                        failure_func=lambda: scorer.failure is not None,
                        failure_reason='wrong_endpoint_stop',
                    )],
                )
                with patch('robomme_plus.tasks.imitation.utils.split_stick.tip_position', return_value=tip):
                    result = _TaskScene.evaluate(native)
                self.assertTrue(result['fail'].item())
                self.assertFalse(result['success'].item())
                self.assertEqual(failure_message(native), (
                    'The drawn pattern does not match the demonstration.' if deviation
                    else 'Stopped at the wrong drawing target.'
                ))

