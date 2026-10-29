"""Private layout searches stop without changing successful recorded phases."""
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from robomme_plus.tasks.imitation.move_cube_support import reference, search_budget
from robomme_plus.tasks.imitation.move_cube_support.search_budget import (
    LayoutSearchExhausted, ReferenceSearchBudget,
)


class LayoutSearchBudgetTests(unittest.TestCase):
    def test_trial_limit_allows_last_trial_but_not_an_extra_one(self):
        budget = ReferenceSearchBudget(max_trials=2)
        budget.start_trial()
        budget.start_trial()
        budget.check_time()  # The last permitted trial can finish.
        with self.assertRaisesRegex(LayoutSearchExhausted, '2 robot trials'):
            budget.start_trial()
        self.assertEqual(budget.trials, 2)

    def test_elapsed_limit_is_shared_by_all_trials(self):
        budget = ReferenceSearchBudget(max_seconds=10)
        budget.started = 100
        with patch.object(search_budget.time, 'monotonic', return_value=109):
            budget.start_trial()
        with patch.object(search_budget.time, 'monotonic', return_value=110):
            with self.assertRaisesRegex(LayoutSearchExhausted, '10.0s'):
                budget.start_trial()
        self.assertEqual(budget.trials, 1)

    def test_invalid_limits_fail_before_simulation(self):
        for trials in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ReferenceSearchBudget(max_trials=trials)
        for seconds in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                ReferenceSearchBudget(max_seconds=seconds)

    def environment(self):
        layout = SimpleNamespace(ramp_height=.12, pan_face='left_side')
        return SimpleNamespace(layout=layout, _layouts={'execution': layout},
                               step=Mock(), close=Mock(), _failed=Mock(return_value=False))

    def test_search_counts_refinements_and_closes_at_limit(self):
        env = self.environment()
        tried = []
        def solve(base, planner, parameters):
            tried.append(parameters)
            base._reference_scoop_stage = 'carrying'
            base._reference_pan_landing = dict(legal=True, along_error_m=.02)
            raise RuntimeError('Landing missed')
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'reset_reference'), \
             patch.object(reference, 'planner_for', return_value=Mock()), \
             patch.object(reference, 'solve', side_effect=solve):
            with self.assertRaisesRegex(LayoutSearchExhausted, '3 robot trials'):
                reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution', max_search_trials=3)
        self.assertEqual(len(tried), 3)
        self.assertNotEqual(tried[0]['scoop_options']['drop_offset'], tried[1]['scoop_options']['drop_offset'])
        env.close.assert_called_once()

    def test_success_on_final_trial_is_retained(self):
        env = self.environment()
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'reset_reference'), \
             patch.object(reference, 'planner_for', return_value=Mock()), \
             patch.object(reference, 'validate_reference_loading_surface'), \
             patch.object(reference, 'solve', side_effect=[RuntimeError('Approach rejected'), {'success': True}]) as solve:
            result = reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution', max_search_trials=2)
        self.assertEqual(solve.call_count, 2)
        self.assertEqual(result['flip'], 1)
        env.close.assert_called_once()

    def test_elapsed_budget_interrupts_ball_probes_before_robot_trials(self):
        env = self.environment()
        budget = ReferenceSearchBudget()
        budget.started = 100
        def probes(base):
            with patch.object(search_budget.time, 'monotonic', return_value=281):
                base.step('probe action')
        config = dict(method='ball', seed=9, layouts={'execution': {}},
                      available_methods={'demonstration': ['ball', 'hook', 'poke'],
                                         'execution': ['ball', 'hook', 'poke']})
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'ReferenceSearchBudget', return_value=budget), \
             patch.object(search_budget.time, 'monotonic', return_value=100), \
             patch.object(reference, 'ball_candidates', side_effect=probes), \
             patch.object(reference, 'solve') as solve:
            with self.assertRaisesRegex(LayoutSearchExhausted, '181.0s'):
                reference.find_reference(config, 'execution')
        solve.assert_not_called()
        self.assertEqual(budget.trials, 0)
        env.close.assert_called_once()

    def test_elapsed_budget_is_not_swallowed_as_another_robot_retry(self):
        env = self.environment()
        with patch('robomme_plus.tasks.imitation.move_cube.MoveCube', return_value=env), \
             patch.object(reference, 'reset_reference'), \
             patch.object(reference, 'planner_for', return_value=Mock()), \
             patch.object(reference, 'solve', side_effect=LayoutSearchExhausted('resample layout')) as solve:
            with self.assertRaises(LayoutSearchExhausted):
                reference.find_reference({'method': 'scoop', 'seed': 9}, 'execution')
        solve.assert_called_once()
        env.close.assert_called_once()



if __name__ == '__main__':
    unittest.main()
