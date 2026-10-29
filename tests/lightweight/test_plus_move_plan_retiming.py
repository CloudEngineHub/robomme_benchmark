"""Retiming changes physical speed without changing the control clock."""
from types import SimpleNamespace
import unittest

import numpy as np

from robomme_plus.tasks.imitation.move_cube_support.plan_retiming import (
    dense_motion, reference_path, retime_path,
)


class PlanRetimingTests(unittest.TestCase):
    def path(self, count=21):
        return dict(position=np.linspace(0, .1, count)[:, None],
                    velocity=np.full((count, 1), .1))

    def test_half_duration_preserves_endpoints_and_consistent_velocity(self):
        for count in (20, 21):
            original = self.path(count)
            result = retime_path(original)
            scale = (count - 1) / (len(result['position']) - 1)
            np.testing.assert_array_equal(result['position'][[0, -1]], original['position'][[0, -1]])
            np.testing.assert_allclose(result['velocity'], .1 * scale)
            np.testing.assert_allclose(np.diff(result['position'][:, 0]), .1 / (len(result['position']) - 1))
            np.testing.assert_array_equal(original['velocity'], np.full((count, 1), .1))

    def test_contact_dwells_and_tiny_corrections_are_not_shortened(self):
        for path in (dict(position=np.ones((20, 7))), self.path(3)):
            self.assertIs(retime_path(path), path)

    def test_dense_context_is_nested_and_restored_after_failure(self):
        base = SimpleNamespace()
        env = SimpleNamespace(unwrapped=base)
        with self.assertRaises(RuntimeError):
            with dense_motion(env):
                with dense_motion(env, False):
                    self.assertTrue(base._reference_dense_motion)
                    raise RuntimeError('planning failed')
        self.assertFalse(base._reference_dense_motion)

    def test_legacy_profiles_and_ball_cue_keep_original_path(self):
        path = self.path()
        for method, stride in [('poke', 1), ('ball', 2), ('cue', 2)]:
            base = SimpleNamespace(_required_method=method, _reference_plan_stride=stride)
            self.assertIs(reference_path(base, path), path)

    def test_pan_loading_lifting_unloading_stay_dense_but_carry_is_faster(self):
        base = SimpleNamespace(_required_method='scoop', _reference_plan_stride=2)
        path = self.path()
        for stage in ('loading', 'lifting', 'unloading'):
            base._reference_scoop_stage = stage
            self.assertIs(reference_path(base, path), path)
        base._reference_scoop_stage = 'carrying'
        self.assertEqual(len(reference_path(base, path)['position']), 11)


if __name__ == '__main__':
    unittest.main()
