"""Rigid phase changes preserve PatternLock node identities and route geometry."""

import unittest

import numpy as np

from robomme_plus.tasks.imitation.utils.pattern_layout import (
    execution_point, grid_layouts, grid_spacing_scale,
)


class PatternShiftTest(unittest.TestCase):
    def test_seeded_rigid_one_cell_shift_and_rotation(self):
        directions, signs = set(), set()
        for seed in range(100):
            layouts = grid_layouts(seed, 5, .09, [-.13, 0])
            self.assertEqual(layouts, grid_layouts(seed, 5, .09, [-.13, 0]))
            demo, execution = layouts['demonstration'], layouts['execution']
            before, after = np.array(demo['positions_xy_m']), np.array(execution['positions_xy_m'])
            directions.add(execution['direction'])
            signs.add(np.sign(execution['yaw_deg']))
            self.assertEqual(demo['yaw_deg'], 0)
            self.assertTrue(5 <= abs(execution['yaw_deg']) <= 10)
            delta = after.mean(axis=0) - before.mean(axis=0)
            np.testing.assert_allclose(delta, execution['translation_xy_m'], atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(delta), .09)
            self.assertEqual(np.count_nonzero(execution['translation_xy_m']), 1)
            np.testing.assert_allclose(execution_point(before, layouts), after, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(before[:, None]-before, axis=2),
                                       np.linalg.norm(after[:, None]-after, axis=2), atol=1e-12)
            # Node order remains row-major; a grid edge rotates by the declared angle.
            edge = after[5] - after[0]
            self.assertAlmostEqual(np.degrees(np.arctan2(edge[1], edge[0])), execution['yaw_deg'])
        self.assertEqual(directions, {'left', 'right', 'forward', 'backward'})
        self.assertEqual(signs, {-1, 1})

    def test_transform_carries_subcell_start_offset(self):
        layouts = grid_layouts(42, 5, .09, [-.13, 0])
        before = np.array(layouts['demonstration']['positions_xy_m'])[12]
        after = np.array(layouts['execution']['positions_xy_m'])[12]
        offset = np.array([.002, -.003])
        moved = execution_point(before + offset, layouts)
        self.assertAlmostEqual(np.linalg.norm(moved - after), np.linalg.norm(offset))
        self.assertFalse(np.allclose(moved, before + offset))

    def test_sampling_does_not_change_global_random_state(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        grid_layouts(42, 5, .09, [-.13, 0])
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_four_spacing_choices_are_seeded_and_independent(self):
        scales = [grid_spacing_scale(seed) for seed in range(200)]
        self.assertEqual(set(scales), {.8, 1.0, 1.2, 1.4})
        for scale in (.8, 1.0, 1.2, 1.4):
            self.assertGreater(scales.count(scale), 30)
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        for seed, scale in enumerate(scales):
            self.assertEqual(scale, grid_spacing_scale(seed))
            original = grid_layouts(seed, 5, .09, [-.13, 0])
            changed = grid_layouts(seed, 5, .09, [-.13, 0], execution_scale=scale)
            for phase in original:
                for key in ('direction', 'yaw_deg', 'translation_xy_m', 'center_xy_m'):
                    self.assertEqual(original[phase][key], changed[phase][key])
            self.assertEqual(original['demonstration']['positions_xy_m'], changed['demonstration']['positions_xy_m'])
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_scaled_grid_and_subcell_offsets_follow_same_transform(self):
        for scale in (.8, 1.0, 1.2, 1.4):
            for seed in range(10):
                layouts = grid_layouts(seed, 5, .09, [-.13, 0], execution_scale=scale)
                demo, execution = layouts['demonstration'], layouts['execution']
                before, after = np.array(demo['positions_xy_m']), np.array(execution['positions_xy_m'])
                self.assertAlmostEqual(demo['spacing_m'], .09)
                self.assertAlmostEqual(execution['spacing_m'], .09 * scale)
                np.testing.assert_allclose(execution_point(before, layouts), after, atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(after[:, None] - after, axis=2),
                                           np.linalg.norm(before[:, None] - before, axis=2) * scale, atol=1e-12)
                offset = np.array([.002, -.003])
                self.assertAlmostEqual(np.linalg.norm(execution_point(before[12] + offset, layouts) - after[12]),
                                       np.linalg.norm(offset) * scale)

    def test_metadata_validates_and_preserves_chosen_spacing(self):
        from robomme_plus.evaluation.metadata import normalize_episode
        record = dict(seed=85, difficulty='medium', oracle_policy_step=180)
        self.assertNotIn('execution_grid_scale', normalize_episode('PatternLock', record, 0))
        for scale in (.8, .9, 1.0, 1.1, 1.2, 1.3, 1.4):
            self.assertEqual(normalize_episode('PatternLock', dict(record, execution_grid_scale=scale), 0)['execution_grid_scale'], scale)
        for invalid in (True, False, None, '0.8', .79, 1.01, 1.31, 1.41, float('nan'), float('inf')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_episode('PatternLock', dict(record, execution_grid_scale=invalid), 0)


if __name__ == '__main__':
    unittest.main()
