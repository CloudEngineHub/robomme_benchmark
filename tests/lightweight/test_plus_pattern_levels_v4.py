"""Level-specific changes preserve demo geometry and legacy episode replay."""
import unittest
import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.imitation.utils.geometry import resolve_config
from robomme_plus.tasks.imitation.utils.pattern_layout import execution_point, grid_layouts, grid_spacing_scale
from robomme_plus.tasks.imitation.utils.variants import pattern_level_tilt, tracing_stick_tilt


class PatternLevelsTest(unittest.TestCase):
    def test_continuous_tilt_ranges_and_full_circle_direction(self):
        for level, low, high in (('medium', 10, 20), ('hard', 20, 30)):
            values = np.array([pattern_level_tilt(seed, level) for seed in range(1000)])
            self.assertTrue(np.all((low <= values[:, 0]) & (values[:, 0] <= high)))
            self.assertGreater(len(set(values[:, 0])), 990)
            self.assertLess(abs(values[:, 0].mean() - (low + high) / 2), .3)
            self.assertLess(values[:, 1].min(), -179)
            self.assertGreater(values[:, 1].max(), 179)
            for seed in range(20):
                self.assertEqual(pattern_level_tilt(seed, level)[1], tracing_stick_tilt(seed)[1])

    def test_profiles_spacing_and_unchanged_demonstration(self):
        for level, count_range, scales in (('easy', (4, 5), {1.}),
                                          ('medium', (6, 8), {.9, 1.1}),
                                          ('hard', (9, 12), {.9, 1.1, 1.2, 1.3})):
            seen = set()
            for seed in range(100):
                config = resolve_config('pattern_trace', seed, level, stick_change_version=4)
                self.assertFalse(config['board'])
                self.assertTrue(count_range[0] <= config['path_node_count'] <= count_range[1])
                self.assertEqual(config['drawing_max_angle_deg'], 8)
                self.assertEqual(config['drawing_height_m'], .08)
                scale = grid_spacing_scale(seed, level)
                seen.add(scale)
                layouts = grid_layouts(seed, 5, .09, [-.13, 0], rotate=config['rotate_grid'],
                                       execution_scale=scale, shift_grid=config['shift_grid'],
                                       shift_distance_m=config.get('grid_shift_m'),
                                       rotation_range_deg=config.get('grid_rotation_range_deg', (5., 10.)))
                self.assertEqual(layouts['demonstration']['spacing_m'], .09)
                self.assertAlmostEqual(layouts['execution']['spacing_m'], .09 * scale)
                if level == 'easy':
                    self.assertEqual(config['execution_stick_fraction'], 1)
                    self.assertIsNone(config['execution_stick_tilt_deg'])
                    self.assertEqual(layouts['execution']['positions_xy_m'], layouts['demonstration']['positions_xy_m'])
                else:
                    self.assertTrue(.25 <= config['execution_stick_fraction'] <= .5)
                    demo, execution = layouts['demonstration'], layouts['execution']
                    expected_shift = .15 if level == 'hard' else .09
                    self.assertAlmostEqual(np.linalg.norm(execution['translation_xy_m']), expected_shift)
                    low, high = (10, 15) if level == 'hard' else (5, 10)
                    self.assertTrue(low <= abs(execution['yaw_deg']) <= high)
                    np.testing.assert_allclose(np.array(execution['center_xy_m']) - demo['center_xy_m'],
                                               execution['translation_xy_m'], atol=1e-12)
                    np.testing.assert_allclose(execution_point(demo['positions_xy_m'], layouts),
                                               execution['positions_xy_m'], atol=1e-12)
                    baseline = grid_layouts(seed, 5, .09, [-.13, 0])
                    self.assertEqual(demo['positions_xy_m'], baseline['demonstration']['positions_xy_m'])
            self.assertEqual(seen, scales)

    def test_record_roundtrip_and_cross_level_rejection(self):
        for level, scale, angle in (('easy', 1., 0.), ('medium', .9, 15.123), ('hard', 1.3, 27.456)):
            record = dict(seed=97, difficulty=level, oracle_policy_step=250, stick_change_version=4,
                          execution_grid_scale=scale, execution_stick_angle_deg=angle)
            row = normalize_episode('PatternLock', record, 0)
            self.assertEqual(normalize_episode('PatternLock', row, 0), row)
            config = resolve_config('pattern_trace', 97, level, stick_change_version=4,
                                    execution_stick_angle_deg=row['execution_stick_angle_deg'])
            self.assertEqual(config['execution_stick_angle_deg'], angle)
            for bad in (True, None, float('nan'), 31, -1):
                with self.assertRaises(ValueError):
                    normalize_episode('PatternLock', dict(record, execution_stick_angle_deg=bad), 0)
        record = dict(seed=1, difficulty='medium', oracle_policy_step=250, stick_change_version=4)
        for field, value in (('execution_stick_angle_deg', 25), ('execution_grid_scale', 1.3)):
            with self.assertRaises(ValueError):
                normalize_episode('PatternLock', dict(record, **{field: value}), 0)
        with self.assertRaises(ValueError):
            normalize_episode('RouteStick', record, 0)


if __name__ == '__main__':
    unittest.main()
