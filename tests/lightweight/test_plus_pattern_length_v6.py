"""Level-specific stick lengths preserve scoring and earlier recorded scenes."""
import unittest

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.imitation.utils.geometry import resolve_config
from robomme_plus.tasks.imitation.utils.variants import tracing_stick_fraction


class PatternLengthV6Test(unittest.TestCase):
    def test_level_ranges_keep_the_seeded_quantile(self):
        for level, low, high in (('medium', .5, .7), ('hard', .35, .6)):
            values = []
            for seed in range(1000):
                old = tracing_stick_fraction('PatternLock', seed, stick_change_version=5)
                value = tracing_stick_fraction('PatternLock', seed, stick_change_version=6, level=level)
                self.assertTrue(low <= value <= high)
                self.assertAlmostEqual((value - low) / (high - low), (old - .4) / .2)
                values.append(value)
            self.assertLess(min(values), low + .001)
            self.assertGreater(max(values), high - .001)
            self.assertAlmostEqual(np.mean(values), (low + high) / 2, delta=.005)
        for level in (None, '', 'unknown'):
            with self.assertRaises(ValueError):
                tracing_stick_fraction('PatternLock', 1, stick_change_version=6, level=level)

    def test_only_medium_and_hard_execution_lengths_change(self):
        for level in ('easy', 'medium', 'hard'):
            for seed in range(100):
                old = resolve_config('pattern_trace', seed, level, stick_change_version=5)
                new = resolve_config('pattern_trace', seed, level, stick_change_version=6)
                self.assertEqual(old.pop('stick_change_version'), 5)
                self.assertEqual(new.pop('stick_change_version'), 6)
                before, after = old.pop('execution_stick_fraction'), new.pop('execution_stick_fraction')
                if level == 'easy':
                    self.assertEqual((before, after), (1., 1.))
                else:
                    low, high = (.5, .7) if level == 'medium' else (.35, .6)
                    self.assertTrue(low <= after <= high)
                self.assertEqual(new, old)

    def test_metadata_preserves_version_and_level_constraints(self):
        for level, scale, angle in (('easy', 1., 0.), ('medium', 1.1, 19.), ('hard', 1.3, 29.)):
            record = dict(seed=97, difficulty=level, oracle_policy_step=300,
                          stick_change_version=6, execution_grid_scale=scale,
                          execution_stick_angle_deg=angle)
            row = normalize_episode('PatternLock', record, 0)
            self.assertEqual(row['stick_change_version'], 6)
            self.assertEqual(normalize_episode('PatternLock', row, 0), row)
            with self.assertRaises(ValueError):
                normalize_episode('RouteStick', record, 0)
        for field, value in (('execution_stick_angle_deg', 25), ('execution_grid_scale', 1.3)):
            with self.assertRaises(ValueError):
                normalize_episode('PatternLock', dict(seed=1, difficulty='medium',
                    oracle_policy_step=300, stick_change_version=6, **{field: value}), 0)


if __name__ == '__main__':
    unittest.main()
