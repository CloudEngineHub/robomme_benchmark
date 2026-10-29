"""Longer PatternLock sticks retain level rules and legacy replay geometry."""
import unittest

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.imitation.utils.geometry import resolve_config
from robomme_plus.tasks.imitation.utils.variants import tracing_stick_fraction


class PatternLengthV5Test(unittest.TestCase):
    def test_new_range_and_legacy_seed_stream(self):
        lengths = []
        for seed in range(1000):
            old = tracing_stick_fraction('PatternLock', seed, stick_change_version=4)
            new = tracing_stick_fraction('PatternLock', seed, stick_change_version=5)
            self.assertGreaterEqual(new, .4)
            self.assertLessEqual(new, .6)
            self.assertAlmostEqual((new - .4) / .2, (old - .25) / .25)
            for version in (1, 2, 3):
                self.assertEqual(old, tracing_stick_fraction(
                    'PatternLock', seed, stick_change_version=version))
            lengths.append(new)
        self.assertLess(min(lengths), .401)
        self.assertGreater(max(lengths), .599)
        self.assertAlmostEqual(np.mean(lengths), .5, delta=.005)

    def test_only_execution_length_changes(self):
        for level in ('easy', 'medium', 'hard'):
            for seed in range(100):
                old = resolve_config('pattern_trace', seed, level, stick_change_version=4)
                new = resolve_config('pattern_trace', seed, level, stick_change_version=5)
                self.assertEqual(new.pop('stick_change_version'), 5)
                self.assertEqual(old.pop('stick_change_version'), 4)
                before = old.pop('execution_stick_fraction')
                after = new.pop('execution_stick_fraction')
                if level == 'easy':
                    self.assertEqual((before, after), (1., 1.))
                else:
                    self.assertTrue(.25 <= before <= .5)
                    self.assertTrue(.4 <= after <= .6)
                self.assertEqual(new, old)

    def test_metadata_roundtrip_and_level_validation(self):
        for level, scale, angle in (('easy', 1., 0.), ('medium', .9, 15.123), ('hard', 1.3, 27.456)):
            record = dict(seed=97, difficulty=level, oracle_policy_step=250,
                          stick_change_version=5, execution_grid_scale=scale,
                          execution_stick_angle_deg=angle)
            row = normalize_episode('PatternLock', record, 0)
            self.assertEqual(normalize_episode('PatternLock', row, 0), row)
            self.assertEqual(row['stick_change_version'], 5)
            self.assertEqual(row['execution_stick_angle_deg'], angle)
            with self.assertRaises(ValueError):
                normalize_episode('RouteStick', record, 0)
        for field, value in (('execution_stick_angle_deg', 25), ('execution_grid_scale', 1.3)):
            record = dict(seed=1, difficulty='medium', oracle_policy_step=250,
                          stick_change_version=5, **{field: value})
            with self.assertRaises(ValueError):
                normalize_episode('PatternLock', record, 0)


if __name__ == '__main__':
    unittest.main()
