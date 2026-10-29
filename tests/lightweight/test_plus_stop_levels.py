"""StopCube pass counts and seeded speed choices by difficulty."""

import unittest

from robomme_plus.tasks.counting.utils.controllers import PROFILES, resolve_counting_config
from robomme_plus.tasks.counting.utils.variations import (
    goal_counts,
    sample_variation,
)


class StopCubeLevelsTest(unittest.TestCase):
    def test_counts_and_speed_profiles_cover_the_requested_choices(self):
        for level, low, high, speeds in (
            ('easy', 3, 8, {'medium'}),
            ('medium', 5, 10, {'high', 'ultra_high'}),
            ('hard', 5, 10, {'mixed_loops', 'mixed_halves'}),
        ):
            self.assertEqual(goal_counts('StopCube', level), tuple(range(low, high + 1)))
            cases = [sample_variation('StopCube', seed, level) for seed in range(1000)]
            self.assertEqual({case['count'] for case in cases}, set(range(low, high + 1)))
            self.assertEqual({case['speed'] for case in cases}, speeds)
            self.assertEqual(cases[53], sample_variation('StopCube', 53, level))
            if level != 'easy':
                choice = 'high' if level == 'medium' else 'mixed_loops'
                choice_count = sum(case['speed'] == choice for case in cases)
                self.assertGreater(choice_count, 400)
                self.assertLess(choice_count, 600)


    def test_hard_runtime_keeps_mixing_speeds_inside_the_episode(self):
        seen = set()
        for seed in range(60):
            config = resolve_counting_config(
                dict(PROFILES['stopcube_random'], profile='stopcube_random', level='hard'),
                seed,
            )
            self.assertIn(config['count'], range(5, 11))
            seen.add(config['speed'])
            if config['speed'] == 'mixed_halves':
                self.assertTrue(all(len(d) == 2 and d[0] < d[1] for d in config['durations']))
            else:
                self.assertEqual(config['speed'], 'mixed_loops')
                self.assertTrue(all(config['durations'][i] > config['durations'][i + 1] for i in (0, 2, 4)))
        self.assertEqual(seen, {'mixed_loops', 'mixed_halves'})


if __name__ == '__main__':
    unittest.main()
