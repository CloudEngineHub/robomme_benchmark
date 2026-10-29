"""Similar cube shades vary by seed without making color a goal cue."""

import colorsys
import unittest

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.reference.utils.highlight_levels import sample_similar_colors


class HighlightColorsTest(unittest.TestCase):
    def test_reproducible_balanced_families_with_subtle_distinct_shades(self):
        for seed in range(30):
            colors, palette = sample_similar_colors(seed, 17)
            self.assertEqual((colors, palette), sample_similar_colors(seed, 17))
            self.assertEqual(len(colors), 17)
            self.assertEqual(len(set(map(tuple, colors))), 17)
            count = palette['family_count']
            self.assertIn(count, (2, 3))
            ids = np.array(palette['family_ids'])
            sizes = np.bincount(ids)
            self.assertLessEqual(sizes.max()-sizes.min(), 1)
            hsv = np.array([colorsys.rgb_to_hsv(*rgb[:3]) for rgb in colors])
            for family in range(count):
                members = hsv[ids == family]
                hue_error = (members[:, 0]-palette['family_hues'][family]+.5) % 1-.5
                self.assertLessEqual(abs(hue_error).max(), .012+1e-10)
                self.assertLessEqual(np.ptp(members[:, 1]), .08+1e-10)
                self.assertLessEqual(np.ptp(members[:, 2]), .07+1e-10)
            self.assertTrue((hsv[:, 2] < .9).all())

    def test_palette_and_assignment_vary_between_episodes(self):
        palettes = [sample_similar_colors(seed, 12)[1] for seed in range(100)]
        self.assertEqual({p['family_count'] for p in palettes}, {2, 3})
        self.assertGreater(np.ptp([p['family_hues'][0] for p in palettes]), .9)
        self.assertGreater(len({tuple(p['family_ids']) for p in palettes}), 80)

    def test_recorded_episodes_keep_independent_colors_unless_explicitly_updated(self):
        record = dict(seed=7, difficulty='hard', oracle_policy_step=1000)
        self.assertEqual(normalize_episode('PickHighlight', record, 0)['color_mode'], 'independent')
        updated = normalize_episode('PickHighlight', {**record, 'color_mode': 'similar'}, 0)
        self.assertEqual(updated['color_mode'], 'similar')
        self.assertEqual(normalize_episode('PickHighlight', updated, 0), updated)
        for invalid in ('unknown', True, None, []):
            with self.assertRaisesRegex(ValueError, 'color_mode'):
                normalize_episode('PickHighlight', {**record, 'color_mode': invalid}, 0)


if __name__ == '__main__':
    unittest.main()
