"""Verify physical silhouettes, empty cutouts and seeded episode shape choices."""

import unittest

import numpy as np
import trimesh

from robomme_plus.tasks.reference.utils.repick_shapes import (
    ASSET_DIR,
    SHAPES,
    shape_assets,
    shape_for_seed,
)


def inside(points, polygon):
    result = np.zeros(len(points), dtype=bool)
    x, y = points.T
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[1] - a[1]) < 1e-12:
            continue
        result ^= ((a[1] > y) != (b[1] > y)) & (
            x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
        )
    return result


class RepickShapesTest(unittest.TestCase):
    def test_shape_selection_is_reproducible_and_covers_all_shapes(self):
        selected = [shape_for_seed(seed) for seed in range(600)]
        self.assertEqual(set(selected), set(SHAPES))
        self.assertEqual(selected, [shape_for_seed(seed) for seed in range(600)])
        self.assertTrue(all(selected.count(shape) > 60 for shape in SHAPES))

    def test_collision_pieces_match_silhouettes_and_preserve_empty_cutouts(self):
        points = np.random.default_rng(42).uniform(-1.05, 1.05, (10000, 2))
        for shape, assets in shape_assets().items():
            with self.subTest(shape=shape):
                polygon = np.asarray(assets["outline_xy"])
                full = trimesh.load(ASSET_DIR / assets["mesh"], force="mesh")
                self.assertTrue(full.is_watertight)
                self.assertTrue(np.all(np.abs(full.vertices) <= 1 + 1e-8))
                self.assertGreater(full.volume, 1)
                self.assertTrue(inside(np.array([[0., 0.]]), polygon)[0])
                union = np.zeros(len(points), dtype=bool)
                volume = 0
                for filename in assets["collision_parts"]:
                    part = trimesh.load(ASSET_DIR / filename, force="mesh")
                    self.assertTrue(part.is_watertight)
                    self.assertTrue(part.is_convex)
                    self.assertTrue(np.all(np.abs(part.vertices) <= 1 + 1e-8))
                    np.testing.assert_allclose(part.bounds[:, 2], [-1, 1])
                    outline = np.unique(part.vertices[:, :2], axis=0)
                    offset = outline - outline.mean(axis=0)
                    outline = outline[np.argsort(np.arctan2(offset[:, 1], offset[:, 0]))]
                    union |= inside(points, outline)
                    volume += part.volume
                self.assertAlmostEqual(volume, full.volume, places=6)
                np.testing.assert_array_equal(union, inside(points, polygon))


if __name__ == "__main__":
    unittest.main()
