"""Pixel-area boundaries and camera geometry for the swap visibility gate."""

import unittest

import numpy as np

from robomme_plus.tasks.reference.utils.place_visibility import (
    camera_rays, occlusion_counts, target_silhouette,
)


class SwapVisibilityTest(unittest.TestCase):
    def test_half_is_allowed_but_more_than_half_is_blocked(self):
        target = np.ones((2, 5), dtype=bool)
        robot = np.zeros_like(target)
        robot[0] = True
        self.assertTrue(occlusion_counts(target, robot)['allowed'])
        robot[1, 0] = True
        result = occlusion_counts(target, robot)
        self.assertFalse(result['allowed'])
        self.assertEqual(result['total_target_pixels'], 10)
        self.assertEqual(result['arm_occluded_pixels'], 6)
        self.assertEqual(result['arm_occluded_fraction'], .6)

    def test_robot_pixels_outside_target_do_not_count(self):
        target = np.zeros((6, 6), dtype=bool)
        target[2:4, 2:4] = True
        result = occlusion_counts(target, ~target)
        self.assertTrue(result['allowed'])
        self.assertEqual(result['arm_occluded_pixels'], 0)

    def test_target_outside_camera_does_not_trigger_an_unseen_swap(self):
        result = occlusion_counts(np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool))
        self.assertFalse(result['allowed'])
        self.assertIsNone(result['arm_occluded_fraction'])

    def test_top_down_silhouette_has_full_area_and_moves_with_target(self):
        intrinsic = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.]])
        # Camera at z=1, with its optical axis pointing down.
        extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.]])
        origin, rays = camera_rays((100, 100), intrinsic, extrinsic)
        np.testing.assert_array_equal(origin, [0, 0, 1])
        center = target_silhouette(origin, rays, [0, 0, 0], .1)
        shifted = target_silhouette(origin, rays, [.2, 0, 0], .1)
        self.assertLess(abs(center.sum() - np.pi * 10 ** 2), 8)
        self.assertAlmostEqual(np.argwhere(center)[:, 1].mean(), 49.5)
        self.assertAlmostEqual(np.argwhere(shifted)[:, 1].mean(), 69.5, delta=.2)
        self.assertFalse(np.any(center & shifted))

    def test_cylinder_edge_is_included_at_low_viewing_angle(self):
        origin = np.array([1., 0, 0])
        rays = np.array([[[-1., 0, 0], [-1, .02, 0], [-1, 0, .02]]])
        mask = target_silhouette(origin, rays, [0, 0, 0], .04, thickness=.01)
        np.testing.assert_array_equal(mask, [[True, True, False]])

    def test_target_behind_camera_has_no_area(self):
        origin = np.zeros(3)
        rays = np.array([[[0., 0, 1]]])
        self.assertFalse(target_silhouette(origin, rays, [0, 0, -1], .1).any())


if __name__ == '__main__':
    unittest.main()
