"""Cubes use their complete oriented silhouette, not the target disk shape."""

import unittest

import numpy as np

from robomme_plus.tasks.reference.utils.highlight_visibility import MAX_ARM_OCCLUSION, CubeVisibility, cube_silhouette
from robomme_plus.tasks.reference.utils.place_visibility import camera_rays, occlusion_counts


class CubeVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.intrinsic = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.]])
        self.extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.]])
        self.origin, self.rays = camera_rays((100, 100), self.intrinsic, self.extrinsic)

    def test_top_face_and_side_faces_project_as_box(self):
        mask = cube_silhouette(self.origin, self.rays, [0, 0, 0], [1, 0, 0, 0], .1)
        self.assertEqual(mask.sum(), 22**2)
        side = cube_silhouette(np.array([1., 0, 0]),
                               np.array([[[-1., 0, 0], [-1, .09, .09], [-1, .2, 0]]]),
                               [0, 0, 0], [1, 0, 0, 0], .1)
        np.testing.assert_array_equal(side, [[True, True, False]])

    def test_yaw_changes_silhouette_and_center_moves_projection(self):
        normal = cube_silhouette(self.origin, self.rays, [0, 0, 0], [1, 0, 0, 0], .1)
        rotated = cube_silhouette(self.origin, self.rays, [0, 0, 0],
                                  [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)], .1)
        self.assertFalse(np.array_equal(normal, rotated))
        shifted = cube_silhouette(self.origin, self.rays, [.2, 0, 0], [1, 0, 0, 0], .1)
        self.assertGreater(np.argwhere(shifted)[:, 1].mean(), 69)

    def test_parallel_rays_and_box_behind_camera_are_rejected(self):
        rays = np.array([[[0., 0, 1], [0, 0, -1]]])
        np.testing.assert_array_equal(cube_silhouette(np.zeros(3), rays, [0, 0, -1],
                                                     [1, 0, 0, 0], .1), [[False, True]])
        self.assertFalse(cube_silhouette(np.zeros(3), rays, [1, 0, -1], [1, 0, 0, 0], .1).any())

    def test_cropped_measurement_matches_full_projection_and_stricter_threshold(self):
        visibility = CubeVisibility(None, .1)
        visibility.intrinsic, visibility.extrinsic = self.intrinsic, self.extrinsic
        visibility.origin, visibility.rays = self.origin, self.rays
        visibility.robot_mask = np.zeros((100, 100), bool)
        visibility.robot_mask[:, :50] = True
        result = visibility.measure([0, 0, 0], [1, 0, 0, 0])
        self.assertEqual(result['total_target_pixels'], 22**2)
        self.assertEqual(result['arm_occluded_fraction'], .5)
        self.assertFalse(result['allowed'])
        visibility.robot_mask[:, 40:] = False
        self.assertTrue(visibility.measure([0, 0, 0], [1, 0, 0, 0])['allowed'])
        visibility.robot_mask[:, :51] = True
        self.assertFalse(visibility.measure([0, 0, 0], [1, 0, 0, 0])['allowed'])
        self.assertFalse(visibility.measure([20, 0, 0], [1, 0, 0, 0])['allowed'])

    def test_exactly_thirty_percent_is_allowed_but_thirty_one_is_not(self):
        mask = np.ones((10, 10), bool)
        robot = np.zeros_like(mask)
        robot[:3] = True
        self.assertTrue(occlusion_counts(mask, robot, maximum_fraction=MAX_ARM_OCCLUSION)['allowed'])
        robot[3, 0] = True
        self.assertFalse(occlusion_counts(mask, robot, maximum_fraction=MAX_ARM_OCCLUSION)['allowed'])
        self.assertTrue(occlusion_counts(mask, robot)['allowed'])  # VideoPlace stays at 50%.


if __name__ == '__main__':
    unittest.main()
