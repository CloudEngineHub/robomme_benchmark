"""Geometry and coverage regressions for the broadened initial layouts."""

import unittest

import numpy as np

from robomme_plus.tasks.reference.utils.table_layout import sample_cube_pool
from robomme_plus.tasks.counting.utils.controllers import PROFILES, resolve_counting_config
from robomme_plus.tasks.counting.utils.motion import (
    CurvedRoute, PICK_BUTTON_RADIUS, ROBOT_BASE_XY, SWING_DODGE_REACH_M,
    seeded_counting_layout, seeded_stop_layout,
)
from robomme_plus.tasks.permanence.utils.routes import packed_initial
from robomme_plus.tasks.permanence.utils.sampling import packed_points


class InitialSamplingTest(unittest.TestCase):
    def test_binfill_apparatus_clears_the_entire_translated_belt(self):
        scenes = [seeded_counting_layout('BinFill', seed) for seed in range(128)]
        for scene in scenes:
            belt = np.array(scene['belt_center_xy'])
            objects = [(np.array(scene['bin_xyz'][:2]), np.sqrt(2) * .05),
                       (np.array(scene['button_xy']), PICK_BUTTON_RADIUS)]
            for point, radius in objects:
                gap = np.linalg.norm(np.maximum(np.abs(point - belt) - [.15, .33], 0))
                self.assertGreaterEqual(gap, radius + .018)
                reach = np.linalg.norm(point - ROBOT_BASE_XY)
                self.assertGreaterEqual(reach, .28 + radius)
                self.assertLessEqual(reach, .76 - radius)
            self.assertGreaterEqual(np.linalg.norm(objects[0][0] - objects[1][0]),
                                    objects[0][1] + objects[1][1] + .025)
        self.assertGreater(np.ptp([s['bin_xyz'][1] for s in scenes]), .55)
        self.assertGreater(np.ptp([s['button_xy'][1] for s in scenes]), .55)
        self.assertGreater(np.ptp([s['belt_center_xy'][0] for s in scenes]), .11)
        self.assertGreater(np.ptp([s['bin_yaw_rad'] for s in scenes]), 6)

    def test_swing_keeps_lr_and_full_dodge_area_clearance(self):
        scenes = [seeded_counting_layout('SwingXtimes', seed) for seed in range(128)]
        for scene in scenes:
            self.assertLess(scene['target_origins'][0][1], 0)
            self.assertGreater(scene['target_origins'][1][1], 0)
            for point, radius in [(scene['cube_xyz'][:2], np.sqrt(2)*.02),
                                  (scene['button_xy'], PICK_BUTTON_RADIUS)]:
                for origin in scene['target_origins']:
                    gap = np.linalg.norm(np.maximum(np.abs(np.array(point)-origin[:2])-SWING_DODGE_REACH_M, 0))
                    self.assertGreaterEqual(gap, radius + .04 + .03)
        self.assertGreater(np.ptp([s['cube_xyz'][0] for s in scenes]), .35)
        self.assertGreater(np.ptp([s['cube_xyz'][1] for s in scenes]), .55)
        self.assertGreater(np.ptp([s['cube_yaw_rad'] for s in scenes]), 6)

    def test_stop_buttons_clear_full_routes_on_every_level(self):
        for level in ('easy', 'medium', 'hard'):
            scenes = []
            for seed in range(32):
                config = resolve_counting_config(dict(PROFILES['stopcube_random'],
                    profile='stopcube_random', level=level), seed)
                route = CurvedRoute(config['shape'], config['durations'],
                    axes=config['axes'], rotation_degrees=config['rotation_degrees'],
                    direction_sign=config.get('direction_sign', 1))
                scene = seeded_stop_layout(seed, route, level)
                self.assertEqual(scene, seeded_stop_layout(seed, route, level))
                outline = route.local_points(np.linspace(0, 2, 16384)) + scene['route_center_xy']
                self.assertGreaterEqual(np.linalg.norm(outline-scene['button_xy'], axis=1).min(),
                                        PICK_BUTTON_RADIUS + np.sqrt(2)*.02 + .035)
                scenes.append(scene)
            self.assertGreater(np.ptp([s['button_xy'][1] for s in scenes]), .5)

    def test_highlight_whole_pool_preserves_count_size_gaps_and_yaw(self):
        obstacle = (np.array([-.2, 0]), PICK_BUTTON_RADIUS)
        for seed in range(64):
            points, yaws = sample_cube_pool(seed, 10, .02, .04, [obstacle])
            self.assertEqual(len(points), 10)
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            self.assertGreaterEqual(distances[~np.eye(10, dtype=bool)].min(), np.sqrt(2)*.08)
            self.assertGreaterEqual(np.linalg.norm(points-obstacle[0], axis=1).min(),
                                    np.sqrt(2)*.06 + PICK_BUTTON_RADIUS)
            self.assertEqual(len(np.unique(yaws)), 10)
            again = sample_cube_pool(seed, 10, .02, .04, [obstacle])
            np.testing.assert_array_equal(points, again[0])

    def test_compact_highlight_pool_fits_seventeen_with_full_clearance(self):
        from robomme_plus.tasks.reference.utils.highlight_levels import COMPACT_CUBE_BOUNDS

        bounds = np.asarray(COMPACT_CUBE_BOUNDS)
        obstacle = (np.array([-.2, 0]), np.sqrt(2) * .025 * 1.5)
        points, yaws = sample_cube_pool(0, 17, .02, .04, [obstacle], bounds=bounds)
        self.assertEqual(len(points), 17)
        self.assertTrue(np.all(points >= bounds[:, 0] + np.sqrt(2) * .02))
        self.assertTrue(np.all(points <= bounds[:, 1] - np.sqrt(2) * .02))
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        self.assertGreaterEqual(distances[~np.eye(17, dtype=bool)].min(), np.sqrt(2) * .08)
        self.assertGreaterEqual(np.linalg.norm(points - obstacle[0], axis=1).min(),
                                np.sqrt(2) * .06 + obstacle[1])
        self.assertEqual(len(np.unique(yaws)), 17)
        again = sample_cube_pool(0, 17, .02, .04, [obstacle], bounds=bounds)
        np.testing.assert_array_equal(points, again[0])

    def test_distractors_are_not_selected_as_farthest_points(self):
        distances = []
        for seed in range(128):
            points, _ = packed_points(np.random.default_rng(seed), 1,
                [[-1, 1], [-1, 1]], .02, [[0, 0]], lambda xy: True)
            distances.append(np.linalg.norm(points[0]))
        self.assertLess(np.mean(distances), .9)

    def test_denser_highlight_pool_keeps_cubes_and_button_clear(self):
        from robomme_plus.tasks.reference.utils.highlight_levels import (
            DENSE_CUBE_BOUNDS, DENSE_CUBE_GAP_FACTOR,
        )

        bounds = np.asarray(DENSE_CUBE_BOUNDS)
        radius, gap = np.sqrt(2) * .02, .02 * DENSE_CUBE_GAP_FACTOR
        for seed in range(16):
            button = np.array([-.2, 0]) + np.random.default_rng(seed).uniform(-.5, .5, 2) * [.1, .4]
            button_radius = np.sqrt(2) * .0375
            points, yaws = sample_cube_pool(
                seed, 17, .02, gap, [(button, button_radius)], bounds=bounds
            )
            self.assertEqual(len(points), 17)
            self.assertTrue(np.all(points >= bounds[:, 0] + radius))
            self.assertTrue(np.all(points <= bounds[:, 1] - radius))
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            self.assertGreaterEqual(distances[~np.eye(17, dtype=bool)].min(), 2 * radius + np.sqrt(2) * gap)
            self.assertGreaterEqual(np.linalg.norm(points - button, axis=1).min(),
                                    radius + button_radius + np.sqrt(2) * gap)
            self.assertEqual(len(np.unique(yaws)), 17)

    def test_both_swap_visitors_have_random_initial_positions(self):
        starts = []
        obstacles = [(np.array([-.19, -.1]), .055), (np.array([-.19, .1]), .055)]
        for seed in range(32):
            points, _ = packed_initial(10, .038, seed, .0045, obstacles)
            starts.append(points[:2])
            distance = np.linalg.norm(points[:, None]-points[None, :], axis=2)
            self.assertGreater(distance[~np.eye(10, dtype=bool)].min(), 2*.038+.012+.009)
        for index in (0, 1):
            self.assertEqual(len(np.unique(np.array(starts)[:, index], axis=0)), 32)


if __name__ == '__main__':
    unittest.main()
