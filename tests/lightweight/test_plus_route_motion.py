"""Hard boards intercept node arrivals; earlier motion records remain replayable."""
import unittest

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.imitation.utils.geometry import resolve_config
from robomme_plus.tasks.imitation.utils.route_layout import default_route_layout, execution_points
from robomme_plus.tasks.imitation.utils.route_motion import (
    board_swing, layout_swing, resolve_board_motion, rotate_points,
)


class RouteMotionTest(unittest.TestCase):
    def test_only_hard_defaults_to_motion_and_every_level_is_visible(self):
        for level in ("easy", "medium", "hard"):
            self.assertFalse(resolve_config("route_trace", 0, level)["board"])
            for seed in range(20):
                layout = default_route_layout(seed, level)
                row = normalize_episode("RouteStick", dict(seed=seed, difficulty=level,
                                        oracle_policy_step=100, execution_route_layout=layout), 0)
                self.assertEqual(normalize_episode("RouteStick", row, 0), row)
                self.assertEqual("board_motion" in layout, level == "hard")
                if "board_motion" not in layout:
                    continue
                self.assertEqual(level, "hard")
                motion = layout["board_motion"]
                self.assertEqual(motion["phases"], ["execution"])
                self.assertEqual(motion["mode"], "arrival_boards")
                self.assertEqual(motion["stroke_m"], .07)
                self.assertEqual(motion["move_s"], 2.)
                self.assertEqual(motion["hold_s"], 1.5)
                self.assertEqual(motion["return_s"], 2.)
                self.assertEqual(motion["arrival_radius_m"], .03)
                self.assertEqual(len(motion["enabled_boards"]), 4)
                self.assertTrue(all(type(value) is bool for value in motion["enabled_boards"]))

    def test_every_hard_episode_samples_independent_board_masks_without_changing_geometry(self):
        masks = []
        np.random.seed(123)
        expected_global_draws = np.random.random(5)
        np.random.seed(123)
        for seed in range(2000):
            layout = default_route_layout(seed, "hard")
            self.assertEqual(layout, default_route_layout(seed, "hard"))
            geometry = {key: value for key, value in layout.items() if key != "board_motion"}
            self.assertEqual(geometry, default_route_layout(seed, "medium"))
            masks.append(layout["board_motion"]["enabled_boards"])
            # Formerly enabled seeds retain their original per-board choices.
            old_rng = np.random.default_rng([seed, 618011])
            if old_rng.random() < .5:
                self.assertEqual(masks[-1], (old_rng.random(4) < .5).tolist())
        np.testing.assert_array_equal(np.random.random(5), expected_global_draws)
        self.assertEqual(len(masks), 2000)
        rates = np.mean(masks, axis=0)
        self.assertTrue(np.all((.45 < rates) & (rates < .55)), rates)
        # Independent draws allow all-stationary and all-moving episodes too.
        self.assertEqual(len({tuple(mask) for mask in masks}), 16)
        self.assertTrue(.45 < np.sum(masks) / (2000 * 4) < .55)

    def test_swing_reverses_at_both_bounds_and_returns_without_drift(self):
        rest = np.array([[-.1, .08 * (i - 4)] for i in range(9)])
        pivot = rest[4]
        distances = np.linalg.norm(rest[:, None] - rest[None, :], axis=-1)
        for seed in range(10):
            motion = resolve_board_motion(seed, {"mode": "rigid_layout"})
            for t, factor in ((0., 0.), (1., 1.), (2., 0.), (3., -1.), (4., 0.), (400., 0.)):
                points, angle = layout_swing(rest, pivot, motion, t)
                self.assertAlmostEqual(np.degrees(angle), motion["direction_sign"] * 15 * factor)
                np.testing.assert_allclose(points[4], pivot, atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(points[:, None] - points[None, :], axis=-1),
                                           distances, atol=1e-12)
                np.testing.assert_allclose(rotate_points(points, pivot, -angle), rest, atol=1e-12)
                if factor == 0:
                    np.testing.assert_allclose(points, rest, atol=1e-12)

    def test_layout_transform_carries_the_whole_swing_to_execution(self):
        targets = np.array([[-.1, .08 * (i - 4)] for i in range(9)])
        rest = targets
        for seed in range(10):
            layout = default_route_layout(seed, "hard")
            motion = resolve_board_motion(seed, {"mode": "rigid_layout"})
            pivot = targets[motion["pivot_target_index"]]
            center = targets.mean(axis=0)
            for time in np.linspace(0, 4, 21):
                demo, angles = layout_swing(rest, pivot, motion, time)
                execution, moved_angles = layout_swing(execution_points(rest, center, layout),
                                                      execution_points(pivot, center, layout), motion, time)
                np.testing.assert_allclose(execution, execution_points(demo, center, layout), atol=1e-12)
                np.testing.assert_array_equal(moved_angles, angles)

    def test_motion_parameters_are_replayable_and_reject_invalid_pivots(self):
        for seed in range(20):
            motion = resolve_board_motion(seed, {})
            self.assertEqual(resolve_board_motion(seed, motion), motion)
            execution = resolve_board_motion(seed, {"phases": ["execution"]})
            self.assertEqual(execution["direction_signs"], motion["direction_signs"])
        for bad in ({"amplitude_deg": 0}, {"amplitude_deg": 31}, {"amplitude_deg": True},
                    {"period_s": 0}, {"period_s": float("nan")}, {"phases": []},
                    {"phases": ["execution", "execution"]},
                    {"pivot_target_indices": [0, 2, 4, 8, 6]},
                    {"pivot_target_indices": [0, 2, 4, 4]},
                    {"direction_signs": [1, 1, 1, True]}, {"other": 1},
                    {"pivot_target_index": 3}, {"pivot_target_index": True},
                    {"direction_sign": 0}, {"direction_sign": True},
                    {"mode": "rigid_layout", "pivot_target_indices": [0, 2, 4, 6]},
                    {"mode": "independent_boards", "direction_sign": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_board_motion(0, bad)

    def test_saved_independent_prototype_keeps_original_behavior(self):
        saved = dict(amplitude_deg=15., period_s=4., phases=["demonstration", "execution"],
                     pivot_target_indices=[0, 2, 4, 6], direction_signs=[1, -1, 1, -1])
        motion = resolve_board_motion(0, saved)
        self.assertEqual(motion["mode"], "independent_boards")
        rest = np.array([[0., .08 * i] for i in (1, 3, 5, 7)])
        pivots = rest - [0., .08]
        points, angles = board_swing(rest, pivots, motion, 1.)
        np.testing.assert_allclose(np.degrees(angles), [15, -15, 15, -15])
        np.testing.assert_allclose(np.linalg.norm(points-pivots, axis=1), .08)


if __name__ == "__main__":
    unittest.main()
