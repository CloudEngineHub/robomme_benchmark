"""Execution-only RouteStick transforms must preserve ordered target identity."""
import unittest

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.imitation.utils.geometry import resolve_config
from robomme_plus.tasks.imitation.utils.route_layout import (
    default_route_layout, disable_board_collisions, execution_points, resolve_route_layout,
)


class RouteLayoutTest(unittest.TestCase):
    def test_current_levels_preserve_easy_and_transform_medium_hard(self):
        points = np.array([[-.1, .08 * (index - 4)] for index in range(9)])
        center = points.mean(axis=0)
        scales, axes, signs = set(), set(), set()
        for seed in range(200):
            easy = default_route_layout(seed, "easy")
            np.testing.assert_allclose(execution_points(points, center, easy), points, atol=1e-15)
            for level in ("easy", "medium", "hard"):
                layout = default_route_layout(seed, level)
                self.assertEqual(layout, default_route_layout(seed, level))
                self.assertEqual(layout, resolve_route_layout(seed, layout))
                self.assertEqual(layout["board_height_scale"], 1.3)
                self.assertTrue(layout["board_collision"])
                self.assertFalse(layout["board_contact_fails"])
                row = normalize_episode("RouteStick", dict(seed=seed, difficulty=level,
                                        oracle_policy_step=100, execution_route_layout=layout), 0)
                self.assertEqual(normalize_episode("RouteStick", row, 0), row)
                if level == "easy":
                    continue
                self.assertTrue(.9 <= layout["spacing_scale"] <= 1)
                self.assertAlmostEqual(np.linalg.norm(layout["translation_xy_m"]), .05)
                self.assertTrue(5 <= abs(layout["yaw_deg"]) <= 10)
                moved = execution_points(points, center, layout)
                np.testing.assert_allclose(moved.mean(axis=0) - center, layout["translation_xy_m"], atol=1e-15)
                np.testing.assert_allclose(np.linalg.norm(np.diff(moved, axis=0), axis=1),
                                           .08 * layout["spacing_scale"])
                scales.add(layout["spacing_scale"])
                axes.add(tuple(layout["translation_xy_m"]))
                signs.add(np.sign(layout["yaw_deg"]))
        self.assertEqual(len(scales), 200)
        self.assertLess(min(scales), .902)
        self.assertGreater(max(scales), .998)
        self.assertEqual(len(axes), 4)
        self.assertEqual(signs, {-1, 1})

    def test_continuous_spacing_is_versioned_and_independent_of_other_draws(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        for seed in range(10):
            baseline = default_route_layout(seed, "medium")
            for scale in (.9, .925, .975, 1.):
                layout = resolve_route_layout(seed, {"version": 3, "spacing_scale": scale})
                self.assertEqual(layout, {**baseline, "spacing_scale": scale})
                for version in (1, 2):
                    if scale not in (.9, 1.):
                        with self.assertRaises(ValueError):
                            resolve_route_layout(seed, {"version": version, "spacing_scale": scale})
        np.testing.assert_array_equal(np.random.random(5), expected)
        for scale in (.899, 1.001, True, "0.95", None, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                resolve_route_layout(0, {"version": 3, "spacing_scale": scale})
        with self.assertRaises(ValueError):
            default_route_layout(0, "unknown")

    def test_board_touch_only_fails_when_contact_failure_rule_is_enabled(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch
        from robomme_plus.tasks.imitation.utils.controllers import setup_occlusion
        from robomme_plus.tasks.imitation.utils.drawing import Drawing

        class Scene:
            task_name = "RouteStick"
            elapsed_steps = torch.tensor([10])
            timestep = 1
            task_list = [{"demonstration": False}]

            def _load_scene(self): pass
            def _initialize_episode(self): pass
            def evaluate(self, solve_complete_eval=False): pass
            def _after_simulation_step(self):
                self.physics_callbacks += 1

        setup_occlusion(Scene)
        for fails in (False, True):
            env = Scene()
            env.physics_callbacks = 0
            env._imitation_config = dict(board_collision=True, obstacle_contact_fails=fails)
            env._stick_short = True
            env._drawing = Drawing()
            env._drawing.begin("execution", 0)
            env.cubes_on_targets = [SimpleNamespace(name="board")]
            env.agent = SimpleNamespace(robot=SimpleNamespace(
                get_links=lambda: [SimpleNamespace(name="panda_hand")]))
            env.scene = SimpleNamespace(get_pairwise_contact_impulses=Mock(return_value=torch.tensor([1.])))
            env._after_simulation_step()
            self.assertEqual(env.physics_callbacks, 1)
            if fails:
                self.assertEqual(env._drawing.failure["reason"], "obstacle_contact")
            else:
                self.assertIsNone(env._drawing.failure)

    def test_contact_failure_can_be_disabled_with_physics_explicitly_enabled(self):
        for physical in (True, False):
            for fails in (True, False):
                layout = dict(version=2, spacing_scale=1., board_height_scale=1.3,
                              translation_xy_m=[0., 0.], yaw_deg=0.,
                              board_collision=physical, board_contact_fails=fails)
                record = dict(seed=0, difficulty="easy", oracle_policy_step=100,
                              execution_route_layout=layout)
                normalized = normalize_episode("RouteStick", record, 0)
                self.assertEqual(normalized["execution_route_layout"], layout)
                self.assertEqual(normalized, normalize_episode("RouteStick", normalized, 0))
        for invalid in (0, 1, "false", None):
            with self.assertRaises(ValueError):
                resolve_route_layout(0, dict(board_contact_fails=invalid))

    def test_board_collision_toggle_is_explicit_and_survives_metadata(self):
        self.assertNotIn("board_collision", resolve_route_layout(0, {}))
        for enabled in (False, True):
            record = dict(seed=0, difficulty="easy", oracle_policy_step=100,
                          execution_route_layout=dict(version=2, board_height_scale=1.3,
                                                      board_collision=enabled))
            normalized = normalize_episode("RouteStick", record, 0)
            self.assertIs(normalized["execution_route_layout"]["board_collision"], enabled)
            self.assertEqual(normalized, normalize_episode("RouteStick", normalized, 0))
        for invalid in (0, 1, "false", None):
            with self.assertRaises(ValueError):
                resolve_route_layout(0, dict(board_collision=invalid))

    def test_collision_toggle_disables_all_shapes_without_removing_boards(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        shapes = [Mock() for _ in range(4)]
        boards = [SimpleNamespace(_bodies=[SimpleNamespace(get_collision_shapes=lambda: shapes[:2])]),
                  SimpleNamespace(_bodies=[SimpleNamespace(get_collision_shapes=lambda: shapes[2:])])]
        disable_board_collisions(boards)
        self.assertEqual(len(boards), 2)
        for shape in shapes:
            shape.set_collision_groups.assert_called_once_with([0, 0, 0, 0])

    def test_seeded_transform_preserves_order_and_changes_intervals(self):
        points = np.array([[-.1, .08 * (index - 4)] for index in range(9)])
        center = points.mean(axis=0)
        for seed in range(20):
            layout = resolve_route_layout(seed, {})
            self.assertEqual(layout, resolve_route_layout(seed, layout))
            self.assertEqual(layout, resolve_route_layout(seed, {}))
            self.assertAlmostEqual(np.linalg.norm(layout["translation_xy_m"]), .05)
            self.assertTrue(5 <= abs(layout["yaw_deg"]) <= 10)
            self.assertTrue(1 <= layout["board_height_scale"] <= 1.5)
            for scale in (.8, .9, 1., 1.1, 1.2):
                chosen = resolve_route_layout(seed, {**layout, "spacing_scale": scale})
                moved = execution_points(points, center, chosen)
                np.testing.assert_allclose(np.linalg.norm(np.diff(moved, axis=0), axis=1), .08 * scale)
                np.testing.assert_allclose(moved.mean(axis=0) - center, chosen["translation_xy_m"], atol=1e-12)
                # Applying the absolute transform twice from the same demo
                # does not compound scale, translation, or rotation on reset.
                np.testing.assert_array_equal(moved, execution_points(points, center, chosen))

    def test_layout_validation_rejects_ambiguous_or_invalid_geometry(self):
        for values in ([], {"unknown": 1}, {"version": True}, {"spacing_scale": 1.01},
                       {"spacing_scale": True}, {"board_height_scale": .9},
                       {"board_height_scale": 1.51}, {"yaw_deg": 0},
                       {"translation_xy_m": [.05, .05]}, {"translation_xy_m": [False, .05]}):
            with self.assertRaises(ValueError):
                resolve_route_layout(0, values)

    def test_straight_stick_and_layout_survive_metadata_roundtrip(self):
        record = dict(seed=3, difficulty="medium", oracle_policy_step=100,
                      stick_change_version=1,
                      execution_route_layout={"spacing_scale": .9, "board_height_scale": 1.25})
        normalized = normalize_episode("RouteStick", record, 0)
        self.assertEqual(normalized, normalize_episode("RouteStick", normalized, 0))
        self.assertNotIn("execution_stick_angle_deg", normalized)
        self.assertEqual(normalized["execution_route_layout"]["board_height_scale"], 1.25)

    def test_fixed_height_layout_can_replay_original_spacing_and_twenty_cm_boards(self):
        parameters = dict(version=2, spacing_scale=1., board_height_scale=2.)
        layout = resolve_route_layout(3, parameters)
        self.assertEqual(layout, resolve_route_layout(3, layout))
        self.assertEqual(layout["board_height_scale"], 2.)
        record = dict(seed=3, difficulty="medium", oracle_policy_step=100,
                      execution_route_layout=layout)
        normalized = normalize_episode("RouteStick", record, 0)
        self.assertEqual(normalized["execution_route_layout"], layout)
        for values in (dict(version=2, board_height_scale=2.01),
                       dict(version=1, board_height_scale=2.), dict(version=4)):
            with self.assertRaises(ValueError):
                resolve_route_layout(3, values)

    def test_taller_boards_can_keep_the_original_layout(self):
        parameters = dict(version=2, spacing_scale=1., board_height_scale=1.5,
                          translation_xy_m=[0., 0.], yaw_deg=0.)
        points = np.array([[-.1, .08 * (index - 4)] for index in range(9)])
        layout = resolve_route_layout(3, parameters)
        np.testing.assert_allclose(execution_points(points, points.mean(axis=0), layout), points, atol=1e-15)
        record = dict(seed=3, difficulty="medium", oracle_policy_step=100,
                      execution_route_layout=layout)
        self.assertEqual(normalize_episode("RouteStick", record, 0)["execution_route_layout"], layout)
        for overrides in (dict(yaw_deg=1), dict(translation_xy_m=[.01, 0]),
                          dict(yaw_deg=True), dict(translation_xy_m=[False, 0])):
            with self.assertRaises(ValueError):
                resolve_route_layout(3, {**parameters, **overrides})


if __name__ == "__main__":
    unittest.main()
