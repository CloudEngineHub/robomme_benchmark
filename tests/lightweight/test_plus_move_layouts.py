"""MoveCube layout coverage and physical placement constraints (no simulator)."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from robomme_plus.tasks.imitation.utils import geometry as g
from robomme_plus.tasks.imitation.utils.variants import imitation_variant
from robomme_plus.evaluation.metadata import load_episode_metadata, normalize_episode
from robomme_plus.tasks.imitation.move_cube_support.config import validate_config, phase_available_methods


class MoveCubeLayouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            method: [g.resolve_config("move_" + method, seed) for seed in range(120)]
            for method in ("hook", "poke", "scoop")
        }

    def test_reproducible_and_tool_sizes_do_not_choose_layout(self):
        for method, configs in self.configs.items():
            config = configs[17]
            self.assertEqual(config, g.resolve_config("move_" + method, 17))
            lengths = (config["execution_tool"]["tool_length_m"], config["tool_length_m"])
            with patch.object(g, "_move_lengths", return_value=lengths):
                other = g.resolve_config("move_" + method, 17)
            for prefix in ("", "execution_"):
                for key in ("cube_xy_m", "target_xy_m", "wall_xy_m", "wall_yaw_rad"):
                    if prefix + key in config:
                        self.assertEqual(config[prefix + key], other[prefix + key])

    def test_layouts_and_resting_tools_are_clear_and_reachable(self):
        for method, configs in self.configs.items():
            for config in configs:
                self.assertEqual(config["layout_version"], 6)
                self.assertGreaterEqual(
                    abs(config["tool_length_m"] - config["execution_tool"]["tool_length_m"]), .05
                )
                self.assertGreaterEqual(math.dist(config["handle_xy_m"], config["execution_handle_xy_m"]), .05)
                for phase, prefix in (("demonstration", ""), ("execution", "execution_")):
                    cube = np.array(config[prefix + "cube_xy_m"])
                    target = np.array(config[prefix + "target_xy_m"])
                    self.assertGreaterEqual(math.dist(cube, target), .12)
                    wall = config.get(prefix + "wall_xy_m")
                    direction = config.get(prefix + "wall_direction_xy")
                    self.assertTrue(g.layout_reachable(config, cube, target, wall, direction))
                    tool = config if not prefix else config["execution_tool"]
                    parts = g.tool_footprint(config[prefix + "handle_xy_m"], config[prefix + "handle_yaw_rad"], tool, method)
                    self.assertTrue(all(g.inside_move_table(g._footprint_points(part, 2)) for part in parts))
                    end = target if wall is None else cube + np.array(direction) * config[prefix + "cube_wall_gap_m"]
                    gaps = g.rest_clearance(parts, cube, target, [(cube, end)], wall, config.get(prefix + "wall_yaw_rad", 0))
                    for key, gap in gaps.items():
                        self.assertGreaterEqual(gap, g.REST_CLEARANCE_M[key])
                    if wall is not None:
                        self.assertAlmostEqual(g.scoop_push_distance_m(cube, wall, direction), config[prefix + "cube_wall_gap_m"])
                        yaw = config[prefix + "wall_yaw_rad"]
                        # Wall local y is the push normal, regardless of yaw.
                        np.testing.assert_allclose([-math.sin(yaw), math.cos(yaw)], direction, atol=1e-12)

    def test_center_starts_varied_targets_and_no_forced_side_switch(self):
        for method, configs in self.configs.items():
            for key in ("cube_xy_m", "target_xy_m", "handle_xy_m"):
                points = np.array([c[prefix + key] for c in configs for prefix in ("", "execution_")])
                self.assertGreater(sum(np.linalg.norm(points, axis=1) < .13), 15, (method, key))
                self.assertTrue(np.all(np.ptp(points, axis=0) > [.32, .48]), (method, key))
            headings = np.array([c[p + "handle_yaw_rad"] for c in configs for p in ("", "execution_")])
            self.assertTrue(np.all(np.histogram(headings, bins=np.linspace(-np.pi, np.pi, 5))[0] > 10), method)
            same_side = sum(c["cube_xy_m"][1] * c["execution_cube_xy_m"][1] > 0 for c in configs)
            self.assertGreater(same_side, 30)
            self.assertLess(same_side, 90)
            for config in configs:
                for key in ("cube_xy_m", "target_xy_m"):
                    self.assertGreaterEqual(math.dist(config[key], config["execution_" + key]), .08)

    def test_scoop_wall_directions_are_balanced_and_change_between_phases(self):
        configs = self.configs["scoop"]
        directions = np.array([c[p + "wall_direction_xy"] for c in configs for p in ("", "execution_")])
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        counts = np.histogram(angles, bins=np.linspace(-np.pi, np.pi, 9))[0]
        self.assertTrue(np.all(counts > 15), counts)
        self.assertTrue(np.all(counts < 50), counts)
        for config in configs:
            self.assertNotEqual(config["wall_xy_m"], config["execution_wall_xy_m"])
            self.assertNotEqual(config["wall_yaw_rad"], config["execution_wall_yaw_rad"])

    def test_default_has_five_methods_and_explicit_direction_aliases_work(self):
        methods = [imitation_variant("MoveCube", seed)[0] for seed in range(600)]
        self.assertEqual(set(methods), {"hook", "poke", "scoop", "ball", "cue"})
        for method in set(methods):
            self.assertGreater(methods.count(method), 85)
        for variant, sign in (("scoop_left", 1), ("scoop_right", -1)):
            _, profile = imitation_variant("MoveCube", 0, variant)
            config = g.resolve_config(profile, 0)
            for prefix in ("", "execution_"):
                np.testing.assert_allclose(config[prefix + "wall_direction_xy"], [0, sign], atol=1e-12)

    def test_axis_aligned_poke_and_inner_reach_crossing(self):
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (.6, .8)):
            axis = g.poke_shaft_axis(d)
            self.assertAlmostEqual(np.linalg.norm(axis), 1)
            self.assertAlmostEqual(float(axis @ d), 0)
            self.assertGreaterEqual(axis[0], 0)
        base = np.array(g.ROBOT_BASE_XY)
        self.assertFalse(g._reachable([base + [.4, 0], base - [.4, 0]]))
        self.assertTrue(g._reachable([base + [.4, .1], base + [.4, -.1]]))

    def test_reachable_cube_still_requires_legal_tool_use(self):
        for method in self.configs:
            evidence = {"correct_contact_steps": 10, "forward_motion_m": .2,
                        "wall_contact_steps": 4, "supported_lift_steps": 5, "carried_distance_m": .2}
            self.assertTrue(g.move_complete(method, evidence, True, True, True))
            for illegal in ("robot_contact_steps", "shaft_contact_steps", "handle_contact_steps", "gripper_contact"):
                self.assertFalse(g.move_complete(method, dict(evidence, **{illegal: 1}), True, True, True))

    def test_cube_orientations_are_random_and_independent_of_wall_heading(self):
        for method, configs in self.configs.items():
            for prefix in ('', 'execution_'):
                yaws = np.array([c[prefix + 'cube_yaw_rad'] for c in configs])
                self.assertGreater(np.ptp(yaws), 6)
                self.assertEqual(len(np.unique(yaws)), len(configs))
                if method == 'scoop':
                    walls = np.array([c[prefix + 'wall_yaw_rad'] for c in configs])
                    self.assertLess(abs(np.corrcoef(yaws, walls)[0, 1]), .25)

    def test_recorded_scenes_keep_geometry_and_budget(self):
        for row in load_episode_metadata("MoveCube"):
            frozen = row["scene_parameters"]
            replay = validate_config(frozen, row["variant"], row["seed"])
            self.assertEqual(frozen, replay)
            for phase, methods in phase_available_methods(replay).items():
                self.assertIn(row["variant"], methods)
                self.assertIsNot(frozen["layouts"][phase], replay["layouts"][phase])
            self.assertEqual(row["max_steps"], math.ceil(1.5 * row["oracle_policy_step"]) + 200)
            self.assertIsNot(frozen, replay)
        new = self.configs["scoop"][5]
        row = normalize_episode("MoveCube", {"seed":5, "variant":"scoop", "difficulty":None,
                                             "oracle_policy_step":100, "scene_parameters":new}, 0)
        self.assertEqual(row["scene_parameters"], new)
        self.assertEqual(g.recorded_move_config(new, "move_scoop", 5), new)
        with self.assertRaisesRegex(ValueError, "identity"):
            g.recorded_move_config(new, "move_scoop", 6)


if __name__ == "__main__":
    unittest.main()
