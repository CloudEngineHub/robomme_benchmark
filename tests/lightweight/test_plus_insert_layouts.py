"""InsertPeg hole bounds and transfer distance without a simulator."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from robomme_plus.tasks.imitation.utils import geometry as g


class InsertPegLayouts(unittest.TestCase):
    def test_hard_preserves_legacy_geometry_and_random_stream(self):
        for seed in range(500):
            legacy = g.resolve_config("insert_peg", seed)
            hard = g.resolve_config("insert_peg", seed, "hard")
            hard.pop("level")
            hard["difficulty_levels"] = False
            self.assertEqual(hard, legacy)

    def test_easier_widths_keep_wall_material_and_transfer_offsets(self):
        for level, bounds in (("easy", (.023, .025)), ("medium", (.022, .023))):
            widths = []
            for seed in range(500):
                config = g.resolve_config("insert_peg", seed, level)
                self.assertEqual(config, g.resolve_config("insert_peg", seed, level))
                half = config["aperture_half_size_m"]
                widths.append(2 * half)
                self.assertTrue(bounds[0] <= 2 * half <= bounds[1])
                centers = []
                for stand in config["stands"].values():
                    y, z = stand["hole_lateral_m"], stand["hole_height_m"]
                    centers.append([y, z])
                    self.assertGreaterEqual(min(.04 - abs(y) - half, z - half,
                                                .08 - z - half), .0048 - 1e-12)
                    self.assertEqual(len(g.insertion_stand_boxes(z, y, half)), 4)
                self.assertTrue(np.all(np.abs(np.subtract(*centers)) >= .005))
                self.assertTrue(np.all(np.abs(np.subtract(*centers)) <= .010))
            self.assertGreater(np.ptp(widths), .9 * (bounds[1] - bounds[0]))

    def test_legacy_metadata_loads_as_unchanged_hard(self):
        from robomme_plus.evaluation.metadata import normalize_episode
        old = dict(seed=123, difficulty=None, oracle_policy_step=100)
        row = normalize_episode("InsertPeg", old, 0)
        self.assertEqual(row["difficulty"], "hard")
        self.assertIsNone(old["difficulty"])
        self.assertEqual(g.resolve_config("insert_peg", 123, row["difficulty"])["stands"],
                         g.resolve_config("insert_peg", 123)["stands"])

    def test_scoring_uses_physical_clearance_and_keeps_entry_requirements(self):
        metrics = dict(axial_m=.03, lateral_m=.0003, normal_m=0,
                       alignment=1., grasp_side_m=.08)
        self.assertFalse(g.insertion_valid(metrics, .0102, .010, .10))
        self.assertTrue(g.insertion_valid(metrics, .011, .010, .10))
        self.assertTrue(g.insertion_valid(metrics, .0115, .010, .10))
        for invalid in (dict(axial_m=.05), dict(alignment=0), dict(grasp_side_m=-.08)):
            self.assertFalse(g.insertion_valid({**metrics, **invalid}, .0125, .010, .10))

    def test_metadata_rejects_stale_hole_width(self):
        from robomme_plus.evaluation.metadata import normalize_episode
        row = dict(seed=123, difficulty="easy", oracle_policy_step=100,
                   hole_width_m=.0204)
        with self.assertRaisesRegex(ValueError, "hole_width_m"):
            normalize_episode("InsertPeg", row, 0)
        row["hole_width_m"] = 2 * g.resolve_config("insert_peg", 123, "easy")["aperture_half_size_m"]
        self.assertEqual(normalize_episode("InsertPeg", row, 0)["hole_width_m"], row["hole_width_m"])

    def test_saved_visible_grasp_cannot_change_hard_or_replay_wrong_offset(self):
        from robomme_plus.evaluation.metadata import normalize_episode

        row = dict(seed=123, difficulty="easy", oracle_policy_step=100,
                   oracle_wrist_visible_grasp=True, oracle_grasp_offset_m=.005)
        self.assertTrue(normalize_episode("InsertPeg", row, 0)["oracle_wrist_visible_grasp"])
        for change in (dict(difficulty="hard"), dict(oracle_wrist_visible_grasp=1),
                       dict(oracle_grasp_offset_m=-.005), dict(oracle_grasp_offset_m=float("nan"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_episode("InsertPeg", {**row, **change}, 0)

    def test_archived_easier_metadata_keeps_legacy_grasp_selection(self):
        from robomme_plus.evaluation.metadata import normalize_episode

        for level in ("easy", "medium", "hard"):
            row = normalize_episode("InsertPeg", dict(seed=123, difficulty=level,
                                                      oracle_policy_step=100), 0)
            self.assertNotIn("oracle_wrist_visible_grasp", row)
            self.assertNotIn("oracle_grasp_offset_m", row)

    def test_seeded_holes_stay_inside_stand_and_shift_five_to_ten_mm_per_axis(self):
        offsets = []
        for seed in range(500):
            config = g.resolve_config("insert_peg", seed)
            self.assertEqual(config, g.resolve_config("insert_peg", seed))
            holes = np.array([
                [config["stands"][phase]["hole_lateral_m"],
                 config["stands"][phase]["hole_height_m"]]
                for phase in g.INSERT_PHASES
            ])
            self.assertTrue(np.all(holes >= [-.025, .015]))
            self.assertTrue(np.all(holes <= [.025, .065]))
            offset = holes[1] - holes[0]
            self.assertTrue(np.all(np.abs(offset) >= .005))
            self.assertTrue(np.all(np.abs(offset) <= .010))
            self.assertGreaterEqual(np.linalg.norm(offset), np.sqrt(2) * .005)
            self.assertLessEqual(np.linalg.norm(offset), np.sqrt(2) * .010)
            offsets.append(offset)
            demo, execution = (config["stands"][phase] for phase in g.INSERT_PHASES)
            self.assertEqual(demo["yaw_offset_deg"], execution["yaw_offset_deg"])
            self.assertLessEqual(abs(demo["yaw_offset_deg"]), 40)
            for lateral, height in holes:
                boxes = g.insertion_stand_boxes(height, lateral)
                self.assertEqual(len(boxes), 4)
                self.assertTrue(all(np.all(np.asarray(half_size) > 0)
                                    for _, half_size in boxes))
        # Both axes use the expanded range in both directions.
        offsets = np.array(offsets)
        self.assertTrue(np.all(offsets.min(axis=0) < -.009))
        self.assertTrue(np.all(offsets.max(axis=0) > .009))
        self.assertEqual(len(np.unique(np.sign(offsets), axis=0)), 4)

    def test_demo_keeps_uniform_full_face_sample(self):
        for seed in range(100):
            expected = np.random.default_rng(seed + 610138).uniform(
                [-.025, .015], [.025, .065]
            )
            demo = g.resolve_config("insert_peg", seed)["stands"]["demonstration"]
            np.testing.assert_array_equal(
                [demo["hole_lateral_m"], demo["hole_height_m"]], expected
            )

    def test_out_of_bounds_noise_is_redrawn_without_moving_demo_or_clipping(self):
        for base, invalid, valid in (
            ([.024, .064], [.010, .010], [-.006, -.007]),
            ([-.024, .016], [-.010, -.010], [.006, .007]),
        ):
            with self.subTest(base=base):
                hole_rng = Mock()
                hole_rng.uniform.side_effect = [
                    np.array(base), np.array(invalid), np.array(valid)
                ]
                yaw_rng = np.random.default_rng(0)
                with patch.object(g.np.random, "default_rng", side_effect=[hole_rng, yaw_rng]):
                    config = g.resolve_config("insert_peg", 0)
                for phase, expected in zip(
                    g.INSERT_PHASES, (base, np.array(base) + valid)
                ):
                    stand = config["stands"][phase]
                    np.testing.assert_array_equal(
                        [stand["hole_lateral_m"], stand["hole_height_m"]], expected
                    )

    def test_small_offset_on_either_axis_is_redrawn(self):
        base = np.array([0., .04])
        valid = np.array([.006, -.007])
        hole_rng = Mock()
        hole_rng.uniform.side_effect = [
            base, np.array([.004, .008]), np.array([-.008, -.004]), base * 0, valid
        ]
        with patch.object(g.np.random, "default_rng", side_effect=[hole_rng, np.random.default_rng(0)]):
            config = g.resolve_config("insert_peg", 0)
        for phase, expected in zip(g.INSERT_PHASES, (base, base + valid)):
            stand = config["stands"][phase]
            np.testing.assert_array_equal(
                [stand["hole_lateral_m"], stand["hole_height_m"]], expected
            )


if __name__ == "__main__":
    unittest.main()
