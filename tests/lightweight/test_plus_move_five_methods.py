"""Five-method identity, diversity, and frozen initialization contract (CPU only)."""

import copy
import json
import unittest

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.registry import TASKS
from robomme_plus.tasks.imitation.move_cube_support import config as c
from robomme_plus.tasks.imitation.move_cube_support.diversity import (
    RELEASE_BANDS,
    sample_pan_loading_surface,
    sampling_plan,
    validate_reference_loading_surface,
)
from robomme_plus.tasks.imitation.move_cube_support.layout import (
    COMBINATIONS,
    DEFAULT_RANGES,
    METHODS,
    PAN_LOADING_FACES,
    PrecisionLayout,
    geometry_check,
    CUE_TARGET_NEIGHBORHOOD_M,
)


class FiveMethodInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = c.sample_config(0)

    def test_all_five_and_all_seven_paired_trios_are_sampled(self):
        samples = [c.method_selection(seed) for seed in range(3000)]
        self.assertEqual({methods for methods, _ in samples}, set(COMBINATIONS))
        self.assertEqual({method for _, method in samples}, set(METHODS))
        for method in METHODS:
            count = sum(required == method for _, required in samples)
            self.assertGreater(count, 500)
            self.assertLess(count, 700)
            self.assertIn(method, TASKS["MoveCube"].variants)
        self.assertTrue(
            all(len(group) == 3 and method in group for group, method in samples)
        )
        self.assertEqual(len(COMBINATIONS), 7)
        self.assertTrue(all("scoop" not in group or "ball" in group for group, _ in samples))

    def test_both_phases_pair_the_ramp_with_ball_for_every_required_method(self):
        for seed in range(100):
            for method in METHODS:
                phases, required = c.phase_method_selection(seed, method)
                self.assertEqual(required, method)
                for group in phases.values():
                    self.assertEqual(len(group), 3)
                    self.assertIn(method, group)
                    if "scoop" in group:
                        self.assertIn("ball", group)

    def test_new_records_reject_unpaired_ramp_but_archives_remain_replayable(self):
        cfg = copy.deepcopy(self.config)
        cfg["method"] = "scoop"
        cfg["available_methods"] = {phase: ["hook", "poke", "scoop"]
                                    for phase in ("demonstration", "execution")}
        with self.assertRaisesRegex(ValueError, "paired with a ball"):
            c.validate_config(cfg)
        cfg["schema"] = c.LEGACY_SCHEMA
        self.assertEqual(c.validate_config(cfg), cfg)

    def test_requested_method_is_present_without_unique_tool_layout(self):
        common = set.intersection(
            *(set(group) for group in self.config["available_methods"].values())
        )
        for method in common:
            other = c.sample_config(0, method)
            self.assertEqual(other["layouts"], self.config["layouts"])
            self.assertEqual(
                other["available_methods"], self.config["available_methods"]
            )

    def test_phase_trios_can_differ_and_share_a_distractor(self):
        different = 0
        for seed in range(300):
            phases, required = c.phase_method_selection(seed)
            a, b = phases.values()
            self.assertEqual(len(a), 3)
            self.assertEqual(len(b), 3)
            self.assertIn(required, set(a) & set(b))
            self.assertGreaterEqual(len(set(a) & set(b)), 2)
            different += a != b
        self.assertGreater(different, 200)
        self.assertLess(different, 300)

    def test_reproducible_independent_phase_layouts(self):
        self.assertEqual(self.config, c.sample_config(0))
        a, b = self.config["layouts"].values()
        for key in (
            "cube_xyz",
            "target_xyz",
            "tool_poses",
            "tool_params",
            "ramp_grade_deg",
            "ramp_width_m",
            "ramp_length_m",
            "cue_pivot_xyz",
            "cue_front_m",
            "desired_release_fraction",
        ):
            self.assertNotEqual(a[key], b[key], key)

    def test_masses_stay_observable_through_the_demonstration(self):
        a, b = self.config["layouts"].values()
        self.assertEqual(a["cube_mass_kg"], b["cube_mass_kg"])
        self.assertEqual(a["ball_mass"], b["ball_mass"])
        self.assertNotEqual(a["target_xyz"], b["target_xyz"])

    def test_batch_keeps_wide_physical_coverage_for_each_method_and_phase(self):
        for method_index in range(5):
            plans = [
                sampling_plan(
                    20261017,
                    DEFAULT_RANGES["cube_mass_kg"],
                    method_index=method_index,
                    episode_slot=slot,
                )
                for slot in range(5)
            ]
            for phase in ("demonstration", "execution"):
                for key, expected in (
                    ("cube_mass_kg", DEFAULT_RANGES["cube_mass_kg"]),
                    ("cube_target_distance_m", (0.08, 0.32)),
                ):
                    bands = sorted(tuple(plan[phase][key]) for plan in plans)
                    self.assertEqual(len(set(bands)), 5)
                    self.assertEqual(bands[0][0], expected[0])
                    self.assertEqual(bands[-1][1], expected[1])
                    self.assertTrue(all(a[1] == b[0] for a, b in zip(bands, bands[1:])))
                self.assertEqual(
                    {plan[phase]["ball_release"]["regime"] for plan in plans},
                    set(RELEASE_BANDS),
                )

    def test_loading_surface_is_seeded_per_episode_and_not_a_method_label(self):
        observed = set()
        for slot in range(30):
            surfaces = []
            for method in range(5):
                plan = sampling_plan(17, DEFAULT_RANGES["cube_mass_kg"],
                                     method_index=method, episode_slot=slot)
                pair = [plan[p]["pan_loading_surface"] for p in ("demonstration", "execution")]
                self.assertEqual(pair[0], pair[1])
                surfaces.append(pair)
            self.assertTrue(all(pair == surfaces[0] for pair in surfaces))
            observed.add(surfaces[0][0])
            self.assertEqual(surfaces[0][0], sample_pan_loading_surface(17, episode_slot=slot))
        self.assertEqual(observed, {"incline", "wall"})

    def test_wall_probability_is_twenty_percent_for_seeds_and_batch_slots(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                plans = [sampling_plan(17 if batched else index, DEFAULT_RANGES["cube_mass_kg"],
                                       method_index=2, episode_slot=index if batched else None)
                         for index in range(2000)]
                for phase in ("demonstration", "execution"):
                    wall_fraction = sum(p[phase]["pan_loading_surface"] == "wall" for p in plans) / len(plans)
                    self.assertGreater(wall_fraction, .17)
                    self.assertLess(wall_fraction, .23)

    def test_scoop_geometry_honors_both_surfaces_and_rejects_fallback(self):
        for surface in ("incline", "wall"):
            seed = next(s for s in range(100) if sample_pan_loading_surface(s) == surface)
            cfg = c.sample_config(seed, "scoop")
            c.validate_config(cfg)
            for phase, layout in cfg["layouts"].items():
                self.assertEqual(cfg["sampling_plan"][phase]["pan_loading_surface"], surface)
                self.assertIn(layout["pan_face"], PAN_LOADING_FACES[surface])
                self.assertEqual(layout["pan_floor_friction"], .8)
                ramp = PrecisionLayout(**layout)
                site, normal = ramp.pan_site()
                self.assertGreater(float((site - ramp.cube) @ normal), .02)
                if surface == 'incline':
                    self.assertGreater(float((ramp.cube - ramp.toe) @ ramp.ramp_direction), 0)
                self.assertIn("ball", cfg["available_methods"][phase])
        cfg = c.sample_config(0, "scoop")
        bad = copy.deepcopy(cfg)
        phase = next(p for p in cfg["layouts"] if cfg["sampling_plan"][p]["pan_loading_surface"] == "incline")
        bad["layouts"][phase]["pan_face"] = "left_side"
        with self.assertRaisesRegex(ValueError, "loading surface violates"):
            c.validate_config(bad)
        # Old frozen records have no loading-surface quota.
        for quota in bad["sampling_plan"].values():
            quota.pop("pan_loading_surface")
        self.assertEqual(c.validate_config(bad), bad)

    def test_reference_surface_audit_checks_contact_not_only_layout_tag(self):
        cfg = dict(method="scoop", sampling_plan={"demonstration": {"pan_loading_surface": "incline"}})
        def result(**counts):
            return {"strict_evidence": {"ramp_surface_contact_steps": counts}}
        validate_reference_loading_surface(cfg, "demonstration", result(incline=2, edge=100))
        for evidence in (result(), result(side_wall=20), result(incline=20, rear_wall=1)):
            with self.assertRaisesRegex(RuntimeError, "loading quota"):
                validate_reference_loading_surface(cfg, "demonstration", evidence)
        cfg["method"] = "ball"  # The pan is a distractor, not a required action.
        validate_reference_loading_surface(cfg, "demonstration", result())

    def test_phase_mass_and_distance_are_independent_choices(self):
        combinations = set()
        for seed in range(300):
            plan = sampling_plan(
                seed, DEFAULT_RANGES["cube_mass_kg"], mass_scope="phase"
            )
            a, b = plan.values()
            for key in ("cube_mass_kg", "cube_target_distance_m"):
                self.assertNotEqual(a[key], b[key])
            short, long = sorted(
                plan.values(), key=lambda p: p["cube_target_distance_m"][0]
            )
            self.assertLessEqual(
                short["ball_release"]["fraction_range"][0],
                long["ball_release"]["fraction_range"][0],
            )
            for quota in plan.values():
                combinations.add(
                    (quota["cube_mass_kg"][0], quota["cube_target_distance_m"][0])
                )
        self.assertEqual(len(combinations), 25)

    def test_three_episode_review_still_spans_mass_and_distance_ranges(self):
        plans = [
            sampling_plan(
                17,
                DEFAULT_RANGES["cube_mass_kg"],
                method_index=3,
                episode_slot=i,
                episodes_per_method=3,
            )
            for i in range(3)
        ]
        for phase in ("demonstration", "execution"):
            for key, limits in (
                ("cube_mass_kg", (0.025, 0.14)),
                ("cube_target_distance_m", (0.08, 0.32)),
            ):
                bands = sorted(tuple(p[phase][key]) for p in plans)
                self.assertEqual(len(set(bands)), 3)
                self.assertEqual((bands[0][0], bands[-1][1]), limits)
        self.assertEqual(
            {
                p[phase]["ball_release"]["regime"]
                for p in plans
                for phase in ("demonstration", "execution")
            },
            set(RELEASE_BANDS),
        )

    def test_release_cases_include_top_and_lower_ramp_heights(self):
        for phase, data in self.config["layouts"].items():
            layout = PrecisionLayout(**data)
            low, high = self.config["sampling_plan"][phase]["ball_release"][
                "fraction_range"
            ]
            self.assertTrue(low <= layout.desired_release_fraction <= high)
            self.assertAlmostEqual(layout.point(1.0)[2], layout.ramp_height)
            self.assertLess(layout.point(0.5)[2], layout.ramp_height)
        cfg = copy.deepcopy(self.config)
        cfg["method"] = "ball"
        phase = "demonstration"
        cfg["oracle_parameters"][phase] = {"fraction": 0.0}
        with self.assertRaisesRegex(ValueError, "release-height quota"):
            c.validate_config(cfg)

    def test_frozen_physical_parameters_cannot_drift_out_of_quota(self):
        cfg = copy.deepcopy(self.config)
        cfg["layouts"]["demonstration"]["cube_mass_kg"] = 0.025
        with self.assertRaisesRegex(ValueError, "masses must match|sampling quota"):
            c.validate_config(cfg)
        # Records made before quotas were introduced remain replayable.
        cfg = copy.deepcopy(self.config)
        del cfg["sampling_plan"]
        self.assertEqual(c.validate_config(cfg), cfg)

    def test_distance_bins_and_dimensions_do_not_collapse_to_small_values(self):
        bins = {c.distance_band(seed, phase) for seed in range(100) for phase in (0, 1)}
        self.assertEqual(len(bins), 5)
        self.assertAlmostEqual(min(a for a, _ in bins), 0.08)
        self.assertAlmostEqual(max(b for _, b in bins), 0.32)
        self.assertEqual(DEFAULT_RANGES["ramp_length_m"], (0.15, 0.30))
        self.assertEqual(DEFAULT_RANGES["ramp_grade_deg"], (10.0, 40.0))
        self.assertGreaterEqual(DEFAULT_RANGES["hook_length_m"][1], 0.4)
        for phase, layout in enumerate(self.config["layouts"].values()):
            distance = np.linalg.norm(
                np.array(layout["target_xyz"]) - layout["cube_xyz"]
            )
            low, high = c.distance_band(0, phase)
            self.assertTrue(low <= distance <= high)
            geometry_check(PrecisionLayout(**layout))

    def test_cue_full_sweep_cannot_bring_cube_into_target_neighborhood(self):
        for phase, data in self.config["layouts"].items():
            if "cue" not in self.config["available_methods"][phase]:
                continue
            layout = PrecisionLayout(**data)
            audit = geometry_check(layout)
            self.assertGreaterEqual(audit["static_target_clearance_m"], .015)
            self.assertGreater(audit["minimum_slow_cube_target_distance_m"], .079)
            # Even a cube at the most favorable edge of the full contact disk
            # stays outside the target neighborhood for every yaw and stroke.
            for angle in np.linspace(-4*np.pi, 4*np.pi, 361):
                edge = layout.cue_pivot[:2] + layout.cue_contact_radius * np.array([np.cos(angle), np.sin(angle)])
                self.assertGreater(np.linalg.norm(layout.target[:2] - edge), CUE_TARGET_NEIGHBORHOOD_M)
            from dataclasses import replace
            close = replace(layout, target_xyz=tuple(layout.cue_pivot +
                (layout.target-layout.cue_pivot)/np.linalg.norm(layout.target-layout.cue_pivot)
                * (layout.cue_contact_radius + CUE_TARGET_NEIGHBORHOOD_M + .005)))
            with self.assertRaisesRegex(ValueError, "neighborhood"):
                geometry_check(close)

    def test_frozen_record_round_trip_and_budget(self):
        cfg = json.loads(json.dumps(self.config))
        self.assertEqual(c.validate_config(cfg, cfg["method"], 0), cfg)
        record = normalize_episode(
            "MoveCube",
            dict(
                seed=0,
                variant=cfg["method"],
                difficulty=None,
                oracle_policy_step=321,
                scene_parameters=cfg,
            ),
            0,
        )
        self.assertEqual(record["max_steps"], 682)
        self.assertEqual(record["scene_parameters"], cfg)

    def test_invalid_or_shortcut_records_are_rejected(self):
        for key, value in [
            ("cue_stroke_m", 0.5),
            ("cube_mass_kg", float("nan")),
            ("target_xyz", self.config["layouts"]["demonstration"]["cube_xyz"]),
        ]:
            cfg = copy.deepcopy(self.config)
            cfg["layouts"]["demonstration"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_config(cfg)
        cfg = copy.deepcopy(self.config)
        cfg["available_methods"] = ["hook", "poke", "scoop"]
        cfg["method"] = "ball"
        with self.assertRaisesRegex(ValueError, "absent"):
            c.validate_config(cfg)
        with self.assertRaisesRegex(ValueError, "seed"):
            c.validate_config(self.config, seed=1)
        cfg = copy.deepcopy(self.config)
        cfg["method"] = "hook"
        cfg["available_methods"] = dict(
            demonstration=["hook", "poke", "scoop"], execution=["hook", "ball", "cue"]
        )
        with self.assertRaisesRegex(ValueError, "shared distractor"):
            c.validate_config(cfg)
        for field in ("tool_params", "tool_poses"):
            cfg = copy.deepcopy(self.config)
            del cfg["layouts"]["demonstration"][field]["scoop"]
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "missing"),
            ):
                c.validate_config(cfg)

    def test_custom_ranges_are_recorded_and_unknown_keys_fail(self):
        ranges = c.validate_ranges({"hook_length_m": (0.3, 0.4)})
        self.assertEqual(ranges["hook_length_m"], [0.3, 0.4])
        with self.assertRaises(ValueError):
            c.validate_ranges({"unknown": (1, 2)})
        with self.assertRaises(ValueError):
            c.validate_ranges({"ball_mass_kg": (0.4, 0.1)})


if __name__ == "__main__":
    unittest.main()
