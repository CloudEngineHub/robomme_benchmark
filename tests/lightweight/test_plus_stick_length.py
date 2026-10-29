"""Seeded execution lengths stay consistent with robot assets and real tips."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from robomme_plus.tasks.imitation.utils.geometry import resolve_config
from robomme_plus.tasks.imitation.utils.split_stick import (
    LOWER_LINK, SHORT_TIP_LINK, STICK_LENGTH_M, STICK_START_M,
    asset_paths, lowered, scoring_point,
)
from robomme_plus.tasks.imitation.utils.variants import (
    TRACE_PROFILES, tracing_stick_fraction,
)


class StickLengthTest(unittest.TestCase):
    def test_lengths_cover_the_uniform_range_and_replay_at_every_level(self):
        samples = {}
        for task, profile in TRACE_PROFILES.items():
            values = np.array([tracing_stick_fraction(task, seed) for seed in range(1000)])
            self.assertTrue(np.all((values >= .25) & (values <= .5)))
            self.assertLess(values.min(), .251)
            self.assertGreater(values.max(), .499)
            self.assertLess(abs(values.mean() - .375), .01)
            self.assertEqual(len(np.unique(values)), len(values))
            samples[task] = values
            for level in ("easy", "medium", "hard"):
                for seed in range(10):
                    cfg = resolve_config(profile, seed, level)
                    expected = 1.0 if level == "easy" else values[seed]
                    self.assertEqual(cfg["execution_stick_fraction"], expected)
        self.assertFalse(np.array_equal(samples["PatternLock"], samples["RouteStick"]))

    def test_route_length_ranges_and_tool_changes_vary_independently(self):
        for level, limits in (("easy", (8, 13)), ("medium", (4, 7)), ("hard", (6, 8))):
            counts = set()
            for seed in range(200):
                config = resolve_config("route_trace", seed, level)
                counts.add(config["segment_count"])
                self.assertEqual(config["shorten_stick"], level != "easy")
                self.assertFalse(config["board"])
                self.assertEqual(config["execution_stick_fraction"],
                                 1.0 if level == "easy" else tracing_stick_fraction("RouteStick", seed))
            self.assertEqual(counts, set(range(limits[0], limits[1] + 1)))

    def test_assets_match_the_fraction_without_overwriting_other_lengths(self):
        with TemporaryDirectory() as directory, patch.object(Path, "home", return_value=Path(directory)):
            saved = {}
            for fraction in (.25, .37654321, .5):
                path = Path(asset_paths(fraction))
                saved[path] = path.read_bytes()
                root = ET.fromstring(saved[path])
                for kind in ("visual", "collision"):
                    upper = root.findall(f"./link[@name='panda_hand']/{kind}")
                    upper = next(item for item in upper if item.find("geometry/cylinder") is not None)
                    lower = root.find(f"./link[@name='{LOWER_LINK}']/{kind}")
                    lengths = [float(item.find("geometry/cylinder").attrib["length"]) for item in (upper, lower)]
                    centers = [float(item.find("origin").attrib["xyz"].split()[2]) for item in (upper, lower)]
                    self.assertAlmostEqual(lengths[0], STICK_LENGTH_M * fraction, places=8)
                    self.assertAlmostEqual(sum(lengths), STICK_LENGTH_M, places=8)
                    self.assertAlmostEqual(centers[0] - lengths[0]/2, STICK_START_M, places=8)
                    self.assertAlmostEqual(centers[0] + lengths[0]/2, centers[1] - lengths[1]/2, places=8)
                    self.assertAlmostEqual(centers[1] + lengths[1]/2, STICK_START_M + STICK_LENGTH_M, places=8)
                tip = root.find(f"./joint[@name='{SHORT_TIP_LINK}_joint']/origin")
                self.assertAlmostEqual(float(tip.attrib["xyz"].split()[2]), STICK_START_M + STICK_LENGTH_M*fraction, places=8)
                self.assertTrue(path.with_suffix(".srdf").exists())
            self.assertEqual(len(saved), 3)
            for path, content in saved.items():
                self.assertEqual(path.read_bytes(), content)

    def test_oracle_offset_puts_real_tip_at_requested_pose_for_any_orientation(self):
        import sapien
        from transforms3d.euler import euler2quat
        from transforms3d.quaternions import quat2mat

        for fraction in (.25, .37654321, .5):
            for angles in ((np.pi, 0, 0), (2.9, .2, -.7)):
                pose = sapien.Pose([.1, -.2, .06], euler2quat(*angles))
                tcp_goal = lowered(pose, fraction)
                real_tip = tcp_goal.p - quat2mat(tcp_goal.q) @ [0, 0, STICK_LENGTH_M*(1-fraction)]
                np.testing.assert_allclose(real_tip, pose.p, atol=1e-8)

    def test_scoring_uses_full_tip_in_demonstration_and_real_tip_in_execution(self):
        tcp, short_tip, other = object(), object(), object()
        env = SimpleNamespace(agent=SimpleNamespace(tcp=tcp), _stick_short_tip=short_tip, _stick_short=False)
        self.assertIs(scoring_point(env, tcp), tcp)
        env._stick_short = True
        self.assertIs(scoring_point(env, tcp), short_tip)
        self.assertIs(scoring_point(env, other), other)

    def test_route_direction_tracks_the_visible_tip_when_the_arm_is_tilted(self):
        import torch
        from robomme_plus.tasks.imitation.route_stick import _TaskScene

        def actor(x, y, z=.02):
            return SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[x, y, z]])))

        for short, expected in ((False, "clockwise"), (True, "counterclockwise")):
            with self.subTest(short=short):
                env = SimpleNamespace(
                    current_task_demonstration=False, _gripper_xy_trace=[],
                    agent=SimpleNamespace(tcp=actor(.045, .01)),
                    _stick_short=short, _stick_short_tip=actor(.045, -.02, .072),
                    elapsed_steps=torch.tensor([1]), use_demonstrationwrapper=True,
                    demonstration_record_traj=False, timestep=0,
                    task_list=[dict(name="route", func=lambda: False)],
                )
                _TaskScene.evaluate(env)
                self.assertTrue(_TaskScene.direction_fail(env, [actor(.09, 0), actor(0, 0), expected]))
                _TaskScene.evaluate(env)
                wrong = "clockwise" if expected == "counterclockwise" else "counterclockwise"
                self.assertFalse(_TaskScene.direction_fail(env, [actor(.09, 0), actor(0, 0), wrong]))


if __name__ == "__main__":
    unittest.main()
