"""Physical tilted-tip geometry and the strict drawing/contact angle gate."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from robomme_plus.tasks.imitation.utils.drawing import Drawing
from robomme_plus.tasks.imitation.utils.geometry import resolve_config
from robomme_plus.tasks.imitation.utils.pattern_dwell import PatternDwell
from robomme_plus.tasks.imitation.utils.split_stick import (
    asset_paths, lowered, tilt_rotation, CHANGED_LINK, SHORT_TIP_LINK,
    STICK_START_M, STICK_LENGTH_M,
)
from robomme_plus.tasks.imitation.utils.variants import tracing_stick_tilt, pattern_stick_tilt


class TiltedStickTest(unittest.TestCase):
    def test_discrete_execution_angles_and_pinned_thirty_degree_config(self):
        angles = [pattern_stick_tilt(seed)[0] for seed in range(300)]
        self.assertEqual(set(angles), {10, 20, 30})
        for angle in (10, 20, 30):
            self.assertGreater(angles.count(angle), 70)
            for seed in (0, 7, 97, 102):
                tilt = pattern_stick_tilt(seed, angle)
                self.assertEqual(tilt, pattern_stick_tilt(seed, angle))
                self.assertEqual(tilt[0], angle)
                self.assertEqual(tilt[1], tracing_stick_tilt(seed)[1])
                old = resolve_config('pattern_trace', seed, 'medium', stick_change_version=2)
                new = resolve_config('pattern_trace', seed, 'medium', stick_change_version=3,
                                     execution_stick_angle_deg=angle)
                self.assertEqual(new['execution_stick_tilt_deg'], tilt)
                self.assertEqual(new['execution_stick_angle_deg'], angle)
                self.assertEqual(new['drawing_max_angle_deg'], 8)
                self.assertEqual(new['drawing_height_m'], .08)
                self.assertEqual(new['execution_stick_fraction'], old['execution_stick_fraction'])
                self.assertEqual(new['path_node_count'], old['path_node_count'])

    def test_discrete_tilt_metadata_roundtrip_and_invalid_values(self):
        from robomme_plus.evaluation.metadata import normalize_episode
        record = dict(seed=97, difficulty='medium', oracle_policy_step=224,
                      stick_change_version=3, execution_stick_angle_deg=30, execution_grid_scale=1.4)
        normalized = normalize_episode('PatternLock', record, 0)
        self.assertEqual(normalize_episode('PatternLock', normalized, 0), normalized)
        self.assertEqual(normalized['execution_stick_angle_deg'], 30)
        for angle in (True, False, None, 0, 15, 29.9, 30.1, '30', float('nan')):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                normalize_episode('PatternLock', dict(record, execution_stick_angle_deg=angle), 0)
        with self.assertRaises(ValueError):
            normalize_episode('PatternLock', dict(record, stick_change_version=2), 0)

    def test_seeded_cone_is_uniform_in_solid_angle_and_bounded(self):
        values = np.array([tracing_stick_tilt(seed) for seed in range(2000)])
        self.assertTrue(np.all((values[:, 0] > 0) & (values[:, 0] < 30)))
        self.assertTrue(np.all(np.abs(values[:, 1]) <= 180))
        self.assertLess(values[:, 0].min(), 1)
        self.assertGreater(values[:, 0].max(), 29.9)
        self.assertLess(abs(np.cos(np.radians(values[:, 0])).mean() - (1 + np.cos(np.pi/6))/2), .003)
        for seed in range(20):
            self.assertEqual(tracing_stick_tilt(seed), tracing_stick_tilt(seed))
            old = resolve_config('pattern_trace', seed, 'medium')
            preview = resolve_config('pattern_trace', seed, 'medium', stick_change_version=2)
            self.assertEqual(old['execution_stick_fraction'], preview['execution_stick_fraction'])
            self.assertEqual(old['path_node_count'], preview['path_node_count'])
            self.assertNotIn('drawing_max_angle_deg', old)
            self.assertEqual(preview['drawing_max_angle_deg'], 8)
            self.assertEqual(preview['drawing_height_m'], old['drawing_height_m'])

    def test_assets_have_full_demo_and_separate_tilted_execution_geometry(self):
        from transforms3d.euler import euler2mat
        with TemporaryDirectory() as directory, patch.object(Path, 'home', return_value=Path(directory)):
            fraction, tilt = .35, [30, -75]
            path = Path(asset_paths(fraction, tilt))
            self.assertNotEqual(path, Path(asset_paths(fraction)))
            root = ET.fromstring(path.read_text())
            # The demonstration's separate full cylinder has an identity mount.
            demo_joint = root.find("./joint[@name='stick_lower_joint']")
            self.assertIsNone(demo_joint.find('origin'))
            demo_visual = root.find("./link[@name='stick_lower']/visual/origin")
            self.assertEqual(demo_visual.attrib['rpy'], '0 0 0')
            joint = root.find(f"./joint[@name='{CHANGED_LINK}_joint']/origin")
            matrix = euler2mat(*map(float, joint.attrib['rpy'].split()))
            np.testing.assert_allclose(matrix, tilt_rotation(tilt), atol=1e-10)
            np.testing.assert_allclose(list(map(float, joint.attrib['xyz'].split())), [0, 0, STICK_START_M])
            tip = root.find(f"./joint[@name='{SHORT_TIP_LINK}_joint']")
            self.assertEqual(tip.find('parent').attrib['link'], CHANGED_LINK)
            self.assertAlmostEqual(float(tip.find('origin').attrib['xyz'].split()[2]), .035)
            for kind in ('visual', 'collision'):
                full = root.find(f"./link[@name='stick_lower']/{kind}/geometry/cylinder")
                short = root.find(f"./link[@name='{CHANGED_LINK}']/{kind}/geometry/cylinder")
                self.assertAlmostEqual(float(full.attrib['length']), .1)
                self.assertAlmostEqual(float(short.attrib['length']), .035)

    def test_oracle_compensates_mount_and_length_to_make_physical_axis_vertical(self):
        import sapien
        from transforms3d.euler import euler2quat
        from transforms3d.quaternions import quat2mat
        for seed in (1, 7, 85):
            tilt = tracing_stick_tilt(seed)
            for yaw in (0, .7, -2):
                desired = sapien.Pose([-.1, .12, .07], euler2quat(np.pi, 0, yaw))
                actual = lowered(desired, .35, tilt)
                hand = quat2mat(actual.q)
                tip_offset = np.array([0, 0, STICK_START_M]) + tilt_rotation(tilt) @ [0, 0, STICK_LENGTH_M*.35]
                actual_tip = actual.p + hand @ (tip_offset - [0, 0, .15])
                np.testing.assert_allclose(actual_tip, desired.p, atol=1e-7)
                np.testing.assert_allclose(hand @ tilt_rotation(tilt) @ [0, 0, 1], [0, 0, -1], atol=1e-7)

    def test_contact_requires_strictly_less_than_eight_degrees_and_existing_height(self):
        drawing = Drawing(max_angle_deg=8)
        drawing.begin('execution', 0)
        for angle, valid in ((0, True), (6, True), (7.999, True), (8, False), (8.001, False), (30, False), (None, False), (float('nan'), False)):
            self.assertEqual(drawing.can_draw([0, 0, .07], angle), valid)
        self.assertTrue(drawing.can_draw([0, 0, .08], 0))
        self.assertFalse(drawing.can_draw([0, 0, .08001], 0))
        self.assertFalse(drawing.can_draw([0, 0, -.001], 0))
        for step in range(10):
            self.assertFalse(drawing.update(step, [0, 0, .07], 9))
        self.assertEqual(drawing.phases['execution']['points'], [])
        self.assertTrue(drawing.update(10, [0, 0, .07], 7))
        self.assertEqual(len(drawing.phases['execution']['points']), 1)

    def test_wrong_angle_cannot_confirm_node_and_interrupts_dwell(self):
        dwell = PatternDwell([[0, 0]], [0])
        tip = [0, 0, .07]
        for step in range(10):
            dwell.update(step, tip, drawing_allowed=False)
        self.assertEqual(dwell.accepted, [])
        for step in range(10, 14):
            dwell.update(step, tip, drawing_allowed=True)
        dwell.update(14, tip, drawing_allowed=False)
        for step in range(15, 21):
            dwell.update(step, tip, drawing_allowed=True)
        self.assertFalse(dwell.success)
        dwell.update(21, tip, drawing_allowed=True)
        self.assertTrue(dwell.success)

    def test_red_activation_uses_physical_tip_axis_even_with_stale_contact(self):
        import torch
        from transforms3d.euler import euler2quat
        from robomme_plus.tasks.imitation.utils.tracing import update_feedback
        drawing = Drawing(max_angle_deg=8)
        drawing.begin('execution', 0)
        tip = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0, 0, .07]])))
        env = SimpleNamespace(_drawing=drawing, _stick_short=True, _stick_short_tip=tip,
                              elapsed_steps=np.array(1),
                              agent=SimpleNamespace(tcp=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0, 0, .07]])))),
                              buttons_grid=[object()], task_name='PatternLock', highlight_starts={},
                              _sampling={'positions': {}})
        with patch('robomme_plus.tasks.imitation.utils.tracing.highlight_obj') as red:
            for step, angle in ((1, 9), (2, 7)):
                env.elapsed_steps = np.array(step)
                tip.pose.q = torch.tensor([euler2quat(np.pi + np.radians(angle), 0, 0)])
                update_feedback(env, lambda *args, **kwargs: True, lambda *args, **kwargs: None)
                self.assertEqual(bool(env.highlight_starts), angle < 8)
            self.assertEqual(env.highlight_starts, {0: 2})
            self.assertEqual(red.call_count, 1)


if __name__ == '__main__':
    unittest.main()
