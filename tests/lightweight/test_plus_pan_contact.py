"""Scoop permits bottom contact while retaining its forbidden-contact rules."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robomme_plus.tasks.imitation.utils import geometry, tools


class PanContactTest(unittest.TestCase):
    def evidence_for_region(self, region):
        cube = SimpleNamespace(p=np.array([0., 0., .02]))
        head = SimpleNamespace(p=np.zeros(3))
        evidence = tools.empty_evidence()
        env = SimpleNamespace(
            _imitation_resetting=False, demonstration_record_traj=False,
            _imitation_phase='execution', _imitation_last_cube=cube.p.copy(),
            _imitation_evidence={'execution': evidence}, cube=cube,
            _imitation_head=head, _imitation_handle=object(), _imitation_shaft=object(),
            _imitation_wall=object(), _imitation_trace_step=-1,
            _imitation_config={'method': 'scoop'},
            _imitation_tool={'pan_length_m': .11, 'pan_width_m': .10},
            _imitation_direction=np.array([1., 0., 0.]), cube_half_size=.02,
            elapsed_steps=np.array(1),
            agent=SimpleNamespace(robot=SimpleNamespace(get_links=lambda: []),
                is_grasping=lambda actor: False, finger1_link=object(), finger2_link=object()),
        )
        with patch.object(tools, 'contact', side_effect=lambda e, a, b: b is head), \
             patch.object(tools, 'position', side_effect=lambda actor, initial=False: actor.p), \
             patch.object(tools, 'rotation', return_value=np.eye(3)), \
             patch.object(tools, 'tool_held', return_value=False), \
             patch.object(tools, 'pan_contacts', return_value={region}):
            tools.monitor(env)
        return evidence

    def test_underside_contact_is_allowed_and_not_an_automatic_success(self):
        region = geometry.pan_contact_region([0., 0., -.002], [0., 0., -1.], .11, .10)
        self.assertEqual(region, 'underside')
        evidence = self.evidence_for_region(region)
        self.assertFalse(geometry.move_failed('scoop', evidence))
        self.assertNotIn('first_illegal_contact', evidence)
        self.assertFalse(geometry.move_complete('scoop', evidence, True, True, True))

    def test_forbidden_rim_faces_still_fail(self):
        contacts = (
            ([0., .052, .012], [0., 1., 0.], 'outer rim'),
            ([0., .05, .028], [0., 0., 1.], 'rim top'),
            ([.055, .05, .012], [1., 0., 0.], 'rim front end'),
        )
        for point, normal, expected in contacts:
            with self.subTest(region=expected):
                region = geometry.pan_contact_region(point, normal, .11, .10)
                self.assertEqual(region, expected)
                evidence = self.evidence_for_region(region)
                self.assertTrue(geometry.move_failed('scoop', evidence))
                self.assertEqual(evidence['first_illegal_contact']['parts'], ['pan '+region])

    def test_cavity_contacts_remain_allowed(self):
        for point, normal in (([0., 0., 0.], [0., 0., 1.]),
                              ([0., .048, .012], [0., -1., 0.])):
            region = geometry.pan_contact_region(point, normal, .11, .10)
            self.assertFalse(geometry.move_failed('scoop', self.evidence_for_region(region)))

    def test_colored_pan_faces_match_the_contact_rules(self):
        assets = Path(tools.__file__).with_name('assets')
        parts = (
            ('floor', np.eye(3), np.ones(3), np.zeros(3)),
            ('rim', np.eye(3), np.array([.002, .05, .015]), np.array([-.053, 0., .013])),
            ('rim', np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
             np.array([.002, .055, .015]), np.array([0., -.05, .013])),
            ('rim', np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]),
             np.array([.002, .055, .015]), np.array([0., .05, .013])),
        )
        for part, rotation, scale, offset in parts:
            for surface in ('inner', 'underside', 'outer'):
                vertices, faces = [], []
                for line in (assets/f'imitation_pan_{part}_{surface}.obj').read_text().splitlines():
                    fields = line.split()
                    if fields and fields[0]=='v':
                        vertices.append([float(x) for x in fields[1:]])
                    elif fields and fields[0]=='f':
                        faces.append([int(x)-1 for x in fields[1:]])
                transformed = (np.array(vertices)*scale) @ rotation.T + offset
                for face in faces:
                    triangle = transformed[face]
                    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                    region = geometry.pan_contact_region(triangle.mean(axis=0), normal, .11, .10)
                    with self.subTest(part=part, surface=surface, region=region):
                        self.assertEqual(region in geometry.LEGAL_PAN_REGIONS, surface!='outer')


if __name__ == '__main__':
    unittest.main()
