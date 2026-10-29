"""Incline contact identification and preservation of loading-surface quotas."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from robomme_plus.tasks.imitation.move_cube_support import layout as l
from robomme_plus.tasks.imitation.move_cube_support.ramp_contact import ramp_contact_surfaces


def ramp_fixture(**changes):
    values = dict(seed=1, cube_xyz=(0, 0, .02), target_xyz=(.2, 0, .02),
                  target_heading_deg=0, cube_mass_kg=.05, ramp_toe_xyz=(-.1, 0, 0),
                  ramp_heading_deg=0, ramp_length_m=.22, ramp_width_m=.22,
                  ramp_grade_deg=25, pan_face="inclined_face", ball_radius=.02,
                  ball_mass=.2, ball_rest_xyz=(-.4, .3, .02), cue_pivot_xyz=(-.2, 0, 0),
                  cue_heading_deg=0, cue_park_deg=30, cue_front_m=.15, rim_label="test",
                  tool_poses={m: ((-.4, -.2 + i*.1, .02), 0) for i, m in enumerate(l.HANDHELD_METHODS)},
                  tool_params=copy.deepcopy(l.ORIGINAL_TOOLS), handle_spacing_m=.06)
    values.update(changes)
    return l.PrecisionLayout(**values)


class RampSurfaceContact(unittest.TestCase):
    def test_side_wall_quota_requires_actual_side_contact(self):
        from robomme_plus.tasks.imitation.move_cube_support.diversity import validate_reference_loading_surface
        from robomme_plus.tasks.imitation.move_cube_support.config import validate_sampling_plan
        from robomme_plus.tasks.imitation.move_cube_support.diversity import sampling_plan
        plan = sampling_plan(1, l.DEFAULT_RANGES['cube_mass_kg'])
        for phase in plan:
            plan[phase]['pan_loading_surface'] = 'side_wall'
        validate_sampling_plan(plan, l.DEFAULT_RANGES)
        cfg = dict(method='scoop', sampling_plan=plan)
        self.assertEqual(set(l.PAN_LOADING_FACES['side_wall']), {'left_side', 'right_side'})
        for phase in plan:
            valid = {'strict_evidence': {'ramp_surface_contact_steps': {'side_wall': 2}}}
            validate_reference_loading_surface(cfg, phase, valid)
            for contacts in ({'rear_wall': 50}, {'incline': 50}, {'side_wall': 1}):
                with self.assertRaises(RuntimeError):
                    validate_reference_loading_surface(cfg, phase, {'strict_evidence': {'ramp_surface_contact_steps': contacts}})

    def test_top_wall_and_toe_are_distinct_under_rotation_and_grade(self):
        for heading in (-140, -25, 0, 95, 180):
            for grade in (10, 25, 40):
                ramp = ramp_fixture(ramp_heading_deg=heading, ramp_grade_deg=grade)
                normal = ramp.ramp_direction * np.sin(ramp.ramp_angle) + [0, 0, np.cos(ramp.ramp_angle)]
                top = ramp.point(.6)
                self.assertEqual(ramp_contact_surfaces(ramp, top, normal), {"incline"})
                self.assertEqual(ramp_contact_surfaces(ramp, top, -normal), {"incline"})
                side = ramp.point(.6, ramp.ramp_width_m / 2, -.015)
                self.assertEqual(ramp_contact_surfaces(ramp, side, ramp.across), {"side_wall"})
                back = ramp.point(1, 0, -.015)
                self.assertEqual(ramp_contact_surfaces(ramp, back, ramp.ramp_direction), {"rear_wall"})
                self.assertEqual(ramp_contact_surfaces(ramp, ramp.toe, -ramp.ramp_direction), {"edge"})

    def test_blocked_incline_cannot_reroute_to_vertical_wall(self):
        ramp = ramp_fixture()
        original = l.PrecisionLayout.pan_site
        faces = []
        def track(instance):
            faces.append(instance.pan_face)
            return original(instance)
        with patch.object(l, "clear", return_value=False), patch.object(l.PrecisionLayout, "pan_site", track):
            with self.assertRaisesRegex(ValueError, "No clear pan route"):
                l.route_pan(ramp, [], np.zeros((4, 2)))
        self.assertTrue(faces)
        self.assertEqual(set(faces), {"inclined_face"})


if __name__ == "__main__":
    unittest.main()
