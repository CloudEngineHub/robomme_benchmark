"""The straight approach stays on a physically facing cube-to-ramp line."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robomme_plus.tasks.imitation.move_cube_support.pan_straight import approach_geometry
from robomme_plus.tasks.imitation.move_cube_support import oracle, pan_straight


class StraightPanGeometryTests(unittest.TestCase):
    def test_explicit_straight_mode_skips_the_old_multileg_route(self):
        env, planner = Mock(), Mock()
        expected = {'success': True}
        with patch.object(pan_straight, 'solve_straight_scoop', return_value=expected) as solve, \
             patch.object(oracle, 'position_cube_with_pan') as routed:
            result = oracle.solve_scoop(env, planner, approach_mode='straight',
                                       carry_pitch=0., loading_pitch=.14,
                                       carry_yaw_deg=81.1, carry_turn_fraction=.45)
            self.assertIs(result, expected)
            routed.assert_not_called()
            self.assertEqual(solve.call_args.kwargs['carry_height_m'], .10)
            self.assertEqual(solve.call_args.kwargs['loading_pitch'], .14)
            self.assertEqual(solve.call_args.kwargs['carry_yaw_deg'], 81.1)
            self.assertEqual(solve.call_args.kwargs['carry_turn_fraction'], .45)

    def test_rotated_approach_has_one_heading_and_keeps_cube_lateral_position(self):
        for angle in (0., 0.7, 2.1, -2.8):
            direction = np.array([np.cos(angle), np.sin(angle), 0.])
            across = np.array([-direction[1], direction[0], 0.])
            site = np.array([.1, -.1, 0.])
            cube = site - .18 * direction + .025 * across + [0., 0., .02]
            layout = SimpleNamespace(pan_face='inclined_face', ramp_width_m=.18,
                                     pan_site=lambda: (site, direction))
            plan = approach_geometry(layout, cube, np.eye(3), {'pan_length_m': .08})
            for key in ('start', 'end', 'above', 'contact'):
                self.assertAlmostEqual(float((plan[key] - cube) @ across), 0., places=12)
            self.assertGreater(float((cube-plan['start']) @ direction), .05)
            self.assertGreater(float((plan['end']-plan['start']) @ direction), .18)
            np.testing.assert_allclose(plan['flat'][:, 0], direction)

    def test_far_side_assignment_is_rejected_before_a_detour_or_motion(self):
        layout = SimpleNamespace(pan_face='right_side', ramp_width_m=.18,
                                 pan_site=lambda: (np.zeros(3), np.array([1., 0., 0.])))
        with self.assertRaisesRegex(RuntimeError, 'does not face'):
            approach_geometry(layout, [.1, 0., .02], np.eye(3), {'pan_length_m': .08})

    def test_cube_rotation_changes_clearance_but_never_pan_heading(self):
        layout = SimpleNamespace(pan_face='inclined_face', ramp_width_m=.18,
                                 pan_site=lambda: (np.zeros(3), np.array([1., 0., 0.])))
        first = approach_geometry(layout, [-.18, .02, .02], np.eye(3), {'pan_length_m': .08})
        a = .7
        rotated_cube = np.array([[np.cos(a), -np.sin(a), 0.],
                                 [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        second = approach_geometry(layout, [-.18, .02, .02], rotated_cube, {'pan_length_m': .08})
        np.testing.assert_allclose(first['direction'], second['direction'])
        np.testing.assert_allclose(first['flat'], second['flat'])
        np.testing.assert_allclose(first['contact'], second['contact'])
        self.assertLess(second['start'][0], first['start'][0])

    def test_cube_footprint_must_fit_the_facing_incline(self):
        layout = SimpleNamespace(pan_face='inclined_face', ramp_width_m=.10,
                                 pan_site=lambda: (np.zeros(3), np.array([1., 0., 0.])))
        with self.assertRaisesRegex(RuntimeError, 'usable incline width'):
            approach_geometry(layout, [-.18, .04, .02], np.eye(3), {'pan_length_m': .08})


if __name__ == '__main__':
    unittest.main()
