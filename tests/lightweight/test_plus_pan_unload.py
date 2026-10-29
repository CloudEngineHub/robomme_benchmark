"""Temporal regressions for carrying, airborne tilt release, and deposition."""

import math
import unittest
import numpy as np
from robomme_plus.tasks.imitation.move_cube_support.pan_unload import (
    PanUnloadTracker, underside_normal, underside_touches_target, pan_touches_target,
)
from robomme_plus.tasks.imitation.move_cube_support.pan import (
    record_pan_contacts, pan_evidence_is_legal,
)


class PanUnloadTest(unittest.TestCase):
    def test_all_bowl_faces_may_touch_cube(self):
        e = {"pan_contact_region_steps": {}, "outer_pan_contact_steps": 0}
        regions = {"floor", "inner rim", "outer rim", "rim top", "rim front end", "underside"}
        record_pan_contacts(e, regions, step=1, cube=np.zeros(3), head=np.zeros(3))
        self.assertTrue(pan_evidence_is_legal(e))
        e["shaft_contact_steps"] = 1
        self.assertFalse(pan_evidence_is_legal(e))

    def test_pan_rim_and_handle_cannot_touch_target_either(self):
        params = dict(pan_length_m=.12, pan_width_m=.10, tool_length_m=.25,
                      handle_half_size_m=[.025, .014, .01], pan_rim_height_m=.02)
        self.assertTrue(pan_touches_target([0, 0, .002], np.eye(3), params, [0, .051, 0], .005))
        self.assertTrue(pan_touches_target([0, 0, .002], np.eye(3), params, [-.25, 0, 0], .005))
        self.assertFalse(pan_touches_target([0, 0, .10], np.eye(3), params, [0, .051, 0], .005))

    def test_target_is_protected_even_between_contact_manifold_vertices(self):
        # All four corners lie outside the target, but the floor covers it.
        self.assertTrue(underside_touches_target([0, 0, 0.002], np.eye(3),
                                                0.12, 0.10, [0, 0, 0], 0.0267))
        self.assertFalse(underside_touches_target([0, 0, 0.08], np.eye(3),
                                                 0.12, 0.10, [0, 0, 0], 0.0267))
        self.assertFalse(underside_touches_target([0.3, 0, 0.002], np.eye(3),
                                                 0.12, 0.10, [0, 0, 0], 0.0267))

    def test_underside_normal_does_not_include_outer_side_faces(self):
        self.assertTrue(underside_normal([0.01, 0.02, -1]))
        self.assertFalse(underside_normal([1, 0, -0.01]))
        self.assertFalse(underside_normal([0, 0, 1]))

    def frame(self, tracker, *, pitch=0, x=0, local_x=0, contact=True,
              table=False, carried=True, held=True, head_z=0.08, head_contact=None):
        c, s = math.cos(pitch), math.sin(pitch)
        r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        head = np.array([x, 0, head_z])
        cube = head + r @ [local_x, 0, 0.023]
        if table:
            cube[2] = 0.02
        return tracker.update(dt=0.02, carried=carried, held=held,
                              contact=contact, head_contact=head_contact, table_contact=table,
                              head=head, rotation=r, cube=cube,
                              target=[0.15, 0, 0.02])

    def unload(self, tracker, *, x=0, table=False):
        self.frame(tracker, pitch=0.2)
        self.frame(tracker, pitch=0.5, x=x, local_x=0.055, table=table)
        for _ in range(3):
            self.frame(tracker, pitch=0.5, x=x, local_x=0.08, contact=False)
        return tracker.evidence

    def test_supported_carry_then_tilt_and_release(self):
        t = PanUnloadTracker()
        self.frame(t, x=-0.10)
        self.frame(t, x=0)
        e = self.unload(t)
        self.assertTrue(e["tilted_release"])
        self.assertEqual(e["max_horizontal_translation_m"], 0)
        self.frame(t, x=-0.08, contact=False, table=True)
        self.assertFalse(e["post_release_contact"])

    def test_translation_during_airborne_tilt_is_diagnostic(self):
        t = PanUnloadTracker()
        e = self.unload(t, x=0.02)
        self.assertTrue(e["tilted_release"])
        self.assertGreater(e["max_horizontal_translation_m"], 0.019)
        self.frame(t, contact=False, table=True)
        self.assertGreater(e["max_horizontal_translation_m"], 0.019)

    def test_final_push_even_before_tilting_is_recorded(self):
        t = PanUnloadTracker()
        self.frame(t)
        self.frame(t, table=True)
        e = self.frame(t, x=0.02, table=True)
        self.assertGreater(e["table_push_distance_m"], 0.019)

    def test_contact_after_releasing_is_permanent(self):
        t = PanUnloadTracker()
        self.unload(t)
        self.frame(t, pitch=0.5, contact=True, table=True)
        self.frame(t, contact=False)
        self.assertTrue(t.evidence["post_release_contact"])

    def test_non_head_brush_does_not_trigger_table_push_or_confirm_separation(self):
        t = PanUnloadTracker()
        self.frame(t, pitch=.2)
        self.frame(t, pitch=.5, local_x=.055)
        for x in (.003, .006, .009):
            self.frame(t, x=x, pitch=.5, table=True, head_contact=False)
        self.assertEqual(t.evidence["table_push_distance_m"], 0)
        self.assertEqual(t.free_seconds, 0)
        self.assertFalse(t.separated)
        self.frame(t, x=.012, pitch=.5, table=True, head_contact=True)
        self.assertGreater(t.evidence["table_push_distance_m"], .002)

    def test_recontact_is_diagnostic_but_pushing_after_release_is_forbidden(self):
        t = PanUnloadTracker()
        self.unload(t)
        self.frame(t, pitch=.5, contact=True, table=True)
        self.assertEqual(t.evidence["table_push_distance_m"], 0)
        self.frame(t, x=.02, pitch=.5, contact=True, table=True)
        self.assertGreater(t.evidence["table_push_distance_m"], .019)

    def test_flat_drop_is_not_tilt_unloading(self):
        t = PanUnloadTracker()
        self.frame(t)
        for _ in range(5):
            self.frame(t, contact=False, table=True)
        self.assertFalse(t.evidence["tilted_release"])

    def test_low_friction_pan_can_unload_before_ten_degrees(self):
        # Native replay: the cube clears the inclined floor around 7.3 degrees.
        # Require actual sliding and above-table separation, not a large angle.
        t = PanUnloadTracker()
        self.frame(t, pitch=math.radians(3.5), local_x=.025)
        self.frame(t, pitch=math.radians(7.28), local_x=.055)
        for _ in range(3):
            self.frame(t, pitch=math.radians(7.3), contact=False)
        self.assertTrue(t.evidence["tilted_release"])

    def test_sliding_onto_table_while_still_supported_is_not_free_deposition(self):
        self.assertFalse(self.unload(PanUnloadTracker(), table=True)["tilted_release"])

    def test_release_too_close_to_table_is_not_airborne_unloading(self):
        t = PanUnloadTracker()
        self.frame(t, pitch=0.2)
        self.frame(t, pitch=0.5, local_x=0.055, head_z=0.03)
        for _ in range(3):
            self.frame(t, pitch=0.5, contact=False)
        self.assertLess(t.evidence["release_cube_rise_m"], 0.008)
        self.assertFalse(t.evidence["tilted_release"])

    def test_airborne_release_requires_sustained_separation(self):
        t = PanUnloadTracker()
        self.frame(t, pitch=0.2)
        self.frame(t, pitch=0.5, x=0.02, local_x=0.055)
        self.frame(t, pitch=0.5, contact=False)
        self.assertFalse(t.evidence["tilted_release"])
        self.frame(t, pitch=0.5, local_x=0.055)  # contact flicker resets the gap
        self.frame(t, pitch=0.5, contact=False)
        self.assertFalse(t.evidence["tilted_release"])
        self.frame(t, pitch=0.5, contact=False)
        self.assertTrue(t.evidence["tilted_release"])

    def test_tilt_in_place_without_prior_carry_cannot_complete(self):
        t = PanUnloadTracker()
        for p in [0.2, 0.5, 0.8]:
            self.frame(t, pitch=p, carried=False)
        self.assertFalse(t.evidence["started"])


if __name__ == "__main__":
    unittest.main()
