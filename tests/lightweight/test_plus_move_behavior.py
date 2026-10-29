"""Regressions for incidental contact versus actually using a movement method."""

import unittest
import numpy as np

from robomme_plus.tasks.imitation.move_cube_support.behavior import (
    ToolLiftTracker, hook_inner_contact, cue_point_velocity,
)
from tests.lightweight.test_plus_move_success import successful
from robomme_plus.tasks.imitation.move_cube_support.criteria import verdict, METHODS


class MethodBehaviorTest(unittest.TestCase):
    def test_grasp_or_table_bump_does_not_use_a_tool(self):
        t = ToolLiftTracker()
        for _ in range(200):
            t.update("hook", [.001, .001], .002)
            t.update("scoop", [.08, 0], .002)  # tipping one end on table
            t.update("cue", [1], .002)
        self.assertEqual(t.used, set())

    def test_sustained_lift_is_use_and_is_latched(self):
        t = ToolLiftTracker()
        for _ in range(49):
            t.update("hook", [.05, .05], .002)
        self.assertNotIn("hook", t.used)
        t.update("hook", [.05, .05], .002)
        t.update("hook", [0, 0], .002)
        self.assertIn("hook", t.used)
        for _ in range(60):
            t.update("ball", [.06], .002)
        self.assertIn("ball", t.used)

    def test_short_hops_do_not_accumulate_into_use(self):
        t = ToolLiftTracker()
        for _ in range(100):
            t.update("poke", [.05, .05], .02)
            t.update("poke", [0, 0], .02)
        self.assertFalse(t.used)

    def test_repeated_pickups_of_the_required_tool_are_allowed(self):
        for method in METHODS:
            with self.subTest(method=method):
                tracker = ToolLiftTracker()
                state = successful(method)
                for _ in range(3):
                    for rises in ([.06, .06], [0., 0.]):
                        for _ in range(60):
                            tracker.update(method, rises, .002)
                        state["tools_used"] = sorted(tracker.used)
                        state["tool_max_displacement_m"] = {method: .3}
                        self.assertTrue(verdict(method, METHODS, state)["success"])
                self.assertEqual(tracker.used, set() if method == "cue" else {method})

    def test_regrasping_correct_tool_does_not_clear_wrong_tool_use(self):
        for required in METHODS:
            for other in METHODS:
                if other in (required, "cue"):
                    continue
                with self.subTest(required=required, other=other):
                    tracker = ToolLiftTracker()
                    state = successful(required)
                    for method in (required, other, required):
                        for _ in range(60):
                            tracker.update(method, [.06, .06], .002)
                        tracker.update(method, [0., 0.], .002)
                    state["tools_used"] = sorted(tracker.used)
                    self.assertIn("another_tool_was_used",
                                  verdict(required, METHODS, state)["violations"])

    def test_wrong_method_propulsion_fails_for_every_required_method(self):
        for required in METHODS:
            with self.subTest(required=required):
                state = successful(required)
                state["wrong_tool_progress_m"] = .003
                self.assertIn("cube_moved_by_another_tool",
                              verdict(required, METHODS, state)["violations"])

    def test_hook_inner_crossbar_and_return_faces(self):
        self.assertTrue(hook_inner_contact([-.006, 0, 0], [-1, 0, 0]))
        self.assertTrue(hook_inner_contact([-.025, .033, 0], [0, -1, 0]))
        self.assertTrue(hook_inner_contact([-.025, -.033, 0], [0, 1, 0]))
        self.assertFalse(hook_inner_contact([.006, 0, 0], [1, 0, 0]))
        self.assertFalse(hook_inner_contact([-.025, .041, 0], [0, 1, 0]))
        self.assertFalse(hook_inner_contact([-.006, 0, -.008], [0, 0, -1]))

    def test_rotational_cue_impact_can_be_dynamic(self):
        v = cue_point_velocity(0, 2, [1, 0, 0], [.2, 0, 0])
        np.testing.assert_allclose(v, [0, .4, 0])
        self.assertLess(np.linalg.norm(cue_point_velocity(.01, .01, [1, 0, 0], [.2, 0, 0])), .08)

    def test_failure_flags_from_old_pan_and_cue_contacts_are_not_reused(self):
        for method in ("hook", "scoop", "cue"):
            s = successful(method)
            if method == "cue":
                s["cue"].update(non_tip_contact=True, swing_hit=True, held_tip_hit=False)
            else:
                s["handheld"][method].update(unheld_cube_contact=True, illegal_surface_contact=True)
            s["final_error_m"] = .20
            result = verdict(method, METHODS, s)
            self.assertFalse(result["failed"])
            self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
