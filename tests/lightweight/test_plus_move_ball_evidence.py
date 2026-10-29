"""Release decisions must survive grasp flicker without allowing real throws."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from robomme_plus.tasks.imitation.move_cube_support.evidence import StrictScene
from robomme_plus.tasks.imitation.move_cube_support.criteria import verdict, METHODS
from tests.lightweight.test_plus_move_success import successful


class BallReleaseEvidence(unittest.TestCase):
    def setUp(self):
        radius, angle = .02, .2
        point = [-.2, 0, (radius + .2 * np.sin(angle)) / np.cos(angle)]
        body = SimpleNamespace(entity="ball", linear_velocity=np.zeros(3), angular_velocity=np.zeros(3))
        contact = SimpleNamespace(
            bodies=[body, SimpleNamespace(entity="ramp")],
            points=[SimpleNamespace(impulse=np.array([0, 0, .01]))],
        )
        state = successful("ball")
        state["ball"] = {k: False if isinstance(v, bool) else 0.0 for k, v in state["ball"].items()}
        self.env = SimpleNamespace(
            strict=state, ball=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([point])), _bodies=[body]),
            ramp=SimpleNamespace(_bodies=[contact.bodies[1]]),
            layout=SimpleNamespace(toe=np.zeros(3), ramp_direction=np.array([1, 0, 0]),
                                   ramp_run=1., ramp_angle=angle, across=np.array([0, 1, 0]),
                                   ramp_width_m=.2, ball_radius=radius),
            scene=SimpleNamespace(px=SimpleNamespace(get_contacts=lambda: [contact])),
            time=1., _strict_ball_release=None, _strict_ball_separated=False,
            _strict_ball_separation_candidate=None, _strict_ball_last_robot_contact=-float("inf"),
            _strict_ball_hit=False, _strict_ball_ground_start=None,
            current_contacts=set(), events={"ball_held_hit": False}, impact_speed={"ball": .3},
            first_contact={},
            strict_event=lambda *args, **kwargs: None,
        )
        self.evidence = state["ball"]

    def sample(self, held=False, contact=False, count=1):
        for _ in range(count):
            self.env.time += .002
            StrictScene._monitor_ball(self.env, held, .002, contact)

    def test_fast_grasp_flicker_with_finger_contact_is_not_a_release(self):
        self.env.ball._bodies[0].linear_velocity = np.array([.2, 0, 0])
        self.sample(held=True, contact=True)
        self.sample(contact=True)
        self.sample(held=True, contact=True)
        self.assertFalse(self.evidence["launched"])
        self.assertIsNone(self.env._strict_ball_release)
        self.assertIsNone(self.env._strict_ball_separation_candidate)
        # The grasp detector can remain false until the fingers finally clear.
        self.sample(contact=True)
        self.env.ball._bodies[0].linear_velocity[:] = 0
        self.sample(count=11)
        self.assertTrue(self.evidence["released_on_ramp"])
        self.assertFalse(self.evidence["launched"])

    def test_short_contact_gap_is_discarded_before_a_gentle_release(self):
        self.sample(held=True, contact=True)
        self.env.ball._bodies[0].linear_velocity = np.array([.2, 0, 0])
        self.sample(count=5)
        self.assertFalse(self.evidence["launched"])
        self.assertIsNone(self.env._strict_ball_release)
        self.sample(contact=True)
        self.env.ball._bodies[0].linear_velocity[:] = 0
        self.sample(count=11)
        self.assertTrue(self.env._strict_ball_separated)
        self.assertFalse(self.evidence["launched"])
        self.assertAlmostEqual(self.evidence["ramp_after_release_s"], .022)

    def test_throw_or_spin_at_separation_cannot_be_hidden_by_deceleration(self):
        for attribute, velocity in (("linear_velocity", [.2, 0, 0]), ("angular_velocity", [10, 0, 0])):
            with self.subTest(attribute=attribute):
                self.setUp()
                self.sample(held=True, contact=True)
                setattr(self.env.ball._bodies[0], attribute, np.array(velocity))
                self.sample()
                setattr(self.env.ball._bodies[0], attribute, np.zeros(3))
                self.sample(count=10)
                self.assertTrue(self.evidence["launched"])
                self.assertIn("ball_launched_instead_of_released",
                              verdict("ball", METHODS, self.env.strict)["violations"])
                self.sample(held=True, contact=True)
                self.sample(count=11)
                self.assertTrue(self.evidence["launched"])
                self.assertTrue(self.evidence["robot_recontact_after_release"])

    def test_release_position_is_taken_at_start_of_separation(self):
        self.sample(held=True, contact=True)
        p = self.env.ball.pose.p
        p[0, 2] += .05
        self.sample()
        p[0, 2] -= .05
        self.sample(count=10)
        self.assertTrue(self.env._strict_ball_separated)
        self.assertFalse(self.evidence["released_on_ramp"])

    def test_confirmation_time_counts_toward_free_roll_and_recontact_is_diagnostic(self):
        self.sample(held=True, contact=True)
        self.env.ball._bodies[0].angular_velocity = np.array([0, 1., 0])
        self.sample(count=30)
        self.env.current_contacts = {"ball"}
        self.sample()
        self.assertAlmostEqual(self.evidence["free_before_hit_s"], .06)
        self.assertGreaterEqual(self.evidence["ramp_after_release_s"], .06)
        self.assertGreaterEqual(self.evidence["rolling_rotation_rad"], .06)
        self.sample(contact=True)
        self.assertTrue(self.evidence["robot_recontact_after_release"])
        self.assertFalse(verdict("ball", METHODS, self.env.strict)["failed"])

    def test_finger_brush_does_not_turn_existing_roll_into_a_throw(self):
        self.sample(held=True, contact=True)
        self.sample(count=30)
        release = self.env._strict_ball_release
        self.env.ball._bodies[0].linear_velocity = np.array([.3, 0, 0])
        self.sample(contact=True)
        last_contact = self.env.time
        self.sample(count=30)
        self.env.current_contacts = {"ball"}
        self.sample()
        self.assertIs(self.env._strict_ball_release, release)
        self.assertFalse(self.evidence["launched"])
        self.assertFalse(verdict("ball", METHODS, self.env.strict)["failed"])
        self.assertAlmostEqual(self.evidence["free_before_hit_s"], self.env.time - last_contact)

    def test_regrasp_discards_old_roll_and_measures_the_new_release(self):
        self.sample(held=True, contact=True)
        self.sample(count=30)
        self.env.current_contacts = {"ball"}
        self.env.first_contact["ball"] = self.env.time
        self.sample()
        self.assertTrue(self.evidence["free_hit"])
        self.env.current_contacts.clear()
        self.sample(held=True, contact=True)
        self.assertFalse(self.env._strict_ball_separated)
        self.assertIsNone(self.env._strict_ball_release)
        self.assertFalse(self.evidence["released_on_ramp"])
        self.assertFalse(self.evidence["free_hit"])
        self.assertEqual(self.evidence["ramp_after_release_s"], 0)
        self.assertNotIn("ball", self.env.first_contact)
        self.assertFalse(verdict("ball", METHODS, self.env.strict)["failed"])
        # A new gentle release earns only its own free-roll duration.
        self.sample(count=11)
        self.assertTrue(self.env._strict_ball_separated)
        self.assertTrue(self.evidence["released_on_ramp"])
        self.assertAlmostEqual(self.evidence["ramp_after_release_s"], .022)
        self.env.impact_speed["ball"] = .09
        self.sample(count=20)
        self.env.current_contacts = {"ball"}
        self.sample()
        self.assertAlmostEqual(self.evidence["impact_speed_mps"], .09)

    def test_regrasp_after_gentle_release_cannot_hide_a_later_throw(self):
        self.sample(held=True, contact=True)
        self.sample(count=11)
        self.sample(held=True, contact=True)
        self.env.ball._bodies[0].linear_velocity = np.array([.2, 0, 0])
        self.sample()
        self.env.ball._bodies[0].linear_velocity[:] = 0
        self.sample(count=10)
        self.assertIn("ball_launched_instead_of_released",
                      verdict("ball", METHODS, self.env.strict)["violations"])

    def test_ungrasped_ball_never_acquires_release_evidence(self):
        self.sample(count=30)
        self.assertFalse(self.env._strict_ball_separated)
        self.assertFalse(self.evidence["released_on_ramp"])


if __name__ == "__main__":
    unittest.main()
