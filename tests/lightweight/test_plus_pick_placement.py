"""PickXtimes requires a lifted release followed by an open-gripper retreat."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robomme_plus.tasks.counting.utils.controllers import MovingPlacement
from robomme_plus.tasks.counting.utils.motion import PausableClock, ReleaseCounter


class PlacementCounterTest(unittest.TestCase):
    def test_settled_release_never_counts_without_five_cm_upward_retreat(self):
        counter = ReleaseCounter(2)
        counter.update(0, True, True, False, False, gripper_open=False, gripper_z_m=.2)
        counter.update(1, False, False, True, True, gripper_open=True, gripper_z_m=.03)
        for step in range(2, 100):
            self.assertIsNone(counter.update(step, False, False, True, True,
                                             gripper_open=True, gripper_z_m=.03))
            self.assertTrue(counter.pause_ready)
        self.assertEqual(counter.count, 0)
        self.assertIsNone(counter.update(100, False, False, True, True,
                                         gripper_open=True, gripper_z_m=.079))
        self.assertEqual(counter.update(101, False, False, True, True,
                                        gripper_open=True, gripper_z_m=.08), 'placement')
        for step in range(102, 110):
            counter.update(step, False, False, True, True, gripper_open=True, gripper_z_m=.2)
        self.assertEqual(counter.count, 1)

    def test_retreat_is_measured_from_each_release_and_requires_open_fingers(self):
        counter = ReleaseCounter(2)
        for base, release_z in ((0, .02), (20, .12)):
            counter.update(base, True, True, False, False, gripper_open=False, gripper_z_m=.2)
            counter.update(base + 1, False, False, True, True,
                           gripper_open=True, gripper_z_m=release_z)
            self.assertIsNone(counter.update(base + 5, False, False, True, True,
                                             gripper_open=True, gripper_z_m=release_z + .049))
            self.assertIsNone(counter.update(base + 6, False, False, True, True,
                                             gripper_open=False, gripper_z_m=release_z + .06))
            counter.update(base + 7, False, False, True, True,
                           gripper_open=True, gripper_z_m=release_z + .06)
            self.assertEqual(counter.update(base + 11, False, False, True, True,
                                            gripper_open=True, gripper_z_m=release_z + .06), 'placement')
        self.assertEqual(counter.count, 2)

    def test_idle_cube_and_low_regrasps_never_count(self):
        counter = ReleaseCounter(2)
        for step in range(50):
            counter.update(step, step % 10 < 5, False, True, True,
                           gripper_open=step % 10 >= 5, gripper_z_m=.02)
            self.assertFalse(counter.pause_ready)
        self.assertEqual(counter.count, 0)

    def test_descent_and_release_pause_before_settling_counts_once(self):
        counter = ReleaseCounter(1)
        counter.update(0, True, True, False, False, gripper_open=False, gripper_z_m=.02)
        counter.update(1, True, False, True, False, gripper_open=False, gripper_z_m=.02)
        self.assertIsNone(counter.update(2, False, False, True, True, gripper_open=True, gripper_z_m=.02))
        self.assertTrue(counter.pause_ready)
        for step in range(3, 6):
            self.assertIsNone(counter.update(step, False, False, True, True, gripper_open=True, gripper_z_m=.02))
        self.assertEqual(counter.update(6, False, False, True, True, gripper_open=True, gripper_z_m=.08), "placement")
        for step in range(7, 30):
            counter.update(step, False, False, True, True, gripper_open=True, gripper_z_m=.02)
        self.assertEqual(counter.count, 1)

    def test_off_target_release_cannot_be_rescued_by_target_passing_underneath(self):
        counter = ReleaseCounter(1)
        counter.update(0, True, True, False, False, gripper_open=False, gripper_z_m=.02)
        counter.update(1, False, False, False, True, gripper_open=True, gripper_z_m=.02)
        for step in range(2, 10):
            counter.update(step, False, False, True, True, gripper_open=True, gripper_z_m=.02)
            self.assertFalse(counter.pause_ready)
        self.assertEqual((counter.count, counter.misses), (0, 1))

    def test_low_moves_and_lost_alignment_while_held_do_not_invalidate_release(self):
        counter = ReleaseCounter(1)
        counter.update(0, True, True, False, False, gripper_open=False, gripper_z_m=.02)
        for step, inside in enumerate((False, True, False, True), 1):
            counter.update(step, True, False, inside, True, gripper_open=False, gripper_z_m=.02)
            self.assertFalse(counter.pause_ready)
        counter.update(5, False, False, True, True, gripper_open=True, gripper_z_m=.02)
        self.assertTrue(counter.pause_ready)
        self.assertEqual(counter.update(9, False, False, True, True, gripper_open=True, gripper_z_m=.08), "placement")
        self.assertEqual(counter.count, 1)

    def test_lost_contact_with_closed_gripper_does_not_release_or_count(self):
        counter = ReleaseCounter(1)
        counter.update(0, True, True, True, False, gripper_open=False, gripper_z_m=.02)
        for step in range(1, 20):
            counter.update(step, False, False, True, True, gripper_open=False, gripper_z_m=.02)
            self.assertFalse(counter.pause_ready)
        self.assertEqual((counter.count, counter.misses, counter.released_at), (0, 0, None))

    def test_alignment_lost_after_release_prevents_counting(self):
        counter = ReleaseCounter(1)
        counter.update(0, True, True, False, False, gripper_open=False, gripper_z_m=.02)
        counter.update(1, False, False, True, True, gripper_open=True, gripper_z_m=.02)
        counter.update(2, False, False, False, True, gripper_open=True, gripper_z_m=.02)
        self.assertFalse(counter.pause_ready)
        self.assertEqual(counter.update(5, False, False, True, True, gripper_open=True, gripper_z_m=.02), "miss")

    def test_regrasp_requires_new_lift_and_airborne_bounce_restarts_settling(self):
        counter = ReleaseCounter(2)
        counter.update(0, True, True, True, False, gripper_open=False, gripper_z_m=.02)
        counter.update(1, False, False, True, True, gripper_open=True, gripper_z_m=.02)
        counter.update(2, True, False, True, True, gripper_open=False, gripper_z_m=.02)
        for step in range(3, 15):
            counter.update(step, False, False, True, True, gripper_open=True, gripper_z_m=.02)
        self.assertEqual(counter.count, 0)
        counter.update(20, True, True, True, False, gripper_open=False, gripper_z_m=.02)
        counter.update(21, False, False, True, True, gripper_open=True, gripper_z_m=.02)
        counter.update(24, False, False, True, False, gripper_open=True, gripper_z_m=.02)
        for step in range(25, 29):
            self.assertIsNone(counter.update(step, False, False, True, True, gripper_open=True, gripper_z_m=.02))
        self.assertEqual(counter.update(29, False, False, True, True, gripper_open=True, gripper_z_m=.08), "placement")

    def test_miss_can_be_retried_but_excess_placements_fail(self):
        counter = ReleaseCounter(1)
        for base, inside in ((0, False), (10, True), (20, True)):
            counter.update(base, True, True, False, False, gripper_open=False, gripper_z_m=.02)
            counter.update(base + 1, False, False, inside, True, gripper_open=True, gripper_z_m=.02)
            counter.update(base + 5, False, False, inside, True, gripper_open=True, gripper_z_m=.08)
        self.assertEqual((counter.count, counter.misses, counter.failure),
                         (2, 1, "too_many_placements"))


class PlacementPauseTest(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(
            cube_half_size=0.02, target_cube=SimpleNamespace(p=np.array([0., 0., .02])),
            target=SimpleNamespace(p=np.array([0., 0., .005])),
            non_target_cubes=[], held=False, num_repeats=3, elapsed_steps=np.array(0),
        )
        self.env.opened, self.env.open_command = True, True
        self.env.agent = SimpleNamespace(robot=SimpleNamespace(
            get_qpos=lambda: np.array([[0.] * 7 + [0.04 if self.env.opened else 0.02] * 2]),
            get_drive_targets=lambda: np.array([[0.] * 7 + [0.04 if self.env.open_command else -0.01] * 2]),
        ), tcp=SimpleNamespace(p=np.array([0., 0., .02])))
        self.controller = MovingPlacement.__new__(MovingPlacement)
        self.controller.env = self.env
        self.controller.config = {"min_lift_m": .05, "landing_clearance_m": .015}
        self.controller.counter = ReleaseCounter(3)
        self.controller.motion_clock = PausableClock()
        self.controller.events = []
        self.controller.success, self.controller.failure = False, None
        self.controller.grasped = lambda cube: self.env.held
        self.controller.stop_with_button = lambda ready: None
        self.controller.flags = lambda: {}
        self.step = 0

    def observe(self, height, held=False, inside=True, *, opened=None, open_command=None,
                gripper_height=None):
        self.step += 1
        self.env.elapsed_steps = np.array(self.step)
        self.controller.control_step = self.step
        self.env.held = held
        self.env.opened = not held if opened is None else opened
        self.env.open_command = not held if open_command is None else open_command
        self.env.target_cube.p = np.array([0. if inside else .15, 0., height])
        self.env.agent.tcp.p = np.array([0., 0., height if gripper_height is None else gripper_height])
        with patch('robomme_plus.tasks.counting.utils.controllers.position',
                   side_effect=lambda actor, initial=False: actor.p.copy()):
            self.controller.evaluate()

    def test_idle_cube_on_route_cannot_stop_target(self):
        for _ in range(50):
            self.observe(.02)
        self.assertFalse(self.controller.motion_clock.paused)
        self.assertEqual(self.controller.counter.count, 0)

    def test_each_pause_and_count_needs_five_cm_lift_above_resting_height(self):
        self.observe(.06, held=True)
        for _ in range(6):
            self.observe(.025)
        self.assertFalse(self.controller.motion_clock.paused)
        self.observe(.10, held=True, inside=False)
        self.observe(.05, held=True)
        self.observe(.025)
        self.assertTrue(self.controller.motion_clock.paused)
        for _ in range(4):
            self.observe(.025, gripper_height=.085)
        self.assertEqual(self.controller.counter.count, 1)
        self.observe(.06, held=True)
        self.assertFalse(self.controller.motion_clock.paused)
        for _ in range(6):
            self.observe(.025)
        self.assertFalse(self.controller.motion_clock.paused)
        self.assertEqual(self.controller.counter.count, 1)

    def test_low_horizontal_moves_are_allowed_but_pause_requires_open_release(self):
        self.observe(.12, held=True, inside=False)
        self.observe(.025, held=True, inside=False)
        self.observe(.025, held=True, inside=True)
        for _ in range(6):
            self.observe(.025, held=True)
        self.assertFalse(self.controller.motion_clock.paused)
        self.assertEqual(self.controller.counter.count, 0)
        self.observe(.025)
        self.assertTrue(self.controller.motion_clock.paused)
        for _ in range(4):
            self.observe(.025, gripper_height=.085)
        self.assertEqual(self.controller.counter.count, 1)

    def test_fast_long_descent_can_pause_and_count_on_release(self):
        self.observe(.25, held=True, inside=False)
        self.observe(.025, held=True, inside=False)
        self.observe(.025)
        self.assertTrue(self.controller.motion_clock.paused)
        for _ in range(4):
            self.observe(.025, gripper_height=.085)
        self.assertEqual(self.controller.counter.count, 1)

    def test_opening_while_cube_is_high_does_not_pause_or_count(self):
        self.observe(.12, held=True)
        for _ in range(8):
            self.observe(.10)
        self.assertFalse(self.controller.motion_clock.paused)
        self.assertEqual(self.controller.counter.count, 0)
        self.observe(.025)
        self.assertTrue(self.controller.motion_clock.paused)
        for _ in range(4):
            self.observe(.025, gripper_height=.16)
        self.assertEqual(self.controller.counter.count, 1)

    def test_target_pauses_on_release_but_count_waits_for_retreat(self):
        self.observe(.12, held=True)
        for _ in range(20):
            self.observe(.025)
        self.assertTrue(self.controller.motion_clock.paused)
        self.assertEqual(self.controller.counter.count, 0)
        self.observe(.025, gripper_height=.074)
        self.assertEqual(self.controller.counter.count, 0)
        self.observe(.025, gripper_height=.075)
        self.assertEqual(self.controller.counter.count, 1)

    def test_both_open_command_and_open_fingers_are_required(self):
        for opened, command in ((False, False), (True, False), (False, True)):
            with self.subTest(opened=opened, command=command):
                self.setUp()
                self.observe(.12, held=True)
                for _ in range(8):
                    self.observe(.025, opened=opened, open_command=command)
                self.assertFalse(self.controller.motion_clock.paused)
                self.assertEqual(self.controller.counter.count, 0)


if __name__ == '__main__':
    unittest.main()
