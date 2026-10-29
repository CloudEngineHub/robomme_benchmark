"""Seeded swing direction agrees with goals, subgoals, dodges and scoring."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robomme_plus.tasks.counting.utils.controllers import PROFILES, ReactiveSwing
from robomme_plus.tasks.counting.utils.goals import counting_goal
from robomme_plus.tasks.counting.utils.subgoals import swingxtimes
from robomme_plus.tasks.counting.utils.variations import sample_variation, swing_start_side
from robomme_plus.tasks.counting.utils.motion import GridEvasions


MODULE = "robomme_plus.tasks.counting.utils.controllers"


class SwingDirectionTest(unittest.TestCase):
    def test_seed_selects_both_directions_reproducibly_at_every_level(self):
        sides = [swing_start_side(seed) for seed in range(1000)]
        self.assertGreater(sides.count("left"), 400)
        self.assertLess(sides.count("left"), 600)
        for level in ("easy", "medium", "hard"):
            for seed in range(100):
                self.assertEqual(sample_variation("SwingXtimes", seed, level)["start_side"], sides[seed])
                self.assertEqual(swing_start_side(seed), sides[seed])

    def test_goal_names_the_first_side_and_repeated_direction(self):
        for first, second in (("left", "right"), ("right", "left")):
            goal = counting_goal("SwingXtimes", color="blue", count=3, start_side=first)
            self.assertIn(f"move it to the {first}-side target and then to the {second}-side target", goal)
            self.assertIn(f"repeating this {first}-to-{second} swing motion three times", goal)

    def controller(self, side, level):
        seed = next(seed for seed in range(100) if swing_start_side(seed) == side)
        actor = lambda: SimpleNamespace(p=np.zeros(3))
        env = SimpleNamespace(
            seed=seed, swing_start_side=side, target_color_name="blue",
            target_right=actor(), target_left=actor(), target_cube=actor(),
            button=actor(), non_target_cubes=[], cube_half_size=.02,
            elapsed_steps=np.array(0), agent=SimpleNamespace(tcp=actor()),
        )
        controller = ReactiveSwing(
            env, dict(PROFILES["swingxtimes_reactive"], level=level),
            SimpleNamespace(highlight_obj=lambda *args, **kwargs: None),
        )
        controller.control_step = 0
        controller.select_target_color = lambda: None
        controller.grasped = lambda cube: True
        controller.stop_with_button = lambda ready: None
        controller.flags = lambda: {}
        controller.reset()
        # Keep the hand clear to isolate visit scoring from evasive movement.
        env.agent.tcp.p = np.array([1., 1., .2])
        return controller

    def setUp(self):
        self.position = patch(
            MODULE + ".position", side_effect=lambda actor, initial=False: actor.p.copy()
        )
        self.place = patch(
            MODULE + ".place",
            side_effect=lambda actor, xyz, **kwargs: setattr(actor, "p", np.array(xyz)),
        )
        self.position.start()
        self.place.start()
        self.addCleanup(self.position.stop)
        self.addCleanup(self.place.stop)

    def visit(self, controller, index):
        controller.control_step += 1
        controller.env.elapsed_steps = np.array(controller.control_step)
        controller.env.target_cube.p = controller.targets[index].p.copy()
        controller.env.target_cube.p[2] = .08
        controller.evaluate()

    def test_actual_first_target_gets_the_guaranteed_dodge_and_subgoal(self):
        for level, maximum in (("medium", 2), ("hard", 4)):
            for side, index in (("right", 0), ("left", 1)):
                with self.subTest(level=level, side=side):
                    controller = self.controller(side, level)
                    self.assertEqual(controller.expected, index)
                    self.assertIn(controller.slides.budget(index), range(1, maximum + 1))
                    for row in controller.dodge_budgets:
                        self.assertTrue(all(0 <= budget <= maximum for budget in row))
                    name, _, target = swingxtimes(controller)
                    self.assertIn(f"{side}-side target", name)
                    self.assertIs(target, controller.targets[index])
                    self.assertIn(f"move it to the {side}-side target", controller.goal)

    def test_scoring_alternates_from_the_chosen_side_for_complete_cycles(self):
        for side, first in (("right", 0), ("left", 1)):
            controller = self.controller(side, "medium")
            for visit in range(2 * controller.env.num_repeats):
                target = first if visit % 2 == 0 else 1 - first
                self.visit(controller, target)
                self.assertIsNone(controller.failure)
                self.assertEqual(controller.count, visit + 1)
                self.assertEqual(controller.expected, 1 - target)

    def test_opposite_start_is_rejected(self):
        for side, wrong_first in (("right", 1), ("left", 0)):
            controller = self.controller(side, "hard")
            self.visit(controller, wrong_first)
            self.assertEqual(controller.count, 0)
            self.assertEqual(controller.failure, "wrong_target_order_or_extra_swing")

    def test_dodge_passing_stationary_cube_does_not_count(self):
        controller = self.controller('right', 'medium')
        origins = np.array([[0., 0., .01], [-.3, 0., .01]])
        controller.slides = GridEvasions(origins, seed=0, max_evades=1)
        hand = np.array([.033 * np.sqrt(3) / 2, -.033 / 2, .07])
        controller.env.target_cube.p = hand.copy()
        controller.env.agent.tcp.p = hand.copy()
        for step in range(5):
            controller.control_step = step
            controller.before_step(step)
            controller.evaluate()
        self.assertEqual(controller.count, 0)
        self.assertIsNone(controller.failure)
        self.assertEqual(controller.slides.events[0]['direction'], 'right')

    def test_fast_approach_waits_for_dodge_then_tracking_can_count(self):
        controller = self.controller('right', 'medium')
        origins = np.array([[0., 0., .01], [-.3, 0., .01]])
        controller.slides = GridEvasions(origins, seed=0, max_evades=1)
        for step in range(5):
            controller.control_step = step
            controller.before_step(step)
            hand = controller.targets[0].p.copy()
            hand[2] = .07
            controller.env.target_cube.p = hand.copy()
            controller.env.agent.tcp.p = hand.copy()
            controller.evaluate()
            self.assertEqual(controller.count, int(step == 4))
            self.assertIsNone(controller.failure)


if __name__ == "__main__":
    unittest.main()
