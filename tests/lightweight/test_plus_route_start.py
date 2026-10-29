"""Execution must touch RouteStick's start with the physical drawing tip."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robomme_plus.common.sim import position
from robomme_plus.tasks.imitation.route_stick import RouteStick


def tip_actor():
    return SimpleNamespace(pose=SimpleNamespace(
        p=torch.tensor([[0., 0., .2]]), q=torch.tensor([[0., 1., 0., 0.]])
    ))


class RouteStartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = RouteStick(seed=0, difficulty="easy", obs_mode="state", sim_backend="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        env = self.env
        env.reset(seed=0)
        # Supply a reference curve for each segment without replaying an episode.
        nodes = [position(node, False)[:2] for node in env.selected_buttons]
        for curve, (start, end) in enumerate(zip(nodes, nodes[1:])):
            delta = end - start
            normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
            for offset, point in enumerate((start, (start + end) / 2 + .1 * normal, end)):
                env._route_trace.update("demonstration", curve, curve * 3 + offset, point)
        env.demonstration_record_traj = True
        env.demonstration_record_traj = False
        env.use_demonstrationwrapper = True
        env.current_task_demonstration = False
        env.timestep = self.first = next(
            i for i, entry in enumerate(env.task_list) if not entry["demonstration"]
        )
        env._gripper_xy_trace = []
        self.tcp, self.short_tip = tip_actor(), tip_actor()
        for name, value in (("agent", SimpleNamespace(tcp=self.tcp)),
                            ("_stick_short_tip", self.short_tip)):
            replacement = patch.object(env, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def move(self, xy, z=.07, *, short=False):
        env = self.env
        env._stick_short = short
        tip = self.short_tip if short else self.tcp
        tip.pose.p[0] = torch.tensor([*xy, z])
        env._elapsed_steps += 1
        return env.evaluate()

    def geometry(self):
        env = self.env
        start, end = [position(node, False)[:2] for node in env.selected_buttons[:2]]
        delta = end - start
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        if env.swing_directions[0] == "counterclockwise":
            normal = -normal
        return start, end, normal

    def test_skipping_start_cannot_complete_first_segment(self):
        start, end, normal = self.geometry()
        point = end - .01 * (end - start) / np.linalg.norm(end - start) + .005 * normal
        result = self.move(point)
        self.assertEqual(self.env.timestep, self.first)
        self.assertFalse(result["success"].item())
        self.assertEqual(self.env.current_task_name, "move to the starting target")

    def test_touching_second_node_before_start_fails(self):
        _, end, normal = self.geometry()
        result = self.move(end + .005 * normal)
        self.assertEqual(self.env.timestep, self.first)
        self.assertTrue(result["fail"].item())
        self.assertEqual(self.env._failure_code, "wrong_drawing_target")

    def test_hidden_board_anchor_is_not_an_unexpected_route_target(self):
        env = self.env
        self.assertFalse(env._imitation_config["obstacle_contact_fails"])
        board_index = env.target_cube_indices[0]
        self.assertNotIn(board_index, env.route_button_indices)
        point = position(env.buttons_grid[board_index], False)[:2]
        result = self.move(point)
        self.assertFalse(result["fail"].item())
        self.assertFalse(result["success"].item())
        self.assertEqual(env.timestep, self.first)

    def test_valid_start_discards_approach_and_allows_complete_route(self):
        env = self.env
        start, end, normal = self.geometry()
        # Approach from the opposite side while above drawing height.
        for _ in range(20):
            self.move((start + end) / 2 - normal, z=.2)
        result = self.move(start)
        self.assertFalse(result["fail"].item())
        self.assertEqual(env.timestep, self.first + 1)
        self.assertEqual(env._gripper_xy_trace, [])
        self.assertEqual(env._route_curve_tasks["execution"][env.timestep], 0)
        for curve, (a, b) in enumerate(zip(env.selected_buttons, env.selected_buttons[1:])):
            start, end = position(a, False)[:2], position(b, False)[:2]
            delta = end - start
            normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
            if env.swing_directions[curve] == "counterclockwise":
                normal = -normal
            for fraction in np.linspace(0, 1, 25)[1:]:
                point = start + fraction * delta + .4 * fraction * (1 - fraction) * normal
                result = self.move(point)
                self.assertFalse(result["fail"].item())
            self.assertEqual(env.timestep, self.first + curve + 2)
        self.assertTrue(result["success"].item())
        self.assertIsNone(env._drawing.failure)
        self.assertIsNone(env._route_trace.failure)

    def test_start_uses_short_tip_and_requires_drawing_height(self):
        env = self.env
        start, _, _ = self.geometry()
        # TCP on the target is insufficient when the real shortened tip is high.
        self.tcp.pose.p[0] = torch.tensor([*start, .07])
        self.move(start, z=.081, short=True)
        self.assertEqual(env.timestep, self.first)
        self.move(start, z=-.001, short=True)
        self.assertEqual(env.timestep, self.first)
        self.tcp.pose.p[0, 2] = .2
        result = self.move(start, short=True)
        self.assertFalse(result["fail"].item())
        self.assertEqual(env.timestep, self.first + 1)

    def test_demonstration_contact_cannot_satisfy_execution_start(self):
        env = self.env
        start, _, _ = self.geometry()
        env.__dict__["_demonstration_flag"] = True
        env._drawing.begin("demonstration", int(env.elapsed_steps.item()))
        for _ in range(3):
            self.move(start)
        self.assertEqual(env.timestep, self.first)
        env.__dict__["_demonstration_flag"] = False
        env._drawing.begin("execution", int(env.elapsed_steps.item()))
        self.move(start)
        self.assertEqual(env.timestep, self.first + 1)

    def test_all_levels_have_a_start_before_the_configured_number_of_curves(self):
        for level in ("easy", "medium", "hard"):
            with self.subTest(level=level):
                env = RouteStick(seed=0, difficulty=level, obs_mode="state", sim_backend="cpu")
                try:
                    execution = [entry for entry in env.task_list if not entry["demonstration"]]
                    self.assertEqual(execution[0]["name"], "move to the starting target")
                    self.assertEqual(len(execution), env._imitation_config["segment_count"] + 1)
                    self.assertNotIn("expected_dir", execution[0])
                    self.assertEqual(len(env._route_curve_tasks["execution"]),
                                     env._imitation_config["segment_count"])
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
