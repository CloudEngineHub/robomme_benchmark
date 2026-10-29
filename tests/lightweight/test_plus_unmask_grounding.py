"""Unmask pickup labels keep the accepted container on the completion frame."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robomme_plus.evaluation.grounding import fill_grounding, subgoal_objects
from robomme_plus.scoring.subgoals import evaluate_tasks
from robomme_plus.tasks.permanence.utils import tasks


def actor(name, x):
    pose = SimpleNamespace(
        p=torch.tensor([[x, 0., .054]]), q=torch.tensor([[1., 0., 0., 0.]])
    )
    return SimpleNamespace(name=name, pose=pose, initial_pose=pose)


def make_task(task_name):
    config = tasks.task_config(task_name, "easy", 0)
    colors = ["red", "red", "blue", "green"]
    config["recipe"].update(
        total_targets=4, target_counts={"red": 2, "blue": 1, "green": 1},
        color_order=["red", "blue", "green"], pickup_sequence=colors,
        extra_colors=["red"],
    )
    covers = [actor(f"cover_{i}", .1 * i) for i in range(4)]
    tcp = actor("tcp", 0.)
    tcp.pose.p[0, 2] = .2
    env = SimpleNamespace(
        seed=0, cube_half_size=.02, elapsed_steps=torch.tensor([0]),
        _permanence_actors=covers, _permanence_colors=colors,
        _permanence_cubes=[actor(f"cube_{i}", .1 * i) for i in range(4)],
        _limit_config=config, _limit_diagnostics={},
        agent=SimpleNamespace(tcp=tcp, is_grasping=lambda obj: False),
    )
    if task_name == "ButtonUnmask":
        env.button = actor("button", -.2)
    elif task_name == "ButtonUnmaskSwap":
        env.button_left = actor("button_left", -.2)
        env.button_right = actor("button_right", -.2)
        env.button = env.button_left
    points = np.array([[.1 * i, 0.] for i in range(4)])
    quaternions = [cover.pose.q.numpy()[0] for cover in covers]
    # Keep the real task builder and pickup/scoring rules; bypass scene layout
    # and rendering so the completion-frame regression needs no simulator.
    with patch.object(tasks.base, "set_position"), \
         patch("robomme_plus.tasks.permanence.utils.view.initialize_view"), \
         patch("robomme_plus.tasks.permanence.utils.reachability.CoverReachability"), \
         patch.object(tasks, "static_layout", return_value=(points, {}, 1, quaternions)), \
         patch.object(tasks, "swap_routes", return_value=(
             SimpleNamespace(), np.arange(4), {})):
        tasks.setup(env, config)
    env._permanence_cover_spawned = {cover.name for cover in covers}
    env.timestep = next(i for i, entry in enumerate(env.task_list)
                        if entry["name"].startswith("pick up"))
    return env


def grounded_subgoal(env, segmentation):
    return fill_grounding(
        env.current_subgoal_segment, env.current_task_name,
        subgoal_objects(env.current_segment), segmentation,
        {i + 1: cover for i, cover in enumerate(env._permanence_actors)},
    )


class UnmaskGroundingTest(unittest.TestCase):
    def test_completed_pickups_keep_their_actor_and_visible_coordinates(self):
        for task_name in tasks.TASKS:
            for red_order in ((0, 1), (1, 0)):
                with self.subTest(task=task_name, red_order=red_order):
                    env = make_task(task_name)
                    covers = env._permanence_actors
                    # Before a pickup, the closest eligible container is suggested.
                    evaluate_tasks(env)
                    self.assertIs(env.current_segment(), covers[0])
                    segmentation = np.zeros((12, 30), dtype=int)
                    for i in range(4):
                        segmentation[2:6, 2 + 6*i:6 + 6*i] = i + 1
                    for index in (*red_order, 2):
                        cover = covers[index]
                        entry = env.task_list[env.timestep]
                        env.agent.tcp.pose.p[0, 0] = cover.pose.p[0, 0]
                        cover.pose.p[0, 2] = .18
                        result = evaluate_tasks(env)
                        self.assertFalse(result["fail"].item())
                        self.assertEqual(env._repeat_progress.picked[-1], cover.name)
                        self.assertEqual(env.current_task_name, entry["name"])
                        self.assertIs(env.current_segment(), cover)
                        self.assertEqual(
                            grounded_subgoal(env, segmentation),
                            entry["subgoal_segment"].replace("<>", f"<3, {3 + 6*index}>"),
                        )
                        # Releasing advances to the next pickup with a fresh target.
                        cover.pose.p[0, 2] = .054
                        result = evaluate_tasks(env)
                        self.assertFalse(result["fail"].item())
                        next_target = env.task_list[env.timestep]["segment"]()
                        self.assertNotIn(next_target.name, env._repeat_progress.picked)

    def test_hidden_picked_container_does_not_ground_a_visible_duplicate(self):
        for task_name in tasks.TASKS:
            with self.subTest(task=task_name):
                env = make_task(task_name)
                env._permanence_actors[0].pose.p[0, 2] = .18
                evaluate_tasks(env)
                segmentation = np.zeros((12, 12), dtype=int)
                segmentation[2:6, 2:6] = 2  # Only the unpicked red container is visible.
                self.assertEqual(grounded_subgoal(env, segmentation), env.current_task_name)


if __name__ == "__main__":
    unittest.main()
