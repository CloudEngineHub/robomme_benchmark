"""PickHighlight accepts any target order without skipping release or finish."""

import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from robomme_plus.evaluation.grounding import subgoal_objects
from robomme_plus.scoring.failures import failure_message
from robomme_plus.tasks.reference.pick_highlight import _TaskScene


class Cube:
    def __init__(self, name):
        self.name = name
        self.pose = SimpleNamespace(p=torch.tensor([[0., 0., .02]]))


def make_task(count=3):
    env = object.__new__(_TaskScene)
    env.target_cubes = [Cube(f'target_{i}') for i in range(count)]
    env.target_cube_names = [cube.name for cube in env.target_cubes]
    env.all_cubes = env.target_cubes + [Cube('distractor')]
    env.holding = set()
    env.button_pressed = False
    env.button = SimpleNamespace(
        get_qpos=lambda: torch.tensor([[-.01 if env.button_pressed else 0.]]))
    env.cap_link = object()
    env.agent = SimpleNamespace(
        tcp=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0., 0., .1]]))),
        is_grasping=lambda cube: torch.tensor([cube in env.holding]),
    )
    env.use_demonstrationwrapper = True
    env.demonstration_record_traj = False
    env.difficulty = 'xhard2'
    env._build_task_list()
    return env


def start(env):
    env.button_pressed = True
    env.evaluate()
    env.button_pressed = False


def pick(env, cube):
    cube.pose.p[0, 2] = .1
    env.holding.add(cube)
    return env.evaluate()


def release(env, cube):
    cube.pose.p[0, 2] = .02
    env.holding.discard(cube)
    return env.evaluate()


class HighlightOrderTest(unittest.TestCase):
    def test_target_selection_gives_all_cubes_and_pairs_equal_chances(self):
        import numpy as np
        from robomme_plus.tasks.reference.utils.highlight_levels import sample_highlight_ids

        for total, count in ((9, 3), (10, 5), (11, 4), (13, 6), (14, 5), (17, 8)):
            hits = np.zeros(total)
            pair_hits = np.zeros((total, total))
            for seed in range(4096):
                selected = sample_highlight_ids(seed, total, count)
                self.assertEqual(len(set(selected)), count)
                self.assertTrue(all(0 <= index < total for index in selected))
                hits[selected] += 1
                pair_hits[np.ix_(selected, selected)] += 1
            # Check both marginals and pairs: a random contiguous slice would
            # pass marginals while incorrectly preferring neighboring IDs.
            self.assertLess(np.max(np.abs(hits / 4096 - count / total)), .04)
            expected_pair = count * (count - 1) / (total * (total - 1))
            self.assertLess(np.max(np.abs(
                pair_hits[~np.eye(total, dtype=bool)] / 4096 - expected_pair
            )), .04)

    def test_target_selection_is_reproducible_and_ignores_other_random_draws(self):
        from robomme_plus.tasks.reference.utils.highlight_levels import sample_highlight_ids

        expected = sample_highlight_ids(42, 17, 9)
        torch.rand(1000, generator=torch.Generator().manual_seed(42))
        self.assertEqual(sample_highlight_ids(42, 17, 9), expected)
        self.assertNotEqual(sample_highlight_ids(43, 17, 9), expected)
        for count in (0, 18):
            with self.assertRaises(ValueError):
                sample_highlight_ids(42, 17, count)

    def test_every_permutation_completes_after_release_and_final_button(self):
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                env = make_task()
                start(env)
                for stage, index in enumerate(order):
                    cube = env.target_cubes[index]
                    info = pick(env, cube)
                    self.assertFalse(info['fail'].item())
                    self.assertFalse(info['success'].item())
                    self.assertEqual(env.timestep, 2 + 2 * stage)
                    # Holding the cube cannot skip its put-down stage.
                    env.evaluate()
                    self.assertEqual(env.timestep, 2 + 2 * stage)
                    info = release(env, cube)
                    self.assertFalse(info['success'].item())
                    self.assertEqual(env.timestep, 3 + 2 * stage)
                self.assertEqual(list(env.target_cube_pickup_counts.values()), [1, 1, 1])
                self.assertFalse(env.evaluate()['success'].item())
                env.button_pressed = True
                self.assertTrue(env.evaluate()['success'].item())

    def test_repeated_target_cannot_replace_a_missing_target(self):
        env = make_task(2)
        start(env)
        first, second = env.target_cubes
        pick(env, first)
        release(env, first)
        pick(env, first)
        self.assertEqual(env.timestep, 3)
        self.assertEqual(env._highlight_picks, [first])
        self.assertFalse(env.evaluate()['success'].item())
        release(env, first)
        pick(env, second)
        release(env, second)
        env.button_pressed = True
        self.assertTrue(env.evaluate()['success'].item())

    def test_grasp_and_lift_are_still_required(self):
        env = make_task()
        start(env)
        cube = env.target_cubes[-1]
        cube.pose.p[0, 2] = .1
        env.evaluate()
        self.assertFalse(env._highlight_picks)
        cube.pose.p[0, 2] = .02
        env.holding.add(cube)
        env.evaluate()
        self.assertFalse(env._highlight_picks)
        pick(env, cube)
        self.assertEqual(env._highlight_picks, [cube])

    def test_grounding_and_oracle_follow_remaining_or_accepted_target(self):
        env = make_task()
        start(env)
        first, _, chosen = env.target_cubes
        pick(env, chosen)
        self.assertEqual(subgoal_objects(env.task_list[1]['segment']), [chosen])
        release(env, chosen)
        next_entry = env.task_list[env.timestep]
        self.assertEqual(subgoal_objects(next_entry['segment']), [first])
        with patch('robomme_plus.tasks.reference.pick_highlight.solve_pickup') as solve:
            planner = object()
            next_entry['solve'](env, planner)
            solve.assert_called_once_with(env, planner, obj=first)

    def test_wrong_cube_and_pickup_before_button_still_fail(self):
        for phase in ('before_button', 'pickup', 'putdown'):
            with self.subTest(phase=phase):
                env = make_task()
                if phase != 'before_button':
                    start(env)
                if phase == 'putdown':
                    pick(env, env.target_cubes[-1])
                info = pick(env, env.all_cubes[-1])
                self.assertTrue(info['fail'].item())
                expected = ('Picked up a cube before pressing the start button.'
                            if phase == 'before_button' else 'Picked up the wrong cube.')
                self.assertEqual(failure_message(env), expected)

    def test_episode_initialization_clears_pickup_progress(self):
        env = make_task()
        start(env)
        pick(env, env.target_cubes[-1])
        # _initialize_episode rebuilds these task entries on every reset.
        env._build_task_list()
        self.assertEqual(env.timestep, 0)
        self.assertEqual(env._highlight_picks, [])
        self.assertIsNone(env.currentpickup)
        self.assertEqual(list(env.target_cube_pickup_counts.values()), [0, 0, 0])
        self.assertFalse(any(env.target_cube_pickup_active.values()))

    def test_wrong_cube_at_finish_fails_even_with_simultaneous_button_press(self):
        for button_pressed in (False, True):
            with self.subTest(button_pressed=button_pressed):
                env = make_task()
                start(env)
                for cube in env.target_cubes:
                    pick(env, cube)
                    release(env, cube)
                finish_stage = env.timestep
                env.button_pressed = button_pressed
                info = pick(env, env.all_cubes[-1])
                self.assertTrue(info['fail'].item())
                self.assertFalse(info['success'].item())
                self.assertEqual(env.timestep, finish_stage)
                self.assertEqual(failure_message(env), 'Picked up the wrong cube.')
                # Releasing the distractor cannot undo a detected failure.
                release(env, env.all_cubes[-1])
                env.button_pressed = True
                info = env.evaluate()
                self.assertTrue(info['fail'].item())
                self.assertFalse(info['success'].item())

    def test_targets_placed_aside_keep_their_identity(self):
        for pick_wrong in (False, True):
            with self.subTest(pick_wrong=pick_wrong):
                env = make_task()
                start(env)
                for index, cube in enumerate(env.target_cubes):
                    pick(env, cube)
                    cube.pose.p[0, :2] = torch.tensor([.3, -.3 + .05 * index])
                    info = release(env, cube)
                    self.assertFalse(info['fail'].item())
                env.button_pressed = True
                # The distractor is at the targets' old position; its identity,
                # rather than its position, must determine whether it is wrong.
                info = pick(env, env.all_cubes[-1]) if pick_wrong else env.evaluate()
                self.assertEqual(info['fail'].item(), pick_wrong)
                self.assertEqual(info['success'].item(), not pick_wrong)


if __name__ == '__main__':
    unittest.main()
