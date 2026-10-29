"""Prevent the demonstrated-position shortcut in both VideoPlace tasks."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robomme_plus.tasks.reference.utils import place_corrections as corrections
from robomme_plus.tasks.reference.utils import place_scoring as scoring
from robomme_plus.tasks.reference.utils import place_swaps as swaps
from robomme_plus.tasks.reference.utils.place_projection import project_spec


class Quaternion:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[1.0, 0.0, 0.0, 0.0]])


def actor(name, point):
    return SimpleNamespace(name=name, p=np.asarray(point, dtype=float), pose=SimpleNamespace(q=Quaternion()))


class CorrectionPairsTest(unittest.TestCase):
    def test_one_cube_projection_keeps_the_selected_before_after_ring_roles(self):
        source = {'objects': {'demo_ids': [0, 1], 'answer_demo_index': 1,
                              'demo_before_target_ids': [6, 4],
                              'demo_after_target_ids': [5, 2]},
                  'actions': {'return_pose_by_object_id': {'answer': [0, 0, .02]}}}
        projected = project_spec(source, 'answer', [])
        self.assertEqual(projected['objects']['demo_ids'], [1])
        self.assertEqual(projected['objects']['demo_before_target_ids'], [4])
        self.assertEqual(projected['objects']['demo_after_target_ids'], [2])
        self.assertEqual(source['objects']['demo_before_target_ids'], [6, 4])

    def test_two_rounds_never_restore_reference_across_layouts_and_pins(self):
        optional = set()
        for count in (4, 6, 7, 8):
            for seed in range(60):
                rng = np.random.default_rng(seed)
                points = swaps.sample_random_layout(count, 0.04, rng)
                goal = seed % count
                # Test both an unchanged goal and a goal already in another slot.
                for reference in (points[goal].copy(), points[(goal + 1) % count].copy()):
                    current = points.copy()
                    pinned = [i for i in range(count) if i != goal][:seed % 2]
                    for _ in range(2):
                        unchanged = np.linalg.norm(current[goal] - reference) < 0.001
                        pair, details = corrections.plan_correction_pair(current, goal, reference, rng, pinned)
                        if unchanged:
                            self.assertIn(goal, pair)
                            self.assertTrue(details['goal_forced'])
                        else:
                            optional.add(goal in pair)
                        self.assertTrue(set(pair).isdisjoint(pinned))
                        current[list(pair)] = current[list(reversed(pair))]
                        self.assertGreater(np.linalg.norm(current[goal] - reference), 0.001)
        self.assertEqual(optional, {False, True})

    def test_impossible_correction_fails_instead_of_leaving_shortcut(self):
        points = np.array([[0, 0], [0, .2], [-.2, 0], [-.2, .2]])
        for pinned in ((0,), (1, 2, 3)):
            with self.assertRaises(RuntimeError):
                corrections.plan_correction_pair(points, 0, points[0], np.random.default_rng(1), pinned)


class CorrectionControllerTest(unittest.TestCase):
    def setUp(self):
        self.actors = [actor(f'target_{i}', [x, y, .005]) for i, (x, y) in enumerate(((-.2, -.2), (0, -.2), (-.2, .2), (0, .2)))]
        self.cube = actor('cube_red_0', [.1, 0, .02])
        self.env = SimpleNamespace(
            seed=42, all_cubes=[self.cube], demonstration_record_traj=True,
            agent=SimpleNamespace(tcp=actor('tcp', [-.1, .05, .15]), is_grasping=lambda cube: False),
            _video_place_layout=np.array([a.p[:2] for a in self.actors]),
            _limit_diagnostics={},
        )
        for item in (
            patch.object(swaps, 'position', lambda a, initial=False: a.p.copy()),
            patch.object(corrections, 'position', lambda a, initial=False: a.p.copy()),
            patch.object(swaps.base, 'set_position', lambda a, p, q=None: setattr(a, 'p', np.array(p))),
            patch.object(swaps.base, 'step_number', lambda env: 0),
            patch.object(swaps.SwapVisibility, 'measure', lambda self, actors: [
                dict(total_target_pixels=100, arm_occluded_pixels=0,
                     arm_occluded_fraction=0., allowed=True) for _ in actors
            ]),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.routes = swaps.ClockedSwaps(self.env, self.actors, .04, .004)
        self.routes.enable_final_corrections(0)

    def planner(self, routes):
        def move(pose):
            self.env.agent.tcp.p = np.asarray(pose.p, dtype=float)
            self.assertFalse(routes.correction_active)
            return 0
        return SimpleNamespace(
            move_to_pose_with_screw=move,
            open_gripper=lambda t: [routes.update() for _ in range(t)],
        )

    def test_corrections_finish_before_home_and_execution_stays_fixed(self):
        routes = self.routes
        routes.record_reference(self.actors[0].p[:2], 12)
        routes.finish_demonstration()
        planner = self.planner(routes)
        self.env._limit_routes = routes
        with patch.object(routes.visibility, 'measure', side_effect=AssertionError('final correction visibility gate')):
            corrections.solve_final_corrections(SimpleNamespace(unwrapped=self.env), planner)
        rounds = routes.metadata['rounds']
        self.assertEqual(len(rounds), 2)
        self.assertTrue(all(r['kind'] == 'correction' and len(r['pairs']) == 1 for r in rounds))
        self.assertIn('target_0', rounds[0]['pairs'][0])
        self.assertEqual(routes.metadata['final_corrections']['hold_end_clock'] - routes.metadata['final_corrections']['hold_start_clock'], 80)
        self.assertTrue(routes.corrections_complete)
        np.testing.assert_allclose(routes.metadata['final_corrections']['hold_tcp_position'], [0, 0, .20])
        self.assertTrue(all(np.allclose(r['start_tcp_position_m'], [0, 0, .20]) for r in rounds))
        self.assertEqual(self.env._limit_diagnostics['demonstration_swap_wait_steps'], 80)
        before = np.array([a.p.copy() for a in self.actors])
        self.env.demonstration_record_traj = False
        for _ in range(100):
            routes.update()
        np.testing.assert_array_equal([a.p for a in self.actors], before)
        self.assertEqual(len(routes.metadata['rounds']), 2)

    def test_existing_round_finishes_before_two_new_one_pair_rounds(self):
        routes = self.routes
        routes.record_reference(self.actors[0].p[:2], 0)
        for _ in range(11):
            routes.update()
        routes.finish_demonstration()
        self.env._limit_routes = routes
        planner = self.planner(routes)
        corrections.solve_final_corrections(SimpleNamespace(unwrapped=self.env), planner)
        self.assertEqual([r['kind'] for r in routes.metadata['rounds']], ['ordinary', 'correction', 'correction'])
        self.assertTrue(routes.corrections_complete)

    def test_eighth_ring_can_be_the_corrected_goal_and_stays_fixed_in_execution(self):
        points = swaps.sample_random_layout(8, .04, np.random.default_rng(17))
        actors = [actor(f'target_{i}', [*point, .005]) for i, point in enumerate(points)]
        self.env._video_place_layout = points.copy()
        routes = swaps.ClockedSwaps(self.env, actors, .04, .004)
        routes.enable_final_corrections(7)
        routes.record_reference(points[7], 0)
        routes.finish_demonstration()
        self.env._limit_routes = routes
        planner = self.planner(routes)
        corrections.solve_final_corrections(SimpleNamespace(unwrapped=self.env), planner)
        self.assertIn('target_7', routes.metadata['rounds'][0]['pairs'][0])
        self.assertTrue(routes.corrections_complete)
        self.assertGreater(np.linalg.norm(actors[7].p[:2] - points[7]), .001)
        final_positions = np.array([a.p.copy() for a in actors])
        self.env.demonstration_record_traj = False
        for _ in range(100):
            routes.update()
        np.testing.assert_array_equal([a.p for a in actors], final_positions)

    def test_handoff_requires_completed_corrections_and_different_endpoint(self):
        with self.assertRaisesRegex(RuntimeError, 'before both correction'):
            self.routes.begin_execution()
        self.routes.record_reference(self.actors[0].p[:2], 0)
        self.routes.correction_rounds_completed = 2
        with self.assertRaisesRegex(RuntimeError, 'demonstrated position'):
            self.routes.begin_execution()

    def test_final_pause_requires_releasing_every_cube(self):
        self.env._limit_routes = self.routes
        self.env.agent.is_grasping = lambda cube: True
        with self.assertRaisesRegex(RuntimeError, 'Release all cubes'):
            corrections.solve_final_corrections(SimpleNamespace(unwrapped=self.env), None)

    def test_failed_central_move_does_not_start_corrections(self):
        self.env._limit_routes = self.routes
        self.routes.record_reference(self.actors[0].p[:2], 0)
        self.routes.finish_demonstration()
        planner = SimpleNamespace(move_to_pose_with_screw=lambda pose: -1)
        self.assertEqual(corrections.solve_final_corrections(SimpleNamespace(unwrapped=self.env), planner), -1)
        self.assertFalse(self.routes.correction_active)
        self.assertEqual(self.routes.metadata['rounds'], [])

    def test_ordinary_swap_requires_strict_height_and_robot_side(self):
        for point in ((0, 0, .15), (.01, 0, .15), (-.1, 0, .10), (-.1, 0, .09)):
            with self.subTest(point=point):
                self.env.agent.tcp.p = np.array(point)
                for _ in range(45):
                    self.routes.update()
                self.assertEqual(self.routes.metadata['rounds'], [])
        self.env.agent.tcp.p = np.array([-.1, 0, .11])
        self.routes.update()
        self.assertEqual(len(self.routes.metadata['rounds']), 1)
        self.assertEqual(self.routes.metadata['rounds'][0]['kind'], 'ordinary')

    def test_active_ordinary_swap_finishes_after_leaving_start_region(self):
        self.routes.update()
        self.assertTrue(self.routes.pairs)
        self.env.agent.tcp.p = np.array([.01, 0, .05])
        for _ in range(80):
            self.routes.update()
        self.assertEqual(len(self.routes.metadata['rounds']), 1)
        self.assertEqual(self.routes.metadata['completed_rounds'], 1)
        np.testing.assert_allclose(
            [a.p[:2] for a in self.actors], self.routes.metadata['rounds'][0]['end_xy']
        )

    def test_occluded_pair_waits_until_arm_clears(self):
        blocked = [dict(total_target_pixels=100, arm_occluded_pixels=51,
                        arm_occluded_fraction=.51, allowed=False) for _ in self.actors]
        with patch.object(self.routes.visibility, 'measure', return_value=blocked):
            self.routes.update()
        self.assertFalse(self.routes.pairs)
        self.assertEqual(self.routes.metadata['rounds'], [])
        self.assertTrue(self.routes.metadata['visibility_rejections'])
        self.assertEqual(self.routes.metadata['ticks'][-1]['reason'], 'arm_occluded')
        self.routes.update()
        self.assertTrue(self.routes.pairs)

    def test_only_occluded_pair_is_removed_from_two_pair_round(self):
        visible = dict(total_target_pixels=100, arm_occluded_pixels=50,
                       arm_occluded_fraction=.5, allowed=True)
        blocked = dict(total_target_pixels=100, arm_occluded_pixels=51,
                       arm_occluded_fraction=.51, allowed=False)
        pairs = [(0, 1), (2, 3)]
        paths = swaps.pair_paths(self.env._video_place_layout, pairs, .04)
        with patch.object(self.routes.visibility, 'measure', return_value=[visible, blocked, visible, visible]), \
                patch.object(swaps, 'plan_neighbor_pairs', return_value=(paths, pairs, [{}, {}])):
            self.routes.update()
        self.assertEqual(self.routes.pairs, [(2, 3)])
        np.testing.assert_allclose(self.actors[0].p[:2], self.env._video_place_layout[0])
        np.testing.assert_allclose(self.actors[1].p[:2], self.env._video_place_layout[1])
        self.assertEqual(self.routes.metadata['rounds'][0]['pairs'], [['target_2', 'target_3']])

    def test_visibility_is_only_checked_at_round_start(self):
        self.routes.update()
        with patch.object(self.routes.visibility, 'measure', side_effect=AssertionError('mid-swap check')):
            for _ in range(swaps.SWAP_STEPS - 1):
                self.routes.update()
        self.assertEqual(self.routes.metadata['completed_rounds'], 1)

    def test_only_proposed_targets_are_measured(self):
        pairs = [(0, 1)]
        paths = swaps.pair_paths(self.env._video_place_layout, pairs, .04)

        def measure(actors):
            self.assertEqual([a.name for a in actors], ['target_0', 'target_1'])
            return [dict(total_target_pixels=100, arm_occluded_pixels=0,
                         arm_occluded_fraction=0., allowed=True) for _ in actors]

        with patch.object(self.routes.visibility, 'measure', side_effect=measure), \
                patch.object(swaps, 'plan_neighbor_pairs', return_value=(paths, pairs, [{}])):
            self.routes.update()
        self.assertEqual(self.routes.pairs, pairs)

    def test_reference_is_captured_at_landing_not_later_lift(self):
        env = self.env
        env.targets = self.actors
        env.target_cube, env.target_target = self.cube, self.actors[0]
        env.timestep, env._limit_reference_task_index = 3, 3
        env._limit_routes = self.routes
        env._limit_grasped = {self.cube.name: False}
        env._limit_pending_release = set()
        env._limit_landing_since = {}
        env._limit_placement_events = []
        self.cube.linear_velocity = torch.zeros((1, 3))
        env.agent.is_grasping = lambda cube: True
        with patch.object(scoring, 'position', lambda a, initial=False: a.p.copy()), \
                patch.object(scoring, 'step_number', side_effect=[36, 37, 38]), \
                patch.object(scoring, 'gripper_released', return_value=True):
            scoring.placement_events(env)
            self.cube.p = np.array([*self.actors[0].p[:2], .02])
            env.agent.is_grasping = lambda cube: False
            scoring.placement_events(env)
            scoring.placement_events(env)
        reference = self.actors[0].p[:2].copy()
        self.actors[0].p[:2] += .1
        np.testing.assert_array_equal(self.routes.reference_xy, reference)
        self.assertEqual(self.routes.metadata['final_corrections']['reference_step'], 38)


class RequestedVisitTest(unittest.TestCase):
    def setUp(self):
        self.cube, self.other = object(), object()
        self.a, self.b = object(), object()

    def pair(self, cube, target):
        return [
            {'name': 'pick up the cube', 'segment': cube, 'demonstration': True},
            {'name': 'drop the cube onto target', 'segment': target, 'demonstration': True},
        ]

    def test_order_uses_requested_visit_even_if_ring_was_visited_twice(self):
        env = SimpleNamespace(task_name='VideoPlaceOrder', target_cube=self.cube,
                              target_target=self.a, which_in_subset=3)
        env.task_list = self.pair(self.cube, self.a) + self.pair(self.other, self.a)
        env.task_list += self.pair(self.cube, self.b) + self.pair(self.cube, self.a)
        self.assertEqual(corrections.reference_placement(env), 7)

    def test_button_uses_last_before_or_first_after_for_requested_cube(self):
        env = SimpleNamespace(task_name='VideoPlaceButton', target_cube=self.cube)
        env.task_list = self.pair(self.cube, self.a) + self.pair(self.cube, self.b)
        env.task_list += [{'name': 'press the button', 'demonstration': True}]
        env.task_list += self.pair(self.other, self.a) + self.pair(self.cube, self.a) + self.pair(self.cube, self.b)
        env.task_flag, env.target_target = True, self.b
        self.assertEqual(corrections.reference_placement(env), 3)
        env.task_flag, env.target_target = False, self.a
        self.assertEqual(corrections.reference_placement(env), 8)

    def test_fifth_visit_ignores_other_cube_and_home_return(self):
        targets = [object() for _ in range(5)]
        env = SimpleNamespace(task_name='VideoPlaceOrder', target_cube=self.cube,
                              target_target=targets[4], which_in_subset=5)
        env.task_list = []
        for target in targets:
            env.task_list += self.pair(self.other, target)
        env.task_list += [
            {'name': 'pick up the cube', 'segment': self.other, 'demonstration': True},
            {'name': 'put the cube back to its original position', 'segment': object(), 'demonstration': True},
        ]
        for target in targets:
            env.task_list += self.pair(self.cube, target)
        expected = len(env.task_list) - 1
        env.task_list += [
            {'name': 'pick up the cube', 'segment': self.cube, 'demonstration': True},
            {'name': 'put the cube back to its original position', 'segment': object(), 'demonstration': True},
        ]
        self.assertEqual(corrections.reference_placement(env), expected)


if __name__ == '__main__':
    unittest.main()
