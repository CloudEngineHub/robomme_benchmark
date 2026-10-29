"""ButtonUnmaskSwap timing, actual rearrangement, passages, and traffic safety."""
import unittest
from types import SimpleNamespace
import numpy as np
from transforms3d.euler import euler2quat

from robomme_plus.tasks.permanence.utils.routes import (
    BOUNDS, CENTRAL_BOUNDS, MARGIN_M, plan_routes, segment_distance,
    rearrangement_audit, completed_passages,
    receiving_side_velocities, counterflow_audit,
    passage_forward, passage_intersects, initial_bounds, packed_initial,
)
from robomme_plus.tasks.permanence.utils.tasks import task_config
from robomme_plus.tasks.permanence.utils.tasks import (
    check_early_opening, button_subgoals, language_goal,
)
from robomme_plus.tasks.permanence.utils.progress import cover_exposure
from robomme_plus.tasks.permanence.utils.board import ButtonDeadline
from robomme_plus.common.motion import swept_gap
from robomme_plus.evaluation.grounding import fill_grounding, subgoal_objects


class RectangularReachability:
    def contains(self, points):
        points = np.asarray(points)
        return ((points[..., 0] >= -.49) & (points[..., 0] <= .20)
                & (np.abs(points[..., 1]) <= .34))

    def segment(self, a, b):
        return bool(self.contains(np.array([a, b])).all())


class ButtonPressRulesTest(unittest.TestCase):
    def test_either_order_and_simultaneous_distinct_presses(self):
        for order in (("left", "right"), ("right", "left")):
            deadline = ButtonDeadline(("left", "right"), 10, 480)
            self.assertFalse(deadline.observe(20, {order[0]: .01}, True, tcp_position=[0, 0, .03]))
            self.assertFalse(deadline.observe(40, {order[1]: .01}, True, tcp_position=[0, 0, .03]))
            self.assertEqual(list(deadline.presses), list(order))
            self.assertTrue(deadline.snapshot()["both_pressed_in_time"])
            self.assertEqual(deadline.motion_stop, 136)
        deadline = ButtonDeadline(("left", "right"), 10, 480)
        self.assertFalse(deadline.observe(20, {"left": .01, "right": .01}, True, tcp_position=[0, 0, .03]))
        self.assertEqual(len(deadline.presses), 2)

    def test_holding_and_threshold_jitter_are_one_press(self):
        deadline = ButtonDeadline(("left", "right"), 10, 480)
        for step, depth in enumerate((.01, .006, .004, .0051, .003, .01), 20):
            self.assertFalse(deadline.observe(step, {"left": depth}, True, tcp_position=[0, 0, .03]))
            # Multiple physics/evaluation observations of one sample are harmless.
            self.assertFalse(deadline.observe(step, {"left": depth}, True, tcp_position=[0, 0, .03]))
        self.assertEqual(deadline.presses, {"left": 20})
        self.assertEqual(len(deadline.press_events), 1)

    def test_release_and_repress_fail_for_either_button(self):
        for name in ("left", "right"):
            deadline = ButtonDeadline(("left", "right"), 10, 480)
            deadline.observe(20, {name: .01}, True, tcp_position=[0, 0, .03])
            for step in (21, 22, 23):
                deadline.observe(step, {name: .002}, True, tcp_position=[0, 0, .08])
            self.assertTrue(deadline.observe(24, {name: .01}, True, tcp_position=[0, 0, .03]))
            self.assertEqual(deadline.failure_reason, "repeated_button_press")
            self.assertEqual(deadline.repeated_button, name)
            self.assertEqual(deadline.failed_at, 24)
            self.assertTrue(deadline.observe(40, {"left": .01, "right": .01}, True, tcp_position=[0, 0, .03]))
            self.assertEqual(len(deadline.presses), 1)
            self.assertEqual(deadline.failed_at, 24)

    def test_full_depth_bounces_without_withdrawal_are_one_press(self):
        deadline = ButtonDeadline(("left", "right"), 10, 480)
        for step, depth in enumerate((.01, 0, .01, 0, 0, 0, .01), 20):
            self.assertFalse(deadline.observe(step, {"left": depth}, True,
                                              tcp_position=[.008, 0, .03]))
        self.assertEqual(deadline.presses, {"left": 20})
        self.assertEqual(len(deadline.press_events), 1)

    def test_brief_withdrawal_does_not_end_a_press(self):
        deadline = ButtonDeadline(("left", "right"), 10, 480)
        deadline.observe(20, {"left": .01}, True, tcp_position=[0, 0, .03])
        for _ in range(30):
            # Physics callbacks within one control step cannot advance the timer.
            self.assertFalse(deadline.observe(21, {}, True, tcp_position=[0, 0, .08]))
        self.assertFalse(deadline.observe(22, {"left": .01}, True, tcp_position=[0, 0, .03]))
        self.assertEqual(len(deadline.press_events), 1)

    def test_release_must_be_continuous(self):
        deadline = ButtonDeadline(("left", "right"), 10, 480)
        deadline.observe(20, {"left": .01}, True, tcp_position=[0, 0, .03])
        deadline.observe(21, {}, True, tcp_position=[0, 0, .08])
        deadline.observe(22, {"left": .003}, True, tcp_position=[0, 0, .08])
        deadline.observe(23, {}, True, tcp_position=[0, 0, .08])
        self.assertFalse(deadline.observe(24, {"left": .01}, True, tcp_position=[0, 0, .03]))
        self.assertEqual(len(deadline.press_events), 1)

    def test_lateral_withdrawal_and_return_also_counts_as_repeat(self):
        deadline = ButtonDeadline(("left", "right"), 10, 480)
        deadline.observe(20, {"left": .01}, True, tcp_position=[0, 0, .03])
        for step in (21, 22, 23):
            deadline.observe(step, {}, True, tcp_position=[0, .10, .03])
        self.assertTrue(deadline.observe(24, {"left": .01}, True, tcp_position=[0, 0, .03]))

    def test_repeat_still_fails_after_motion_and_after_deadline(self):
        for repeat_step in (50, 200, 500):
            deadline = ButtonDeadline(("left", "right"), 10, 480)
            deadline.observe(20, {"left": .01}, True, tcp_position=[0, 0, .03])
            for step in (21, 22, 23):
                deadline.observe(step, {}, True, tcp_position=[0, 0, .08])
            deadline.observe(40, {"right": .01}, True, tcp_position=[0, .15, .03])
            self.assertTrue(deadline.observe(repeat_step, {"left": .01}, repeat_step < 136, tcp_position=[0, 0, .03]))
            self.assertEqual(deadline.failure_reason, "repeated_button_press")

    def test_missing_button_deadline_is_still_a_failure(self):
        deadline = ButtonDeadline(("left", "right"), 10, 480)
        deadline.observe(20, {"right": .01}, True, tcp_position=[0, 0, .03])
        self.assertTrue(deadline.observe(480, {"left": .01}, False, tcp_position=[0, 0, .03]))
        self.assertEqual(deadline.failure_reason, "button_deadline_missed")

    def test_subgoals_follow_the_actual_order_and_ground_remaining_button(self):
        left_cap, right_cap = object(), object()
        left = SimpleNamespace(name="button_left", links=[left_cap])
        right = SimpleNamespace(name="button_right", links=[right_cap])
        for first, second in ((left, right), (right, left)):
            env = SimpleNamespace(button=left, button_left=left, button_right=right,
                                  cap_links={left.name: [left_cap], right.name: [right_cap]},
                                  _permanence_presses={})
            tasks = button_subgoals(env)
            self.assertEqual([task["name"] for task in tasks],
                             ["press the first button", "press the second button"])
            self.assertEqual([task["subgoal_segment"] for task in tasks],
                             ["press the first button at <>", "press the second button at <>"])
            self.assertEqual(subgoal_objects(tasks[0]["segment"]), [right_cap])
            self.assertFalse(tasks[0]["func"]())
            env._permanence_presses[first.name] = 20
            self.assertTrue(tasks[0]["func"]())
            self.assertEqual(subgoal_objects(tasks[0]["segment"]), first.links)
            self.assertFalse(tasks[1]["func"]())
            self.assertEqual(tasks[1]["segment"](), second.links)
            # One grounded point follows the remaining button, even when the
            # robot presses the other button before the recommended first one.
            remaining = second.links[0]
            segmentation = np.zeros((16, 16), dtype=int)
            segmentation[2:6, 4:8] = 7
            for region, expected in (((2, 6, 4, 8), '<3, 5>'),
                                     ((8, 12, 10, 14), '<9, 11>')):
                segmentation.fill(0)
                y0, y1, x0, x1 = region
                segmentation[y0:y1, x0:x1] = 7
                self.assertEqual(fill_grounding(
                    tasks[1]['subgoal_segment'], tasks[1]['name'],
                    subgoal_objects(tasks[1]['segment']), segmentation, {7: remaining}),
                    f'press the second button at {expected}')
            env._permanence_presses[second.name] = 40
            self.assertTrue(tasks[1]["func"]())

    def test_goal_uses_requested_once_per_button_wording(self):
        recipe = task_config("ButtonUnmaskSwap", "hard", 0)["recipe"]
        self.assertTrue(language_goal(recipe).startswith("First press both buttons once, then "))


class EarlyOpeningTest(unittest.TestCase):
    def test_closed_covers_allow_arbitrary_yaw_and_settling(self):
        for cube_yaw in np.linspace(-np.pi, np.pi, 17):
            for cover_yaw in np.linspace(-np.pi, np.pi, 13):
                for z in (.052, .054):
                    self.assertIsNone(cover_exposure(
                        [0, 0, z], euler2quat(np.pi, 0, cover_yaw),
                        [0, 0, .02/1.2], euler2quat(0, 0, cube_yaw), .02))

    def test_lift_tilt_sideways_knock_and_escaped_cube(self):
        closed = np.array([0, 0, .054]); cube = np.array([0, 0, .02/1.2])
        q = euler2quat(np.pi, 0, 0); cq = [1, 0, 0, 0]
        for xyz, rotation, point in (
                (closed+[0, 0, .01], q, cube),
                (closed+[.02, 0, 0], q, cube),
                (closed, euler2quat(np.pi+.6, 0, 0), cube),
                (closed, q, cube+[.04, 0, 0])):
            self.assertIsNotNone(cover_exposure(xyz, rotation, point, cq, .02))

    @staticmethod
    def scene(presses=(), spawned=True, parked=True):
        import torch
        def actor(name, p, q):
            return SimpleNamespace(name=name, pose=SimpleNamespace(
                p=np.array(p), q=torch.tensor(q, dtype=torch.float64)))
        cover = actor('cover', [.02, 0, .054], [0, 1, 0, 0])
        cube = actor('cube', [40, 40, 2] if parked else [.05, 0, .02/1.2], [1, 0, 0, 0])
        return SimpleNamespace(_limit_config={'task': 'ButtonUnmaskSwap'},
            _permanence_cover_spawned={'cover'} if spawned else set(),
            _limit_failed=False, _permanence_presses=dict.fromkeys(presses, 12),
            _permanence_early_opening=None,
            _limit_routes=SimpleNamespace(actors=[cover], location=lambda i: np.zeros(2)),
            _limit_hidden=[(cube, 0, .02/1.2)], cube_half_size=.02,
            elapsed_steps=np.array(15), _limit_diagnostics={})

    def test_zero_or_one_press_fails_and_stays_failed(self):
        for presses in ((), ('left',)):
            for parked in (True, False):
                env = self.scene(presses=presses, parked=parked)
                check_early_opening(env)
                self.assertTrue(env._limit_failed)
                self.assertEqual(env._failure_code, "early_container_opening")
                event = env._permanence_early_opening
                env._permanence_presses = {'left': 12, 'right': 16}
                env._limit_routes.actors[0].pose.p = np.array([0, 0, .054])
                check_early_opening(env)
                self.assertTrue(env._limit_failed)
                self.assertIs(env._permanence_early_opening, event)

    def test_reveal_and_two_distinct_accepted_presses_allow_exposure(self):
        for env in (self.scene(spawned=False), self.scene(presses=('left', 'right'))):
            check_early_opening(env)
            self.assertFalse(env._limit_failed)
            self.assertIsNone(getattr(env, "_failure_code", None))
            self.assertIsNone(env._permanence_early_opening)


class OneWayPassageTest(unittest.TestCase):
    def test_reverse_motion_is_blocked_in_both_halves_and_mouths(self):
        for direction in (-1, 1):
            for y in (-.24, -.14, .14, .24):
                a = np.array([-.33, y]); b = a + [0, -.005*direction]
                self.assertFalse(passage_forward(a, b, direction, -.30, [-.26, .26]))
                self.assertTrue(passage_forward(b, a, direction, -.30, [-.26, .26]))
            self.assertTrue(passage_forward([-.33, .1], [-.33, .1], direction, -.30, [-.26, .26]))
            self.assertTrue(passage_forward([-.33, .1], [-.32, .1], direction, -.30, [-.26, .26]))

    def test_complete_segments_and_central_return_flow(self):
        # Endpoints outside the corridor do not permit cutting through it.
        self.assertTrue(passage_intersects([-.4, .4], [-.4, -.4], -.30, [-.26, .26]))
        self.assertFalse(passage_forward([-.4, .4], [-.4, -.4], 1, -.30, [-.26, .26]))
        # A bounding box alone would incorrectly block this diagonal.
        self.assertFalse(passage_intersects([-.4, .4], [-.1, .2], -.30, [-.26, .26]))
        for direction in (-1, 1):
            self.assertTrue(passage_forward([0, .2], [0, -.2], direction, -.30, [-.26, .26]))
            self.assertTrue(passage_forward([0, -.2], [0, .2], direction, -.30, [-.26, .26]))

    def test_passage_completion_uses_both_end_gates(self):
        lane = dict(x_max_m=-.30, y_range_m=[-.26, .26], direction=1)
        meta = dict(robot_side_crossing_xy_m=[[-.33, -.22], [-.33, .22]],
                    one_way_passage=lane,
                    passage_routes=[dict(actor_index=0, direction='right-to-left')])
        path = np.zeros((100, 1, 2)); path[:, 0, 0] = -.40
        path[:, 0, 1] = np.linspace(-.24, .24, 100)
        self.assertEqual(len(completed_passages(path, meta)), 1)
        self.assertEqual(completed_passages(path[:50], meta), [])
        self.assertEqual(completed_passages(path[::-1], meta), [])
        path[:, 0, 0] = -.10
        self.assertEqual(completed_passages(path, meta), [])


class InitialAreaTest(unittest.TestCase):
    def test_all_footprints_start_between_buttons_and_board(self):
        radius = np.sqrt(2) * .0275
        # A skewed pair must use the board-facing edge of both buttons.
        obstacles = [([-.24, -.10], .055), ([-.22, .10], .055)]
        bounds = initial_bounds(obstacles)
        np.testing.assert_allclose(bounds, [[-.153, .173], [-.30, .30]])
        for count in (6, 8, 10, 11, 13):
            p, _ = packed_initial(count, radius, count, .00675, obstacles,
                                  reachable_area=RectangularReachability())
            self.assertTrue(np.all(p - radius >= bounds[:, 0]))
            self.assertTrue(np.all(p + radius <= bounds[:, 1]))
            self.assertGreaterEqual(swept_gap(p, p, np.full(count, radius)), MARGIN_M)

    def test_too_small_initial_area_fails_instead_of_expanding(self):
        with self.assertRaisesRegex(ValueError, 'No initial area'):
            packed_initial(13, np.sqrt(2)*.0275, 0, .00675,
                           [([.05, -.1], .055), ([.05, .1], .055)])


class SharedVelocityTest(unittest.TestCase):
    def test_shared_drift_keeps_local_reversals_and_respects_speed(self):
        for direction in (-1, 1):
            local = np.array([[.002, .003], [0, -.004], [-.003, 0], [.001, .002]])
            local[:, 1] *= direction
            saved = local.copy()
            receiving = np.array([True, True, True, False])
            limits = np.full(4, .005)
            result = receiving_side_velocities(local, receiving, direction, limits)
            np.testing.assert_array_equal(local, saved)
            np.testing.assert_array_equal(result[~receiving], local[~receiving])
            self.assertLessEqual(np.mean(result[receiving, 1]*direction), -.25*.005+1e-12)
            self.assertTrue(np.any(result[receiving, 1]*direction > 0))
            self.assertTrue(np.all(np.linalg.norm(result, axis=1) <= limits+1e-12))

    def test_drift_overcomes_opposing_local_steering_without_over_speed(self):
        rng = np.random.default_rng(19)
        for count in (1, 3, 13):
            for direction in (-1, 1):
                caps = rng.uniform(.002, .00675, count)
                local = np.column_stack((np.zeros(count), direction*caps))
                result = receiving_side_velocities(local, np.ones(count, bool), direction, caps)
                self.assertLessEqual(np.mean(result[:, 1]*direction), -.25*caps.mean()+1e-12)
                self.assertTrue(np.all(np.linalg.norm(result, axis=1) <= caps+1e-12))
                np.testing.assert_array_equal(
                    receiving_side_velocities(local, np.zeros(count, bool), direction, caps), local)


class ButtonSwapMotionTest(unittest.TestCase):
    def test_original_button_controlled_timing(self):
        for level in ('easy', 'medium', 'hard'):
            config = task_config('ButtonUnmaskSwap', level, 0)
            self.assertEqual(config['moving_steps'], 470)
            self.assertEqual(config['reveal_steps'], 10)
            self.assertEqual(config['convergence_steps'], 0)
            for last_press in (50, 230, 479):
                deadline = ButtonDeadline(('left', 'right'), 10, 480, 96)
                deadline.observe(9, {'left': .01}, True, tcp_position=[0, 0, .03])
                self.assertFalse(deadline.presses)
                deadline.observe(10, {'left': .01}, True, tcp_position=[0, 0, .03])
                deadline.observe(last_press, {'right': .01}, True, tcp_position=[0, 0, .03])
                self.assertEqual(deadline.motion_stop, min(last_press+96, 480))
                self.assertFalse(deadline.observe(480, {}, False, tcp_position=[0, 0, .03]))
            missed = ButtonDeadline(('left', 'right'), 10, 480)
            missed.observe(10, {'left': .01}, True, tcp_position=[0, 0, .03])
            self.assertTrue(missed.observe(480, {'right': .01}, False, tcp_position=[0, 0, .03]))
            self.assertNotIn('right', missed.presses)

    def test_latest_counts_and_speed_remain(self):
        expected = {'easy': ((6, 8), (3, 3)), 'medium': ((9, 10), (3, 4)),
                    'hard': ((11, 13), (3, 4))}
        for level, ranges in expected.items():
            seen = [set(), set()]
            passages = set()
            for seed in range(200):
                c = task_config('ButtonUnmaskSwap', level, seed)
                passages.add(c['button_passage_count'])
                self.assertEqual(c['button_passage_count'], task_config('ButtonUnmaskSwap', level, seed)['button_passage_count'])
                self.assertEqual(c['speed'], .00675 if level == 'hard' else .0045)
                counts = c['min_containers'], c['recipe']['total_targets']
                for value, (low, high), values in zip(counts, ranges, seen):
                    self.assertTrue(low <= value <= high)
                    values.add(value)
                self.assertEqual(sum(c['recipe']['target_counts'].values()), counts[1])
            for values, (low, high) in zip(seen, ranges):
                self.assertEqual(values, set(range(low, high+1)))
            self.assertEqual(passages, {2, 3} if level == 'hard' else {2})

    @classmethod
    def setUpClass(cls):
        cls.radius = np.sqrt(2)*.0275
        cls.arguments = dict(count=13, radius=cls.radius, steps=300, speed=.00675,
            obstacles=[([-.2, -.1], .055), ([-.2, .1], .055)],
            extra_obstacles=[([.20, 0.0], np.hypot(.015, .0475))],
            reachable_area=RectangularReachability())
        cls.examples = [plan_routes(seed=seed, passage_count=2 if seed == 0 else 3,
                                    **cls.arguments) for seed in (0, 1, 2)]

    def test_determinism_and_real_passages(self):
        repeated, _ = plan_routes(seed=0, **self.arguments)
        np.testing.assert_array_equal(repeated, self.examples[0][0])
        for path, meta in self.examples:
            self.assertEqual(len(meta['passage_actor_indices']), 2 if meta['seed'] == 0 else 3)
            self.assertEqual(len({r['direction'] for r in meta['passage_routes']}), 1)
            passages = completed_passages(path, meta)
            self.assertGreaterEqual(len(passages), 1)
            self.assertTrue(set(p['actor_index'] for p in passages).issubset(meta['passage_actor_indices']))
            self.assertGreater(len(meta['traffic_recovery_events']), 0)

    def test_all_passage_motion_is_one_way_including_recovery(self):
        for path, meta in self.examples:
            lane = meta['one_way_passage']
            self.assertTrue(passage_forward(path[:-1], path[1:], lane['direction'],
                                           lane['x_max_m'], lane['y_range_m']).all())
            central = meta['central_actor_indices']
            self.assertFalse(passage_intersects(path[:-1, central], path[1:, central],
                                               lane['x_max_m'], lane['y_range_m']).any())
            for event in meta['traffic_recovery_events']:
                self.assertTrue(passage_forward(event['origin_xy_m'], event['goal_xy_m'],
                                                lane['direction'], lane['x_max_m'], lane['y_range_m']))

    def test_translation_and_jitter_do_not_count_as_rearrangement(self):
        initial = np.array([[0., 0.], [.1, .01], [.02, .2], [.22, .21]])
        path = initial[None]+np.linspace(0, .4, 80)[:, None, None]
        audit = rearrangement_audit(path)
        self.assertFalse(any(audit['two_neighbor_set_changed']))
        self.assertEqual(audit['relative_order_changed_with_count'], [0]*4)
        rear = np.array([[-.32, -.22], [-.32, .22]])
        path = np.zeros((100, 1, 2)); path[:, 0, 0] = -.32
        path[:, 0, 1] = .003*np.sin(np.arange(100))
        meta = dict(robot_side_crossing_xy_m=rear, passage_routes=[dict(actor_index=0, direction='right-to-left')])
        self.assertEqual(completed_passages(path, meta), [])

    def test_receiving_side_traffic_is_sent_toward_the_departure_side(self):
        self.assertEqual({m['traffic_direction'] for _, m in self.examples}, {-1, 1})
        for path, meta in self.examples:
            direction = meta['traffic_direction']
            expected = set()
            for step, event in enumerate(meta['receiving_side_velocity_steps']):
                expected = {i for i in expected if path[step, i, 1]*direction >= -.08}
                expected.update(i for i in meta['central_actor_indices'] if path[step, i, 1]*direction > 0)
                self.assertEqual(event['receiving_actor_indices'], sorted(expected))
                if expected:
                    self.assertGreater(event['preferred_sum_toward_a_m_per_step'], 0)
            audit = counterflow_audit(path, meta)
            self.assertGreater(audit['receiving_side_total_motion_toward_a_m'], 0)
            self.assertGreater(audit['receiving_side_reverse_local_moves'], 0)
            self.assertTrue(audit['redistributed_actor_indices'])

    def test_departure_side_does_not_develop_a_stationary_cluster(self):
        for path, meta in self.examples:
            distance = np.linalg.norm(path[:, :, None] - path[:, None, :], axis=3)
            neighbors = ((distance < .13) & (distance > 0)).sum(axis=2)
            on_a = path[:, :, 1] * meta['traffic_direction'] < 0
            for stop in range(60, len(path), 10):
                span = np.linalg.norm(np.ptp(path[stop-60:stop+1], axis=0), axis=1)
                crowded = np.mean(on_a[stop-60:stop+1] & (neighbors[stop-60:stop+1] >= 2), axis=0) > .9
                self.assertFalse(np.any(crowded & (span < .025)), (meta['seed'], stop))

    def test_initial_argument_is_not_mutated(self):
        initial = self.examples[0][0][0].copy(); saved = initial.copy()
        plan_routes(seed=0, initial=initial, **{**self.arguments, 'steps': 40})
        np.testing.assert_array_equal(initial, saved)

    def test_routes_keep_continuous_clearance_and_speed(self):
        np.testing.assert_allclose(np.ptp(CENTRAL_BOUNDS, axis=1), [.58, .60])
        for path, meta in self.examples:
            bounds = np.asarray(meta['initial_bounds_xy_m'])
            self.assertTrue(np.all(path[0] - self.radius >= bounds[:, 0]))
            self.assertTrue(np.all(path[0] + self.radius <= bounds[:, 1]))
            self.assertTrue(np.all(path >= BOUNDS[:, 0]+self.radius))
            self.assertTrue(np.all(path <= BOUNDS[:, 1]-self.radius))
            delta = np.diff(path, axis=0)
            self.assertLessEqual(np.linalg.norm(delta, axis=2).max(), .00675+1e-12)
            for obstacle in meta['obstacles']:
                xy = np.array(obstacle['xy'])
                t = np.clip(np.sum((xy-path[:-1])*delta, axis=2)
                            / np.maximum(np.sum(delta*delta, axis=2), 1e-20), 0, 1)
                gap = np.linalg.norm(path[:-1]+t[..., None]*delta-xy, axis=2)
                self.assertGreaterEqual(gap.min()-obstacle['radius_m']-self.radius, MARGIN_M)
            for before, after in zip(path[:-1], path[1:]):
                self.assertGreaterEqual(swept_gap(before, after, np.full(13, self.radius)), MARGIN_M)
                for a, b in zip(before, after):
                    gap = segment_distance(a, b, np.array([[-.2, -.1]]), np.array([[-.2, .1]]))[0]
                    self.assertGreaterEqual(gap-.055-self.radius, MARGIN_M)


if __name__ == '__main__':
    unittest.main()
