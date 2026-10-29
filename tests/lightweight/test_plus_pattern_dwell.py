"""PatternLock distinguishes passing from a confirmed stationary stay."""

import unittest

import numpy as np

from robomme_plus.tasks.imitation.utils.pattern_dwell import PatternDwell
from robomme_plus.tasks.imitation.utils.drawing import Drawing
from robomme_plus.tasks.imitation.utils.pattern_layout import global_path, grid_layouts
from robomme_plus.tasks.imitation.utils.geometry import resolve_config


class DrawingIdleTest(unittest.TestCase):
    def drawings(self):
        for task in ('pattern_trace', 'route_trace'):
            for level in ('easy', 'medium', 'hard'):
                options = {'stick_change_version': 4} if task == 'pattern_trace' else {}
                config = resolve_config(task, 97, level, **options)
                yield task, level, Drawing(
                    height_m=config['drawing_height_m'],
                    idle_limit_steps=config['drawing_idle_limit_steps'],
                    max_angle_deg=config.get('drawing_max_angle_deg'),
                )

    def test_initial_wait_allows_450_steps_then_fails(self):
        for task, level, drawing in self.drawings():
            with self.subTest(task=task, level=level):
                drawing.begin('execution', 0)
                for step in range(1, 451):
                    drawing.update(step, [0, 0, .09], 0)
                self.assertIsNone(drawing.failure)
                drawing.update(451, [0, 0, .09], 0)
                self.assertEqual(drawing.failure['reason'], 'idle')
                self.assertEqual(drawing.failure['idle_steps'], 451)

    def test_valid_drawing_resets_the_idle_budget(self):
        for task, level, drawing in self.drawings():
            with self.subTest(task=task, level=level):
                drawing.begin('execution', 0)
                for step in range(1, 441):
                    drawing.update(step, [0, 0, .09], 0)
                self.assertTrue(drawing.update(441, [0, 0, .07], 0))
                for step in range(442, 892):
                    drawing.update(step, [0, 0, .09], 0)
                self.assertIsNone(drawing.failure)
                self.assertTrue(drawing.update(892, [0, 0, .07], 0))
                self.assertEqual(drawing.phases['execution']['idle_steps'], 0)
                for step in range(893, 1344):
                    drawing.update(step, [0, 0, .09], 0)
                self.assertEqual(drawing.failure['reason'], 'idle')

    def test_demonstration_wait_does_not_consume_execution_budget(self):
        drawing = Drawing()
        drawing.begin('demonstration', 0)
        for step in range(1, 501):
            drawing.update(step, [0, 0, .09])
        self.assertIsNone(drawing.failure)
        drawing.begin('execution', 500)
        for step in range(501, 951):
            drawing.update(step, [0, 0, .09])
        self.assertIsNone(drawing.failure)
        self.assertTrue(drawing.update(951, [0, 0, .07]))


class PatternDwellTest(unittest.TestCase):
    def setUp(self):
        self.scorer = PatternDwell([[0, 0], [.09, 0], [.18, 0]], [0, 2])
        self.step = 0

    def sample(self, xyz, count=1):
        for _ in range(count):
            self.scorer.update(self.step, xyz)
            self.step += 1

    def test_six_stationary_intervals_and_duplicate_evaluations(self):
        self.sample([0, 0, .07], 6)
        self.assertEqual(self.scorer.accepted, [])
        for _ in range(20):
            self.scorer.update(self.step - 1, [0, 0, .07])
        self.assertEqual(self.scorer.accepted, [])
        self.sample([0, 0, .07])
        self.assertEqual(self.scorer.accepted, [0])
        self.assertEqual(self.scorer.events[0]['dwell_steps'], 6)
        self.sample([0, 0, .07], 50)
        self.assertIsNone(self.scorer.failure)
        self.assertEqual(len(self.scorer.events), 1)

    def test_cross_wrong_and_correct_targets_without_selecting(self):
        for x in np.arange(-.03, .221, .005):
            self.sample([x, 0, .07])
        self.assertEqual(self.scorer.accepted, [])
        self.assertIsNone(self.scorer.failure)

    def test_intermediate_node_can_be_crossed_between_confirmed_endpoints(self):
        self.sample([0, 0, .07], 7)
        for x in np.arange(.005, .181, .005):
            self.sample([x, 0, .07])
        self.assertEqual(self.scorer.accepted, [0])
        self.sample([.18, 0, .07], 6)
        self.assertTrue(self.scorer.success)
        self.assertEqual(self.scorer.accepted, [0, 2])

    def test_wrong_stop_fails_without_red_confirmation(self):
        self.sample([0, 0, .07], 7)
        self.sample([.09, 0, .07], 7)
        self.assertEqual(self.scorer.failure['reason'], 'wrong_endpoint_stop')
        self.assertEqual(self.scorer.accepted, [0])
        self.sample([.18, 0, .07], 10)
        self.assertFalse(self.scorer.success)
        self.assertEqual(len(self.scorer.events), 1)

    def test_high_tip_and_interrupted_dwell_do_not_count(self):
        self.sample([0, 0, .081], 20)
        self.assertEqual(self.scorer.accepted, [])
        self.sample([0, 0, .07], 5)
        self.sample([.004, 0, .07])  # still inside the target, but too fast
        self.sample([.004, 0, .07], 5)
        self.assertEqual(self.scorer.accepted, [])
        self.sample([.004, 0, .07])
        self.assertEqual(self.scorer.accepted, [0])

    def test_missing_steps_and_new_phase_cannot_inherit_dwell(self):
        self.sample([0, 0, .07], 6)
        self.step += 20
        self.sample([0, 0, .07])
        self.assertEqual(self.scorer.accepted, [])
        self.sample([0, 0, .07], 6)
        self.assertEqual(self.scorer.accepted, [0])
        fresh = PatternDwell([[0, 0], [.09, 0], [.18, 0]], [0, 2])
        self.assertEqual(fresh.accepted, [])
        self.assertIsNone(fresh.held_node)

    def test_returning_to_previous_node_after_departure_is_wrong_stop(self):
        self.sample([0, 0, .07], 7)
        self.sample([.03, 0, .07])
        self.sample([0, 0, .07], 7)
        self.assertEqual(self.scorer.failure['reason'], 'wrong_endpoint_stop')

    def test_entry_interval_does_not_count_toward_dwell(self):
        for outside, inside in (
            ([.0105, 0, .07], [.0095, 0, .07]),
            ([0, 0, .0805], [0, 0, .0795]),
        ):
            with self.subTest(outside=outside):
                scorer = PatternDwell([[0, 0]], [0])
                scorer.update(0, outside)
                for step in range(1, 7):
                    scorer.update(step, inside)
                self.assertFalse(scorer.success)  # Only 0.25 s inside.
                scorer.update(7, inside)
                self.assertTrue(scorer.success)

    def test_wrong_stop_also_requires_full_dwell_after_entry(self):
        self.sample([.1005, 0, .07])
        self.sample([.0995, 0, .07], 6)
        self.assertIsNone(self.scorer.failure)
        self.sample([.0995, 0, .07])
        self.assertEqual(self.scorer.failure['reason'], 'wrong_endpoint_stop')


class PatternSegmentsTest(unittest.TestCase):
    def setUp(self):
        self.xy = np.array(grid_layouts(42, 5, .09, [-.13, 0])['execution']['positions_xy_m'])
        self.nodes = [0, 4, 24, 20]
        self.scorer = PatternDwell(self.xy, self.nodes)
        self.drawing = Drawing()
        self.drawing.begin('execution', 0)
        self.step = 0
        self.tip = np.r_[self.xy[self.nodes[0]], .07]

    def sample(self, tip, count=1):
        for _ in range(count):
            self.scorer.update(self.step, tip)
            self.drawing.update(self.step, tip)
            self.step += 1
        self.tip = np.asarray(tip)

    def move(self, goal):
        start = self.tip.copy()
        count = max(1, int(np.ceil(np.linalg.norm(goal - start) / .005)))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            self.sample(start + fraction * (goal - start))

    def test_continuous_detour_cannot_pass_with_correct_endpoints(self):
        self.sample(self.tip, 7)
        for node in self.nodes[1:]:
            goal = np.r_[self.xy[node], .07]
            delta = goal[:2] - self.tip[:2]
            detour = (self.tip + goal) / 2
            detour[:2] += .15 * np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
            self.move(detour)
            self.move(goal)
            self.sample(goal, 6)
        self.assertIsNone(self.drawing.failure)
        self.assertFalse(self.scorer.success)
        self.assertEqual(self.scorer.accepted, [0])
        self.assertEqual(self.scorer.failure['reason'], 'wrong_drawing_pattern')

    def test_straight_segments_pass_on_shifted_rotated_grid(self):
        self.sample(self.tip, 7)
        for node in self.nodes[1:]:
            goal = np.r_[self.xy[node], .07]
            self.move(goal)  # Cross intermediate grid nodes without stopping.
            self.sample(goal, 6)
        self.assertTrue(self.scorer.success)
        self.assertEqual(self.scorer.accepted, self.nodes)
        self.assertIsNone(self.drawing.failure)

    def test_later_segment_cannot_replace_current_segment(self):
        self.sample(self.tip, 7)
        # This is on the final segment, but the next endpoint must be node 4.
        self.move(np.r_[(self.xy[0] + self.xy[20]) / 2, .07])
        self.assertEqual(self.scorer.failure['reason'], 'wrong_drawing_pattern')

    def test_collinear_overshoot_is_outside_the_segment(self):
        self.sample(self.tip, 7)
        delta = self.xy[4] - self.xy[0]
        self.move(np.r_[self.xy[4] + .03 * delta / np.linalg.norm(delta), .07])
        self.assertEqual(self.scorer.failure['reason'], 'wrong_drawing_pattern')

    def test_small_tracking_error_is_allowed(self):
        delta = self.xy[4] - self.xy[0]
        offset = .008 * np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        self.tip[:2] += offset
        self.sample(self.tip, 7)
        self.move(np.r_[self.xy[4] + offset, .07])
        self.sample(self.tip, 6)
        self.assertEqual(self.scorer.accepted, [0, 4])
        self.assertIsNone(self.scorer.failure)

    def test_segment_tracking_tolerance_is_distinct_from_endpoint_radius(self):
        delta = self.xy[4] - self.xy[0]
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        midpoint = (self.xy[0] + self.xy[4]) / 2
        self.sample(self.tip, 7)
        self.move(np.r_[midpoint + .019 * normal, .07])
        self.assertIsNone(self.scorer.failure)
        self.move(np.r_[midpoint + .021 * normal, .07])
        self.assertEqual(self.scorer.failure['reason'], 'wrong_drawing_pattern')


class PatternLevelsTest(unittest.TestCase):
    def test_level_features_and_seeded_global_routes(self):
        for seed in range(100):
            for level in ('easy', 'medium', 'hard'):
                config = resolve_config('pattern_trace', seed, level)
                self.assertEqual(config, resolve_config('pattern_trace', seed, level))
                self.assertEqual(config['board'], level == 'hard')
                self.assertEqual(config['route_mode'], 'adjacent' if level == 'easy' else 'any_grid')
                self.assertEqual(config['shorten_stick'], level != 'easy')
                fraction = config['execution_stick_fraction']
                self.assertTrue(fraction == 1 if level == 'easy' else .25 <= fraction <= .5)
                layouts = grid_layouts(seed, 5, .09, [-.13, 0], rotate=config['rotate_grid'])
                angle = layouts['execution']['yaw_deg']
                self.assertTrue(angle == 0 if level == 'easy' else 5 <= abs(angle) <= 10)
                if level != 'easy':
                    path = global_path(seed, 5, config['path_node_count'])
                    self.assertEqual(path, global_path(seed, 5, config['path_node_count']))
                    self.assertEqual(len(set(path)), len(path))
                    self.assertEqual(len(path), config['path_node_count'])
                    self.assertTrue(any(max(abs(a//5-b//5), abs(a%5-b%5)) > 1
                                        for a, b in zip(path, path[1:])))


if __name__ == '__main__':
    unittest.main()
