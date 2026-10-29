"""Local rearrangements change relative positions quickly without touching the robot."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robomme_plus.tasks.reference.utils.highlight_motion import (
    DURATION, INTERVAL_PER_LEVEL, MIN_DISPLACEMENT, START_CLEARANCE, STOP_CLEARANCE,
    LocalAdjustments, candidate_groups, changed_arrangement,
    clear_path, group_path, plan_adjustment,
)

INTERVAL = INTERVAL_PER_LEVEL['hard']


class Visible:
    def capture(self):
        pass

    def measure(self, center, quaternion):
        return dict(allowed=True, total_target_pixels=100, arm_occluded_pixels=0,
                    arm_occluded_fraction=0.)


def controller(seed=3, level="hard"):
    return LocalAdjustments(seed, .02, level=level, start_step=120, visibility=Visible())


class Actor:
    def __init__(self, xy, z=.02):
        self.pose = SimpleNamespace(p=torch.tensor([[*xy, z]]), q=torch.tensor([[1., 0., 0., 0.]]))

    def set_pose(self, pose):
        self.pose = SimpleNamespace(p=torch.tensor(pose.p[None]), q=torch.tensor(pose.q[None]))

    def set_linear_velocity(self, value):
        self.linear_velocity = value

    def set_angular_velocity(self, value):
        self.angular_velocity = value


def scene(count=2):
    angle = np.arange(count)*2*np.pi/count
    positions = np.array([-.12, 0]) + .08*np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    cubes = [Actor(xy) for xy in positions]
    held = Actor([-.34, -.20], z=.12)
    return SimpleNamespace(
        all_cubes=cubes + [held], target_cubes=cubes, _highlight_picks=[held],
        elapsed_steps=torch.tensor([120]), failureflag=torch.tensor([False]),
        successflag=torch.tensor([False]), button=Actor([-.31, .20]),
        agent=SimpleNamespace(tcp=held, is_grasping=lambda cube: torch.tensor([cube is held])),
    )


def positions(env):
    return np.array([c.pose.p.numpy()[0, :2] for c in env.all_cubes])


def three_group_scene():
    env = scene()
    env.all_cubes = [Actor([x, y]) for y in (-.17, 0, .17) for x in (-.20, -.06)]
    env.agent.tcp = Actor([-.6, -.6])
    env.agent.is_grasping = lambda cube: torch.tensor([False])
    env.button = Actor([-.5, .4])
    return env


class HighlightMotionTest(unittest.TestCase):
    def test_two_to_five_cube_routes_reach_independently_chosen_destinations(self):
        for count in (2, 3, 4, 5):
            env = scene(count)
            start = positions(env)[:count]
            goal = np.roll(start, count//2, axis=0) + [.012, -.006]
            path = group_path(start, goal, np.random.default_rng(3), arc=True)
            self.assertEqual(path.shape, (DURATION+1, count, 2))
            np.testing.assert_array_equal(path[0], start)
            np.testing.assert_array_equal(path[-1], goal)
            self.assertTrue(changed_arrangement(start, goal))

    def test_reject_small_drifts_and_shared_translation(self):
        start = positions(scene(3))[:3]
        self.assertFalse(changed_arrangement(start, start + [.04, -.02]))
        self.assertFalse(changed_arrangement(start, start + [.16, 0]))

    def test_planning_is_seeded_and_accepts_no_goal_labels(self):
        env = scene(5)
        start = positions(env)[:5]
        args = (start, list(range(5)), list(range(5)), np.array([-.31, .2]),
                np.array([-.34, -.2]), np.sqrt(2)*.02)
        first = plan_adjustment(*args, np.random.default_rng(7))
        second = plan_adjustment(*args, np.random.default_rng(7))
        self.assertIsNotNone(first)
        self.assertTrue(2 <= len(first['indices']) <= 5)
        self.assertEqual(first['indices'], second['indices'])
        np.testing.assert_array_equal(first['path'], second['path'])
        path = first['path']
        self.assertTrue(changed_arrangement(path[0], path[-1]))
        stationary = start[[i for i in range(5) if i not in first['indices']]]
        self.assertTrue(clear_path(path, stationary, args[3], args[4], args[5], endpoint_gap=args[5]))
        self.assertGreaterEqual(np.linalg.norm(path[-1]-path[0], axis=1).min(), MIN_DISPLACEMENT)
        self.assertTrue((np.linalg.norm(path[-1, :, None]-start, axis=-1).min(axis=1) > .01).all())

    def test_continuous_collision_checks_reject_crossings_between_clear_endpoints(self):
        crossing = np.array([[[-.20, 0], [-.04, 0]], [[-.04, 0], [-.20, 0]]])
        self.assertFalse(clear_path(crossing, [], [-.31, .2], [-.34, -.2], np.sqrt(2)*.02))
        path = np.array([[[-.2, -.2]], [[-.2, .2]]])
        self.assertFalse(clear_path(path, [], [.04, .2], [-.2, 0], np.sqrt(2)*.02))
        self.assertFalse(clear_path(path, [], [-.2, 0], [-.5, -.5], np.sqrt(2)*.02))

    def test_five_cube_pool_has_a_clear_substantial_rearrangement(self):
        start = positions(scene(5))[:5]
        plan = plan_adjustment(start, range(5), range(5), np.array([-.31, .2]),
                               np.array([-.34, -.2]), np.sqrt(2)*.02, np.random.default_rng(7))
        self.assertIsNotNone(plan)
        self.assertTrue(2 <= len(plan['indices']) <= 5)
        self.assertTrue(changed_arrangement(plan['path'][0], plan['path'][-1]))
        self.assertTrue(clear_path(plan['path'], [], [-.31, .2], [-.34, -.2], np.sqrt(2)*.02))

    def test_event_finishes_in_40_steps_and_leaves_held_cube_untouched(self):
        env = scene()
        controller = LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        held_before = positions(env)[-1].copy()
        orientations_before = [c.pose.q.clone() for c in env.all_cubes]
        for step in range(120, 120+DURATION):
            env.elapsed_steps[0] = step
            controller.tick(env)
        self.assertFalse(controller.active)
        self.assertEqual(len(controller.events), 1)
        event = controller.events[0]
        self.assertEqual(event['outcome'], 'completed')
        self.assertEqual(event['movement_steps'], DURATION)
        self.assertEqual(event['end_step'] - event['start_step'], DURATION)
        np.testing.assert_array_equal(positions(env)[-1], held_before)
        for i in event['cube_indices']:
            self.assertFalse(torch.allclose(env.all_cubes[i].pose.q, orientations_before[i]))
        self.assertTrue(torch.equal(env.all_cubes[-1].pose.q, orientations_before[-1]))

    def test_events_repeat_without_pickups_or_a_three_event_cap(self):
        env = scene()
        env._highlight_picks = []
        controller = LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        for step in range(120, 120+6*(DURATION+INTERVAL)):
            env.elapsed_steps[0] = step
            controller.tick(env)
        completed = [e for e in controller.events if e['outcome'] == 'completed']
        self.assertGreaterEqual(len(completed), 4)
        self.assertEqual(completed[1]['start_step'], completed[0]['end_step'] + INTERVAL)

    def test_forty_stationary_steps_before_next_batch(self):
        env, motion = scene(), controller()
        for step in range(120, 120+DURATION):
            env.elapsed_steps[0] = step
            motion.tick(env)
        before = positions(env).copy()
        for step in range(120+DURATION, 120+DURATION+INTERVAL):
            env.elapsed_steps[0] = step
            motion.tick(env)
            self.assertEqual(len(motion.events), 1)
            np.testing.assert_array_equal(positions(env), before)
        env.elapsed_steps[0] = 120+DURATION+INTERVAL
        motion.tick(env)
        self.assertEqual(len(motion.events), 2)
        self.assertEqual(motion.events[1]['start_step'], 120+DURATION+INTERVAL)

    def test_level_specific_pause_after_completed_movement(self):
        for level, interval in [('easy', 100), ('medium', 60), ('hard', 40)]:
            env, motion = scene(), controller(level=level)
            for step in range(120, 120+DURATION):
                env.elapsed_steps[0] = step
                motion.tick(env)
            self.assertEqual(motion.events[0]['outcome'], 'completed')
            end = motion.events[0]['end_step']
            before = positions(env).copy()
            for step in range(end, end+interval):
                env.elapsed_steps[0] = step
                motion.tick(env)
                self.assertEqual(len(motion.events), 1)
                np.testing.assert_array_equal(positions(env), before)
            env.elapsed_steps[0] = end+interval
            motion.tick(env)
            self.assertEqual(motion.events[1]['start_step'], end+interval)

    def test_level_specific_pause_after_last_cluster_is_interrupted(self):
        for level, interval in [('easy', 100), ('medium', 60), ('hard', 40)]:
            env, motion = scene(), controller(level=level)
            motion.tick(env)
            far_tcp = env.agent.tcp.pose.p.clone()
            env.agent.tcp.pose.p[0, :2] = env.all_cubes[0].pose.p[0, :2]
            env.elapsed_steps[0] = 121
            motion.tick(env)
            self.assertEqual(motion.events[0]['outcome'], 'stopped_for_clearance')
            self.assertEqual(motion.next_attempt, 121+interval)
            env.agent.tcp.pose.p = far_tcp
            for step in range(122, 121+interval):
                env.elapsed_steps[0] = step
                motion.tick(env)
                self.assertEqual(len(motion.events), 1)
            env.elapsed_steps[0] = 121+interval
            motion.tick(env)
            self.assertEqual(motion.events[1]['start_step'], 121+interval)

    def test_batches_choose_one_to_three_with_seeded_timing(self):
        outcomes = []
        for _ in range(2):
            env, motion = scene(), controller()
            for step in range(120, 120+6*(DURATION+INTERVAL)):
                env.elapsed_steps[0] = step
                motion.tick(env)
            outcomes.append([(e['batch_id'], e['requested_clusters'], e['start_step'])
                             for e in motion.events])
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual({row[1] for row in outcomes[0]}, {1, 2, 3})

    def test_difficulty_controls_simultaneous_cluster_count_and_group_size(self):
        for level, allowed, maximum in [('easy', {1, 2}, 3), ('medium', {1, 2, 3}, 4), ('hard', {1, 2, 3}, 5)]:
            requested = []
            for seed in range(30):
                env, motion = three_group_scene(), controller(seed, level)
                motion.tick(env)
                requested.append(motion.events[0]['requested_clusters'])
                self.assertLessEqual(len(motion.active), requested[-1])
                for group in motion.active:
                    self.assertTrue(2 <= len(group['indices']) <= maximum)
                if requested[-1] == 1:
                    self.assertEqual(len(motion.active), 1)
            self.assertEqual(set(requested), allowed)

    def test_each_level_samples_only_its_allowed_cluster_sizes(self):
        for level, allowed in [('easy', {2, 3}), ('medium', {2, 3, 4}), ('hard', {2, 3, 4, 5})]:
            sizes = set()
            for count in sorted(allowed):
                for seed in (1, 7):
                    env, motion = scene(count), controller(seed, level)
                    motion.cluster_range = (1, 1)
                    motion.tick(env)
                    self.assertEqual(len(motion.events), 1)
                    size = len(motion.events[0]['cube_indices'])
                    self.assertIn(size, allowed)
                    sizes.add(size)
            self.assertEqual(sizes, allowed)

    def test_waiting_increases_anchor_and_partner_selection(self):
        points = np.array([[x, y] for x in (-.2, -.06) for y in (-.07, .07)])
        counts = {}
        for skipped in (0, 3):
            anchors, selected, partners = np.zeros(4, int), np.zeros(4, int), np.zeros(4, int)
            for seed in range(2000):
                ids, anchor, _ = next(candidate_groups(
                    points, range(4), np.random.default_rng(seed), 2, {0: skipped}))
                anchors[anchor] += 1
                selected[ids] += 1
                if anchor == 1:
                    partners[next(i for i in ids if i != anchor)] += 1
            counts[skipped] = anchors, selected, partners
        # Weight 4 vs 1 gives anchor probability 4/7 (baseline 1/4).
        self.assertTrue(.53 < counts[3][0][0]/2000 < .61)
        self.assertTrue(.22 < counts[0][0][0]/2000 < .28)
        self.assertGreater(counts[3][1][0], counts[0][1][0] * 1.4)
        self.assertTrue((counts[3][0] > 150).all())  # Others retain a chance.
        # With another anchor fixed, cube 0 also gets more partner selections.
        self.assertGreater(counts[3][2][0]/counts[3][2].sum(), .6)
        self.assertLess(counts[0][2][0]/counts[0][2].sum(), .4)

    def test_participation_counts_once_per_started_event_and_resets(self):
        env, motion = scene(), controller()
        for step in range(120, 120+DURATION):
            env.elapsed_steps[0] = step
            motion.tick(env)
        self.assertEqual(motion.cube_uses, {0: 1, 1: 1})
        self.assertEqual(motion.last_moved_batch, {0: 1, 1: 1})
        self.assertEqual(motion.events[0]['selection_weights'], [1, 1])
        env.elapsed_steps[0] = 120+DURATION+INTERVAL
        motion.tick(env)
        self.assertEqual(motion.cube_uses, {0: 2, 1: 2})
        self.assertEqual(motion.events[1]['selection_weights'], [1, 1])
        self.assertEqual(motion.last_moved_batch, {0: 2, 1: 2})
        fresh = controller()
        self.assertEqual(fresh.cube_uses, {})
        self.assertEqual(fresh.last_moved_batch, {})

    def test_plan_blocked_before_first_movement_keeps_waiting_priority(self):
        env, motion = scene(), controller()
        initial = positions(env)
        def measure(center, quaternion):
            return dict(allowed=bool((np.linalg.norm(initial-np.asarray(center)[:2], axis=1) < 1e-6).any()))
        with patch.object(motion.visibility, 'measure', side_effect=measure):
            motion.tick(env)
        self.assertEqual(len(motion.events), 1)
        self.assertEqual(motion.events[0]['outcome'], 'stopped_for_arm_occlusion')
        self.assertEqual(motion.cube_uses, {})
        self.assertEqual(motion.last_moved_batch, {})
        np.testing.assert_array_equal(positions(env), initial)
        env.elapsed_steps[0] = motion.next_attempt
        motion.tick(env)
        self.assertEqual(motion.events[1]['selection_weights'], [2, 2])
        self.assertEqual(motion.last_moved_batch, {0: 2, 1: 2})

    def test_unselected_cubes_gain_priority_while_moved_cubes_reset(self):
        env, motion = three_group_scene(), controller()
        motion.cluster_range = (1, 1)
        waits = []

        def choose_pair(pos, eligible, *args, **kwargs):
            waits.append(dict(kwargs['cube_waits']))
            pair = (0, 1) if motion.batch_id == 1 else (2, 3)
            return plan_adjustment(pos, [i for i in eligible if i in pair], *args, **kwargs)

        with patch('robomme_plus.tasks.reference.utils.highlight_motion.plan_adjustment',
                   side_effect=choose_pair):
            for step in range(120, 120+DURATION):
                env.elapsed_steps[0] = step
                motion.tick(env)
            env.elapsed_steps[0] = motion.next_attempt
            motion.tick(env)
        self.assertEqual(waits, [dict.fromkeys(range(6), 0),
                                 {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1}])
        self.assertEqual(motion.events[1]['cube_indices'], [2, 3])
        self.assertEqual(motion.events[1]['selection_weights'], [2, 2])
        self.assertEqual(motion.last_moved_batch, {0: 1, 1: 1, 2: 2, 3: 2})

    def test_failed_planning_rounds_accumulate_priority(self):
        env, motion = scene(), controller()
        with patch('robomme_plus.tasks.reference.utils.highlight_motion.plan_adjustment',
                   return_value=None):
            for step in (120, 120+INTERVAL, 120+2*INTERVAL):
                env.elapsed_steps[0] = step
                motion.tick(env)
        self.assertFalse(motion.events)
        env.elapsed_steps[0] = 120+3*INTERVAL
        motion.tick(env)
        self.assertEqual(motion.events[0]['selection_weights'], [4, 4])

    def test_safety_excluded_cubes_gain_priority_without_being_moved(self):
        env, motion = scene(), controller()
        far_tcp = env.agent.tcp.pose.p.clone()
        env.agent.tcp.pose.p[0, :2] = torch.tensor([-.12, 0])
        for step in (120, 120+INTERVAL):
            env.elapsed_steps[0] = step
            motion.tick(env)
        self.assertFalse(motion.events)
        env.agent.tcp.pose.p = far_tcp
        env.elapsed_steps[0] = 120+2*INTERVAL
        motion.tick(env)
        self.assertEqual(motion.events[0]['selection_weights'], [3, 3])

    def test_smaller_gripper_exclusion_still_protects_nearby_cubes(self):
        path = np.array([[[-.12, 0]], [[-.12, 0]]])
        self.assertTrue(clear_path(path, [], [-.5, .4], [-.22, 0], .028))  # 10 cm.
        self.assertFalse(clear_path(path, [], [-.5, .4], [-.20, 0], .028))  # 8 cm start.
        self.assertTrue(clear_path(path, [], [-.5, .4], [-.20, 0], .028,
                                   gripper_clearance=STOP_CLEARANCE))
        self.assertFalse(clear_path(path, [], [-.5, .4], [-.17, 0], .028,
                                    gripper_clearance=STOP_CLEARANCE))  # 5 cm stop.

    def test_previously_picked_cubes_can_move_again_after_putdown(self):
        env, controller = scene(), LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        env._highlight_picks = list(env.all_cubes)
        controller.tick(env)
        self.assertTrue(controller.active)
        self.assertEqual(set(controller.active[0]['indices']), {0, 1})

    def test_gripper_approach_stops_in_place_without_finishing_the_swap(self):
        env = scene()
        controller = LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        for step in range(120, 140):
            env.elapsed_steps[0] = step
            controller.tick(env)
        env.agent.tcp.pose.p[0, :2] = env.all_cubes[0].pose.p[0, :2]
        before = positions(env).copy()
        env.elapsed_steps[0] = 140
        controller.tick(env)
        self.assertFalse(controller.active)
        self.assertEqual(controller.events[0]['outcome'], 'stopped_for_clearance')
        np.testing.assert_array_equal(positions(env), before)

    def test_does_not_start_when_all_resting_cubes_are_near_the_gripper(self):
        env, controller = scene(), LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        env.agent.tcp.pose.p[0, :2] = torch.tensor([-.12, 0])
        before = positions(env).copy()
        for step in range(120, 155):
            env.elapsed_steps[0] = step
            controller.tick(env)
        self.assertEqual(controller.events, [])
        np.testing.assert_array_equal(positions(env), before)

    def test_no_movement_before_highlights_end_but_no_pickup_required(self):
        env = scene()
        controller = LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        env.elapsed_steps[0] = 100
        controller.tick(env)
        self.assertFalse(controller.active)
        env.elapsed_steps[0] = 120
        controller.tick(env)
        self.assertTrue(controller.active)
        env = scene()
        env._highlight_picks = []
        controller = LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        controller.tick(env)
        self.assertTrue(controller.active)

    def test_first_movement_follows_last_cue_in_both_highlight_modes(self):
        from robomme_plus.tasks.reference.utils.highlight_timing import highlight_schedule

        for mode in ('simultaneous', 'staggered'):
            schedule = highlight_schedule(3, 5, mode=mode, duration_steps=30)
            last = max(t['end_step'] for t in schedule)
            env = scene()
            motion = LocalAdjustments(3, .02, level="hard", start_step=last+1, visibility=Visible())
            before = positions(env).copy()
            for step in range(last+1):
                env.elapsed_steps[0] = step
                motion.tick(env)
                self.assertFalse(motion.events)
                np.testing.assert_array_equal(positions(env), before)
            env.elapsed_steps[0] = last+1
            motion.tick(env)
            self.assertEqual(motion.events[0]['start_step'], last+1)

    def test_extended_highlight_window_does_not_start_motion_at_120(self):
        env = scene()
        motion = LocalAdjustments(3, .02, level="hard", start_step=211, visibility=Visible())
        for step in (120, 160, 210):
            env.elapsed_steps[0] = step
            motion.tick(env)
            self.assertFalse(motion.events)
        env.elapsed_steps[0] = 211
        motion.tick(env)
        self.assertEqual(motion.events[0]['start_step'], 211)

    def test_visible_cue_blocks_motion_even_after_scheduled_end(self):
        env = scene()
        cue_id = id(env.target_cubes[0])
        env._highlight_cache = {cue_id: {'is_highlighted': True}}
        motion = LocalAdjustments(3, .02, level="hard", start_step=55, visibility=Visible())
        env.elapsed_steps[0] = 55
        motion.tick(env)
        self.assertFalse(motion.events)
        env._highlight_cache[cue_id]['is_highlighted'] = False
        env.elapsed_steps[0] = 56
        motion.tick(env)
        self.assertEqual(motion.events[0]['start_step'], 56)

    def test_stale_cue_cache_from_previous_scene_does_not_block_current_targets(self):
        env = scene()
        env._highlight_cache = {id(Actor([0, 0])): {'is_highlighted': True}}
        motion = LocalAdjustments(3, .02, level="hard", start_step=55, visibility=Visible())
        env.elapsed_steps[0] = 55
        motion.tick(env)
        self.assertEqual(motion.events[0]['start_step'], 55)

    def test_terminal_episode_stops_movement_and_does_not_restart(self):
        env, controller = scene(), LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
        controller.tick(env)
        before = positions(env).copy()
        env.successflag[0] = True
        env.elapsed_steps[0] += 1
        controller.tick(env)
        self.assertFalse(controller.active)
        self.assertEqual(controller.events[-1]['outcome'], 'episode_finished')
        for step in range(130, 150):
            env.elapsed_steps[0] = step
            controller.tick(env)
        np.testing.assert_array_equal(positions(env), before)
        self.assertEqual(len(controller.events), 1)

    def test_reset_reproduces_the_same_first_event(self):
        outcomes = []
        for _ in range(2):
            env, controller = scene(), LocalAdjustments(3, .02, level="hard", start_step=120, visibility=Visible())
            controller.tick(env)
            outcomes.append(controller.events)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_three_disjoint_groups_start_together_and_complete_safely(self):
        env, motion = three_group_scene(), controller(4)
        motion.cluster_range = (3, 3)
        motion.tick(env)
        self.assertEqual(len(motion.active), 3)
        ids = [i for group in motion.active for i in group['indices']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({e['start_step'] for e in motion.events}, {120})
        for step in range(121, 120+DURATION):
            before = positions(env)
            env.elapsed_steps[0] = step
            motion.tick(env)
            self.assertTrue(clear_path(np.stack((before, positions(env))), [],
                                       [-.5, .4], [-.6, -.6], motion.radius))
        self.assertEqual([e['outcome'] for e in motion.events], ['completed']*3)

    def test_candidate_regions_include_center_without_corner_priority(self):
        points = np.array([[x, y] for x in (-.28, -.13, .02) for y in (-.18, 0, .18)])
        counts = np.zeros((3, 3), int)
        for seed in range(900):
            _, _, cell = next(candidate_groups(points, range(9), np.random.default_rng(seed), 2, {}))
            counts[cell] += 1
        self.assertTrue((counts > 65).all(), counts)
        self.assertTrue((counts < 135).all(), counts)

    def test_reserved_paths_reject_cross_group_collision_between_frames(self):
        one = np.array([[[-.22, 0]], [[-.06, 0]]])
        other = np.array([[[-.06, 0]], [[-.22, 0]]])
        self.assertTrue(clear_path(one, [], [-.5, .4], [-.6, -.6], .028))
        self.assertFalse(clear_path(one, [], [-.5, .4], [-.6, -.6], .028,
                                    moving_obstacles=other))

    def test_occluded_cubes_do_not_start(self):
        env, motion = scene(), controller()
        with patch.object(motion.visibility, 'measure', return_value={'allowed': False}):
            motion.tick(env)
        self.assertEqual(motion.events, [])

    def test_occlusion_stops_only_affected_group_and_preserves_its_pose(self):
        env, motion = three_group_scene(), controller(4)
        motion.tick(env)
        blocked = motion.active[0]
        before = positions(env).copy()
        hidden_xy = before[blocked['indices'][0]]
        def measure(center, quaternion):
            blocked = np.linalg.norm(np.asarray(center)[:2]-hidden_xy) < .001
            return dict(allowed=not blocked, arm_occluded_fraction=.6 if blocked else 0.)
        env.elapsed_steps[0] += 1
        with patch.object(motion.visibility, 'measure', side_effect=measure):
            motion.tick(env)
        self.assertEqual(blocked['event']['outcome'], 'stopped_for_arm_occlusion')
        np.testing.assert_array_equal(positions(env)[blocked['indices']], before[blocked['indices']])
        self.assertTrue(motion.active)
        self.assertTrue(any(e['movement_steps'] == 2 for e in motion.events))

    def test_next_pose_is_checked_before_moving_into_occlusion(self):
        env, motion = scene(), controller()
        motion.tick(env)
        before = positions(env).copy()
        def measure(center, quaternion):
            # Current cube poses are visible, but the next poses are covered.
            return dict(allowed=bool((np.linalg.norm(before-np.asarray(center)[:2], axis=1) < 1e-6).any()))
        env.elapsed_steps[0] += 1
        with patch.object(motion.visibility, 'measure', side_effect=measure):
            motion.tick(env)
        self.assertFalse(motion.active)
        np.testing.assert_array_equal(positions(env), before)
        self.assertEqual(motion.events[0]['outcome'], 'stopped_for_arm_occlusion')

    def test_stopped_group_becomes_obstacle_before_other_group_moves(self):
        env, motion = three_group_scene(), controller()
        env.all_cubes = [Actor([x, y]) for x in (-.12, -.21) for y in (-.16, .16)]
        initial = positions(env)
        motion.next_attempt = 1000
        for ids in ([0, 1], [2, 3]):
            path = np.repeat(initial[ids][None], DURATION+1, axis=0)
            path[1:] += [.08, 0]
            event = dict(start_step=120, outcome='moving')
            motion.events.append(event)
            motion.active.append(dict(indices=ids, path=path, progress=0, event=event,
                                      z=np.full(2, .02), yaw=np.zeros(2), q=[[1, 0, 0, 0]]*2))
        env.agent.is_grasping = lambda cube: torch.tensor([cube is env.all_cubes[0]])
        motion.tick(env)
        self.assertFalse(motion.active)
        self.assertEqual([e['outcome'] for e in motion.events],
                         ['stopped_for_contact', 'stopped_for_clearance'])
        np.testing.assert_array_equal(positions(env), initial)


if __name__ == '__main__':
    unittest.main()
