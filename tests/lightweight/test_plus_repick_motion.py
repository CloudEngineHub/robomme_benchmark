"""VideoRepick route geometry, independent curves, and destination timing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robomme_plus.tasks.reference.utils import repick_controller as controller
from robomme_plus.tasks.reference.utils.repick_motion import (
    initial_yaws,
    plan_rotations,
    plan_routes,
    swept_gap,
    yaw_quaternion,
)
from robomme_plus.tasks.reference.utils.repick_tasks import (
    BOARD_SIZE,
    BOARD_X,
    BOARD_Y,
    LEVELS,
    choose_target,
    level_config,
)


class RepickRandomizationTest(unittest.TestCase):
    def test_target_is_uniform_over_every_episode_object_including_extras(self):
        for count in range(4, 11):
            actors = [SimpleNamespace(name=f"bin_{i}" if i < 4 else f"swarm_extra_{i-4}")
                      for i in range(count)]
            counts = dict.fromkeys((actor.name for actor in actors), 0)
            for seed in range(4000):
                chosen = choose_target(actors, seed)
                self.assertTrue(any(chosen is actor for actor in actors))
                counts[chosen.name] += 1
            with self.subTest(count=count):
                frequencies = np.array(list(counts.values())) / 4000
                self.assertTrue(np.all(np.abs(frequencies - 1 / count) < .025))
                self.assertIs(choose_target(actors, 42), choose_target(actors, 42))
                self.assertEqual(len(actors), count)

    def test_initial_yaws_are_independent_uniform_and_reproducible(self):
        angles = np.array([initial_yaws(10, seed) for seed in range(1024)])
        np.testing.assert_array_equal(angles[42], initial_yaws(10, 42))
        self.assertTrue(np.all((angles >= 0) & (angles < 2 * np.pi)))
        self.assertEqual(len(np.unique(angles[42])), 10)
        for column in angles.T:
            histogram, _ = np.histogram(column, bins=12, range=(0, 2 * np.pi))
            self.assertTrue(np.all(np.abs(histogram / len(column) - 1 / 12) < .035))
        correlations = np.corrcoef(angles, rowvar=False)
        self.assertLess(np.abs(correlations[~np.eye(10, dtype=bool)]).max(), .12)


class RepickMotionTest(unittest.TestCase):
    def test_every_level_and_cube_count_keeps_clearance_and_pickup_space(self):
        radius = np.sqrt(2) * .02
        for level in LEVELS:
            config = level_config(level)
            obstacles = [([-.1869, .0122], .055)]
            if level == "hard":
                obstacles.append((
                    [BOARD_X, BOARD_Y],
                    np.linalg.norm(np.asarray(BOARD_SIZE)[:2] / 2),
                ))
            low, high = config["cube_count_range"]
            for count in range(low, high + 1):
                with self.subTest(level=level, count=count):
                    paths, metadata = plan_routes(
                        count, radius, 17642932 + count, steps=820,
                        speed=config["speed"], obstacles=obstacles,
                        workspace_bounds=config["workspace_bounds"],
                        max_group_diameter=config["max_group_diameter"],
                        exchange_destinations=config["exchange_destinations"],
                        settle_steps=config["settle_steps"],
                        settle_spacing=config["settle_spacing"],
                        goal_refresh_range=config["goal_refresh_range"],
                    )
                    bounds = np.asarray(config["workspace_bounds"])
                    self.assertTrue(np.all(paths >= bounds[:, 0] + radius))
                    self.assertTrue(np.all(paths <= bounds[:, 1] - radius))
                    delta = np.diff(paths, axis=0)
                    self.assertLessEqual(np.linalg.norm(delta, axis=2).max(),
                                         config["speed"] + 1e-12)
                    diameters = np.linalg.norm(
                        paths[:, :, None] - paths[:, None, :], axis=3
                    )
                    self.assertLessEqual(diameters.max(), config["max_group_diameter"])
                    for before, after in zip(paths[:-1], paths[1:]):
                        self.assertGreaterEqual(
                            swept_gap(before, after, np.full(count, radius)), .004
                        )
                    # Closest approach along each segment, including endpoints.
                    for center, obstacle_radius in obstacles:
                        center = np.asarray(center)
                        t = np.clip(
                            np.sum((center - paths[:-1]) * delta, axis=2)
                            / np.maximum(np.sum(delta * delta, axis=2), 1e-20), 0, 1
                        )
                        distance = np.linalg.norm(
                            paths[:-1] + t[:, :, None] * delta - center, axis=2
                        )
                        self.assertGreaterEqual(distance.min(), radius + obstacle_radius + .004)
                    final_gaps = diameters[-1][~np.eye(count, dtype=bool)]
                    initial_gaps = diameters[0][~np.eye(count, dtype=bool)]
                    self.assertGreaterEqual(initial_gaps.min(), .10)
                    self.assertGreaterEqual(final_gaps.min(), config["settle_spacing"] - .01)
                    self.assertGreater(min(metadata["route_lengths_m"]), .15)
                    updates = [event["step"] for event in metadata["destination_updates"]]
                    self.assertEqual(updates[0], 0)
                    self.assertTrue(np.all(np.diff(updates) >= 20))
                    self.assertTrue(np.all(np.diff(updates) <= 50))
                    self.assertLess(updates[-1], metadata["settle"]["start_step"])
                    exchanges = metadata["destination_permutations"]
                    self.assertEqual(bool(exchanges), level in ("medium", "hard"))
                    self.assertTrue(all(e["assignment"] != list(range(count)) for e in exchanges))
                    # The actual trajectories change relative ordering.
                    initial_order = np.argsort(paths[0, :, 1])
                    self.assertTrue(any(
                        not np.array_equal(np.argsort(p[:, 1]), initial_order)
                        for p in paths[40:660:40]
                    ))
                    changes = metadata["relative_position_changes"]
                    self.assertGreaterEqual(
                        changes["pairs_changing_distance_by_at_least_4cm"],
                        .8 * changes["pair_count"],
                    )
                    self.assertGreater(changes["median_pair_distance_range_m"], .06)
                    self.assertGreaterEqual(changes["cubes_changing_nearest_neighbor"],
                                            count - 1)

    def test_dense_hard_scene_can_spread_away_from_obstacles_for_pickup(self):
        config = level_config("hard")
        initial = [
            [.052803904, -.202718452], [-.082148246, -.163129687],
            [.072408631, -.016681992], [-.002265504, -.017118251],
            [-.118210405, -.047673728], [.089545652, -.095425919],
            [-.051125143, -.084280506], [-.061032631, .044755474],
            [-.172921628, -.096832804], [.020602548, -.084800109],
        ]
        paths, metadata = plan_routes(
            10, np.sqrt(2) * .02, 39571977, steps=820,
            initial=initial, obstacles=[([-.166629836, .049717687], .055),
                                        ([BOARD_X, BOARD_Y], .027041635)],
            workspace_bounds=config["workspace_bounds"],
            max_group_diameter=config["max_group_diameter"],
            exchange_destinations=True, settle_steps=config["settle_steps"],
            settle_spacing=config["settle_spacing"],
        )
        self.assertGreaterEqual(metadata["settle"]["final_closest_pair_m"], .09)
        self.assertLessEqual(metadata["maximum_planned_group_diameter_m"], .30)
        self.assertGreaterEqual(metadata["minimum_continuous_pair_gap_m"], .004)

    def test_actual_curves_loop_and_reverse_with_independent_timing(self):
        paths, metadata = plan_routes(
            2, .025, 52, steps=420, initial=[[0, -.6], [0, .6]],
            workspace_bounds=[[-1., 1.], [-1., 1.]],
        )
        observed = set()
        for event in metadata["motion_primitives"]:
            start, end = event["start_step"], event["end_step"]
            if end > 420:
                continue
            track = paths[start:end + 1, event["actor"]]
            velocity = np.diff(track, axis=0)
            if event["style"] == "loop":
                radial = track - event["loop_center_xy_m"]
                angle = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
                self.assertGreater(abs(angle[-1] - angle[0]), np.pi)
                observed.add("loop")
            elif event["style"] == "reversal":
                forward = velocity @ np.asarray(event["axis"])
                self.assertGreater(forward.max(), .001)
                self.assertLess(forward.min(), -.001)
                observed.add("reversal")
            elif event["style"] == "s_curve":
                sideways = velocity @ np.asarray(event["normal"])
                self.assertGreater(sideways.max(), .001)
                self.assertLess(sideways.min(), -.001)
                observed.add("s_curve")
        self.assertEqual(observed, {"s_curve", "loop", "reversal"})
        starts = [
            [e["start_step"] for e in metadata["motion_primitives"] if e["actor"] == actor]
            for actor in range(2)
        ]
        self.assertNotEqual(starts[0], starts[1])

    def test_seed_reproducibility_and_invalid_refresh_intervals(self):
        args = dict(count=4, radius=np.sqrt(2) * .02, steps=240,
                    workspace_bounds=[[-.35, .15], [-.25, .25]],
                    max_group_diameter=.30, exchange_destinations=True)
        first, first_metadata = plan_routes(seed=42, **args)
        same, same_metadata = plan_routes(seed=42, **args)
        different, _ = plan_routes(seed=43, **args)
        np.testing.assert_array_equal(first, same)
        self.assertEqual(first_metadata, same_metadata)
        self.assertFalse(np.allclose(first, different))
        for interval in ((0, 50), (50, 20), (20.5, 50), (20,)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                plan_routes(seed=42, goal_refresh_range=interval, **args)


class RepickRotationTest(unittest.TestCase):
    def test_independent_seeded_directions_with_distinct_fixed_slow_speeds(self):
        angles, metadata = plan_rotations(10, 42, 820)
        same, same_metadata = plan_rotations(10, 42, 820)
        other, _ = plan_rotations(10, 43, 820)
        np.testing.assert_array_equal(angles, same)
        self.assertEqual(metadata, same_metadata)
        self.assertFalse(np.allclose(angles, other))
        rates = np.degrees(np.diff(angles, axis=0)) * 20
        self.assertLessEqual(np.abs(rates).max(), 20 + 1e-10)
        self.assertLessEqual(np.abs(np.diff(rates, axis=0)).max(), 1 + 1e-10)
        signs = np.array(metadata["direction_signs"])
        self.assertEqual(set(signs), {-1, 1})
        self.assertTrue(np.all(rates * signs > 0))
        self.assertEqual(metadata["directions"], [
            "counterclockwise" if sign > 0 else "clockwise" for sign in signs
        ])
        speeds = np.array(metadata["angular_speeds_deg_s"])
        self.assertEqual(len(np.unique(speeds)), 10)
        self.assertLess(speeds.min(), 6.5)
        self.assertGreater(speeds.max(), 18.5)
        np.testing.assert_allclose(rates[20:], np.tile(signs * speeds, (800, 1)),
                                   atol=1e-10)
        # Independent choices permit every combination, including all objects
        # turning the same way by chance; do not force alternation or balance.
        patterns = {
            tuple(plan_rotations(4, seed, 2)[1]["direction_signs"])
            for seed in range(256)
        }
        self.assertEqual(len(patterns), 16)
        for actor in range(10):
            self.assertGreater(np.ptp(np.degrees(angles[:, actor])), 20)
            if actor:
                self.assertFalse(np.allclose(angles[:, 0], angles[:, actor]))

    def test_quaternions_preserve_upright_cubes_and_bounding_circle(self):
        from transforms3d.quaternions import quat2mat

        corners = np.array([[x, y, .02] for x in (-.02, .02) for y in (-.02, .02)])
        base = yaw_quaternion([1, 0, 0, 0], .7)
        for angle in np.linspace(-4 * np.pi, 4 * np.pi, 101):
            q = yaw_quaternion(base, angle)
            self.assertAlmostEqual(np.linalg.norm(q), 1)
            rotated = corners @ quat2mat(q).T
            np.testing.assert_allclose(rotated[:, 2], .02)
            np.testing.assert_allclose(np.linalg.norm(rotated[:, :2], axis=1),
                                       np.sqrt(2) * .02)
        np.testing.assert_allclose(yaw_quaternion([1, 0, 0, 0], np.pi),
                                   [0, 0, 0, 1], atol=1e-12)


class Quaternion:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[1., 0., 0., 0.]])


class RepickRotationControllerTest(unittest.TestCase):
    def setUp(self):
        points = np.array([[-.2, -.2], [-.2, 0], [0, -.2], [0, 0]])
        self.actors = [SimpleNamespace(name=str(i), p=np.array([*p, .02]),
                                      pose=SimpleNamespace(q=Quaternion()))
                       for i, p in enumerate(points)]
        self.held = set()
        env = SimpleNamespace(
            seed=42, control_freq=20,
            agent=SimpleNamespace(is_grasping=lambda actor: actor.name in self.held),
            _limit_config={"exchange_destinations": False, "goal_refresh_range": [20, 50]},
        )

        def set_pose(actor, p, q=None):
            actor.p = np.array(p)
            actor.q = np.array(q)

        for item in (
            patch.object(controller, "position", lambda actor, initial=False: actor.p.copy()),
            patch.object(controller, "set_position", set_pose),
            patch.object(controller, "plan_routes", return_value=(
                np.repeat(points[None], 121, axis=0),
                {"max_speed_m_per_control_step": .0045},
            )),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.routes = controller.Routes(env, self.actors, np.sqrt(2) * .02, .0045,
                                        steps=120)

    def test_rotation_continues_during_xy_waits(self):
        positions = np.array([actor.p.copy() for actor in self.actors])
        for _ in range(30):
            self.routes.update()
        np.testing.assert_array_equal([actor.p for actor in self.actors], positions)
        for actor in self.actors:
            self.assertFalse(np.allclose(actor.q, [1, 0, 0, 0]))
            self.assertAlmostEqual(np.linalg.norm(actor.q), 1)
        self.assertFalse(np.allclose(self.actors[0].q, self.actors[1].q))

    def test_held_raised_pinned_and_inactive_cubes_are_not_forced_to_rotate(self):
        self.held.add("0")
        self.routes.pinned.add(1)
        self.actors[2].p[2] = .10
        before = [(actor.p.copy(), actor.q.copy()) for actor in self.actors[:3]]
        for _ in range(30):
            self.routes.update()
        for actor, (p, q) in zip(self.actors, before):
            np.testing.assert_array_equal(actor.p, p)
            np.testing.assert_array_equal(actor.q, q)
        np.testing.assert_array_equal(self.routes.rotation_cursors, [0, 0, 0, 30])
        self.routes.active = False
        q = self.actors[3].q.copy()
        self.routes.update()
        np.testing.assert_array_equal(self.actors[3].q, q)


if __name__ == "__main__":
    unittest.main()
