"""A landing must identify both the requested cube and its demonstrated ring."""

from types import SimpleNamespace
import unittest
import numpy as np
import torch

from robomme_plus.scoring.predicates import _coerce_failure_result, sequential_task_check
from robomme_plus.tasks.reference.utils.place_scoring import bind_placement_checks, placement_events


class PlacementIdentity(unittest.TestCase):
    def setUp(self):
        self.cubes = [SimpleNamespace(name=name) for name in ("red", "blue")]
        self.target = SimpleNamespace(name="ring")
        self.native_failed = False
        self.held = None
        tasks = []
        for cube, demo in ((self.cubes[1], True), (self.cubes[0], False)):
            tasks.extend([
                dict(name="pick up the cube", segment=cube, demonstration=demo, func=lambda: False),
                dict(name="drop onto target", segment=self.target, demonstration=demo,
                     func=lambda: False, failure_func=lambda: self.native_failed, failure_reason="wrong_placement"),
            ])
        self.env = SimpleNamespace(
            all_cubes=self.cubes, targets=[self.target], task_list=tasks,
            _limit_placement_events=[], agent=SimpleNamespace(is_grasping=lambda c: c is self.held),
        )
        bind_placement_checks(self.env)

    def event(self, index, cube, target="ring"):
        self.env._limit_placement_events.append(dict(task_index=index, cube_name=cube, target_name=target))

    def test_correct_cube_and_ring_are_required_for_each_demo_and_execution(self):
        for index, correct, wrong in ((1, "blue", "red"), (3, "red", "blue")):
            with self.subTest(index=index):
                entry = self.env.task_list[index]
                self.event(index, wrong)
                self.assertFalse(entry["func"]())
                self.env._limit_placement_events.clear()
                self.event(index, correct)
                self.assertTrue(entry["func"]())

    def test_wrong_cube_on_correct_ring_fails_with_cube_reason(self):
        self.event(3, "blue")
        self.env.timestep = 3
        done, _, failed, _ = sequential_task_check(self.env, self.env.task_list, True)
        self.assertFalse(done)
        self.assertTrue(failed)
        self.assertEqual(self.env._failure_code, "wrong_cube")

    def test_wrong_ring_or_no_ring_fails_with_placement_reason(self):
        for target in ("other_ring", None):
            with self.subTest(target=target):
                self.setUp()
                self.event(3, "red", target)
                self.env.timestep = 3
                self.assertTrue(sequential_task_check(self.env, self.env.task_list, True)[2])
                self.assertEqual(self.env._failure_code, "wrong_placement")

    def test_failure_checks_survive_rebinding_and_wrong_cube_pickups(self):
        bind_placement_checks(self.env)
        entry = self.env.task_list[3]
        self.event(3, "red")
        self.assertTrue(entry["func"]())
        self.assertFalse(_coerce_failure_result(entry["failure_func"]()))
        self.native_failed = True
        self.assertTrue(_coerce_failure_result(entry["failure_func"]()))
        self.native_failed = False
        self.held = self.cubes[1]
        self.assertTrue(_coerce_failure_result(entry["failure_func"]()))

    def test_previous_subgoal_landings_do_not_complete_the_next(self):
        self.event(1, "red")
        self.assertFalse(self.env.task_list[3]["func"]())
        self.assertFalse(_coerce_failure_result(self.env.task_list[3]["failure_func"]()))


class NativePlacementIdentity(unittest.TestCase):
    def test_both_tasks_bind_the_selected_cube_in_the_full_scene(self):
        from robomme_plus.tasks.reference.video_place_order import VideoPlaceOrder
        from robomme_plus.tasks.reference.video_place_button import VideoPlaceButton

        for cls in (VideoPlaceOrder, VideoPlaceButton):
            with self.subTest(task=cls.__name__):
                env = cls(seed=0, difficulty="hard", obs_mode="state")
                try:
                    index, entry = next((i, t) for i, t in enumerate(env.task_list)
                                        if not t["demonstration"] and "place" in t["name"])
                    wrong = next(c for c in env.all_cubes if c is not env.target_cube)
                    event = dict(task_index=index, cube_name=wrong.name, target_name=env.target_target.name)
                    env._limit_placement_events.append(event)
                    self.assertFalse(entry["func"]())
                    self.assertTrue(_coerce_failure_result(entry["failure_func"]()))
                    event["cube_name"] = env.target_cube.name
                    self.assertTrue(entry["func"]())
                    self.assertFalse(_coerce_failure_result(entry["failure_func"]()))
                finally:
                    env.close()


class PlacementRelease(unittest.TestCase):
    def setUp(self):
        def actor(name, point):
            return SimpleNamespace(name=name, pose=SimpleNamespace(p=np.array([point], float)),
                                   linear_velocity=torch.zeros((1, 3)))
        self.cube = actor('cube', [0, 0, .02])
        self.rings = [actor('correct', [0, 0, .005]), actor('wrong', [.2, 0, .005])]
        self.held, self.opening = True, False
        self.native_failed = False
        robot = SimpleNamespace(get_qpos=lambda: torch.full((1, 9), .04 if self.opening else .015),
                                get_drive_targets=lambda: torch.full((1, 9), .04 if self.opening else 0.))
        self.env = SimpleNamespace(
            all_cubes=[self.cube], targets=self.rings, timestep=1, elapsed_steps=torch.tensor([0]),
            agent=SimpleNamespace(robot=robot, tcp=actor('tcp', [0, 0, .025]),
                                  is_grasping=lambda c: self.held),
            _limit_grasped={}, _limit_pending_release=set(), _limit_placement_events=[],
            _limit_landing_since={}, demonstration_record_traj=False,
            task_list=[dict(name='pick up the cube', segment=self.cube, demonstration=False, func=lambda: False),
                       dict(name='place the cube', segment=self.rings[0], demonstration=False,
                            func=lambda: False, failure_func=lambda: self.native_failed)],
        )
        bind_placement_checks(self.env)
        self.tick()

    def tick(self):
        self.env.elapsed_steps += 1
        placement_events(self.env)

    def test_closed_gripper_contact_flicker_cannot_succeed_or_fail(self):
        self.held, self.native_failed = False, True
        for x in (0, .2, .4):
            self.cube.pose.p[0, 0] = x
            self.env.agent.tcp.pose.p[0, 0] = x
            for _ in range(3):
                self.tick()
            self.assertFalse(sequential_task_check(self.env, self.env.task_list, True)[0])
            self.assertFalse(self.env.current_task_failure)
        self.assertEqual(self.env._limit_placement_events, [])

    def test_open_release_needs_two_settled_control_steps(self):
        self.held, self.opening = False, True
        self.cube.linear_velocity[0, 0] = .3
        for _ in range(3):
            self.tick()
        self.assertEqual(self.env._limit_placement_events, [])
        self.cube.linear_velocity.zero_()
        self.tick()
        placement_events(self.env)  # repeated observation is not a second step
        self.assertEqual(self.env._limit_placement_events, [])
        self.tick()
        self.assertEqual(len(self.env._limit_placement_events), 1)
        self.retreat()
        self.assertTrue(sequential_task_check(self.env, self.env.task_list, True)[0])

    def retreat(self, height=.05):
        self.env.agent.tcp.pose.p[0, 2] += height
        self.tick()

    def land(self):
        self.held, self.opening = False, True
        self.tick()
        self.tick()
        self.assertEqual(len(self.env._limit_placement_events), 1)

    def test_correct_landing_completes_only_after_open_gripper_rises_5cm(self):
        self.land()
        for _ in range(3):
            self.tick()  # hovering at release height
        self.assertFalse(sequential_task_check(self.env, self.env.task_list, True)[0])
        self.retreat(.03)
        self.assertFalse(sequential_task_check(self.env, self.env.task_list, True)[0])
        self.opening = False  # closing the gripper while rising does not count
        self.retreat(.03)
        self.assertFalse(sequential_task_check(self.env, self.env.task_list, True)[0])
        self.opening = True
        self.tick()
        self.assertTrue(sequential_task_check(self.env, self.env.task_list, True)[0])
        self.assertTrue(self.env._limit_placement_events[0]["retreated"])

    def test_wrong_ring_fails_at_landing_without_waiting_for_retreat(self):
        self.cube.pose.p[0, 0] = self.env.agent.tcp.pose.p[0, 0] = .2
        self.land()
        self.assertTrue(sequential_task_check(self.env, self.env.task_list, True)[1])
        self.assertEqual(self.env.current_task_failure_reason if hasattr(self.env, "current_task_failure_reason")
                         else "wrong_placement", "wrong_placement")

    def test_regrasp_before_retreat_supersedes_the_landing(self):
        self.land()
        self.held, self.opening = True, False
        self.tick()
        self.held, self.opening = False, True
        self.retreat()
        self.assertTrue(self.env._limit_placement_events[0]["superseded"])
        self.assertFalse(sequential_task_check(self.env, self.env.task_list, True)[0])

    def test_demonstration_landing_completes_without_retreat(self):
        self.env.demonstration_record_traj = True
        self.env.task_list[1]["demonstration"] = True
        # Not the route-reference landing; demonstration routes are untouched.
        self.env._limit_reference_task_index, self.env.target_cube, self.env.target_target = -1, None, None
        bind_placement_checks(self.env)
        self.land()
        self.assertTrue(self.env._limit_placement_events[0]["retreated"])

    def test_regrasp_cancels_pending_landing(self):
        self.held, self.opening = False, True
        self.tick()
        self.held = True
        self.tick()
        self.held = False
        self.tick()
        self.assertEqual(self.env._limit_placement_events, [])
        self.tick()
        self.assertEqual(len(self.env._limit_placement_events), 1)

    def test_real_slip_away_from_closed_fingers_still_fails_on_bare_table(self):
        self.held = False
        self.cube.pose.p[0, 0] = .4
        self.tick()
        self.tick()
        self.assertTrue(sequential_task_check(self.env, self.env.task_list, True)[2])
        self.assertEqual(self.env._failure_code, 'wrong_placement')


if __name__ == "__main__":
    unittest.main()
