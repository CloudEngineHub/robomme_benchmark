"""Static counting episodes restart after partial, completed and failed runs."""

import unittest

import numpy as np
import sapien
import torch

from robomme_plus.tasks.counting.pick_xtimes import PickXtimes
from robomme_plus.tasks.counting.swing_xtimes import SwingXtimes
from robomme_plus.tasks.counting.bin_fill import BinFill


class CountingReset(unittest.TestCase):
    def test_easy_tasks_clear_sequence_failures_timers_and_restore_cubes(self):
        for cls in (PickXtimes, SwingXtimes, BinFill):
            with self.subTest(task=cls.__name__):
                env = cls(seed=0, difficulty="easy", obs_mode="state")
                try:
                    for state in ("partial", "complete", "failed"):
                        with self.subTest(state=state):
                            env.timestep = len(env.task_list) if state == "complete" else 2
                            env.current_task_index = env.timestep
                            env._last_task_index = env.timestep
                            env._timelimit_deadlines = {env.timestep: 10}
                            env._timewindow_timers = {"old": 99}
                            env.first_timestep = 99
                            env.current_task_failure = state == "failed"
                            env.failureflag = torch.tensor([state == "failed"])
                            env.successflag = torch.tensor([state == "complete"])
                            env._failure_code = "wrong_cube" if state == "failed" else None
                            cube = env.all_cubes[0]
                            env.currentpickup = cube
                            cube.set_pose(sapien.Pose([10, 10, 0]))
                            cube.set_linear_velocity([1, 0, 0])
                            if cls is PickXtimes:
                                env._static_placement.count = env.num_repeats
                                env._static_placement.released_at = 10
                                env._static_placement.release_gripper_z_m = .02
                            if cls is SwingXtimes:
                                env.swing_count = env.max_swings
                                env.swing_over_limit = True
                            _, info = env.reset()
                            self.assertEqual(env.timestep, 0)
                            self.assertEqual(env.current_task_index, 0)
                            self.assertFalse(info["success"].item())
                            self.assertFalse(info["fail"].item())
                            self.assertFalse(env.current_task_failure)
                            self.assertIsNone(env.currentpickup)
                            self.assertIsNone(env._failure_code)
                            self.assertEqual(env._timelimit_deadlines, {})
                            self.assertEqual(env._timewindow_timers, {})
                            self.assertFalse(hasattr(env, "first_timestep"))
                            np.testing.assert_allclose(cube.pose.p.cpu(), cube.initial_pose.p.cpu(), atol=1e-6)
                            np.testing.assert_allclose(cube.linear_velocity.cpu(), 0, atol=1e-6)
                            if cls is PickXtimes:
                                self.assertEqual(env._static_placement.count, 0)
                                self.assertIsNone(env._static_placement.released_at)
                                self.assertIsNone(env._static_placement.release_gripper_z_m)
                                self.assertEqual(env._static_placement_events, [])
                            if cls is SwingXtimes:
                                self.assertEqual(env.swing_count, 0)
                                self.assertFalse(env.swing_over_limit)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
