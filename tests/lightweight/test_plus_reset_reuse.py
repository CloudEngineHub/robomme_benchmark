"""Public resets restore the episode instead of inheriting prior progress."""

import unittest

import numpy as np
import sapien

from robomme_plus.common.sim import position


class ResetReuse(unittest.TestCase):
    def test_reference_tasks_restore_cubes_and_clear_previous_episode_timers(self):
        from robomme_plus.tasks.reference.pick_highlight import PickHighlight
        from robomme_plus.tasks.reference.video_place_button import VideoPlaceButton
        from robomme_plus.tasks.reference.video_place_order import VideoPlaceOrder
        from robomme_plus.tasks.reference.video_repick import VideoRepick

        for cls in (PickHighlight, VideoPlaceButton, VideoPlaceOrder, VideoRepick):
            with self.subTest(task=cls.__name__):
                env = cls(seed=0, difficulty='hard', obs_mode='state')
                try:
                    env.reset(seed=0)
                    cubes = list(env._repick_level_actors if cls is VideoRepick else env.all_cubes)
                    expected = [position(c, False).copy() for c in cubes]
                    for _ in range(2):
                        env.timestep = len(env.task_list)
                        env._last_task_index = env.timestep
                        env._timelimit_deadlines = {999: 1}
                        env._timewindow_timers = {'old': -100}
                        env.first_timestep = -100
                        cubes[0].set_pose(sapien.Pose([0, .25, .1]))
                        cubes[0].set_linear_velocity([1, 0, 0])
                        _, info = env.reset(seed=0)
                        self.assertLess(env.timestep, len(env.task_list))
                        self.assertFalse(bool(info['success']) or bool(info['fail']))
                        self.assertNotIn(999, env._timelimit_deadlines)
                        self.assertNotIn('old', env._timewindow_timers)
                        self.assertNotEqual(getattr(env, 'first_timestep', None), -100)
                        np.testing.assert_allclose([position(c, False) for c in cubes], expected, atol=1e-6)
                        np.testing.assert_allclose(cubes[0].linear_velocity.cpu(), 0, atol=1e-6)
                finally:
                    env.close()

    def test_insert_reuses_first_public_reset_scene_including_entry_side(self):
        from robomme_plus.tasks.imitation.insert_peg import InsertPeg

        env = InsertPeg(seed=0, difficulty='hard', obs_mode='state')
        try:
            env.reset(seed=0)
            expected = [position(p, False).copy() for p in env.pegs]
            side, end = env.direction, env.obj_flag
            for reconfigure in (False, False, True):
                env.reset(seed=0, options={'reconfigure': reconfigure})
                self.assertEqual((env.direction, env.obj_flag), (side, end))
                np.testing.assert_allclose([position(p, False) for p in env.pegs], expected, atol=1e-6)
        finally:
            env.close()

    def test_reconfigure_preserves_layout_answer_and_stop_disc_scale(self):
        from robomme_plus.tasks.counting.bin_fill import BinFill
        from robomme_plus.tasks.counting.stop_cube import StopCube
        from robomme_plus.tasks.reference.video_place_order import VideoPlaceOrder

        for cls, level in ((BinFill, 'easy'), (StopCube, 'hard'), (VideoPlaceOrder, 'hard')):
            with self.subTest(task=cls.__name__):
                env = cls(seed=0, difficulty=level, obs_mode='state')
                try:
                    env.reset(seed=0)
                    expected = [position(c, False).copy() for c in env.all_cubes] if cls is not StopCube else None
                    radius = env._sampling['positions'].get('target', {}).get('radius_factor')
                    answer = (env.target_target.name, env.which_in_subset) if cls is VideoPlaceOrder else None
                    for _ in range(2):
                        env.reset(seed=0, options={'reconfigure': True})
                        if expected is not None:
                            np.testing.assert_allclose([position(c, False) for c in env.all_cubes], expected, atol=1e-6)
                        if cls is StopCube:
                            self.assertEqual(env._sampling['positions']['target']['radius_factor'], radius)
                        if cls is VideoPlaceOrder:
                            self.assertEqual((env.target_target.name, env.which_in_subset), answer)
                finally:
                    env.close()
