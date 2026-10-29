"""VideoRepick goal identity must not influence its already generated layout.

Requires the simulator; run with CUDA_VISIBLE_DEVICES set to an available GPU.
"""
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from robomme_plus import register_envs
from robomme_plus.registry import TASKS
from robomme_plus.tasks.reference.utils import repick_tasks


class RepickTargetLayoutTest(unittest.TestCase):
    def test_changing_goal_preserves_layout_and_actor_order_on_every_level(self):
        register_envs('VideoRepick')
        for level in ('easy', 'medium', 'hard'):
            layouts = []
            for goal in (0, -1):
                def choose(actors, seed):
                    xy = np.array([
                        actor.pose.p.cpu().numpy().reshape(3)[:2]
                        for actor in actors
                    ])
                    self.assertTrue(np.all(np.abs(xy) < .5),
                                    'Goal selected before tabletop placement')
                    layouts.append(xy.copy())
                    return actors[goal]

                with self.subTest(level=level, goal=goal), patch.object(
                    repick_tasks, 'choose_target', choose
                ):
                    env = gym.make(
                        TASKS['VideoRepick'].gym_id, seed=42, difficulty=level,
                        obs_mode='none', control_mode='pd_joint_pos',
                        reward_mode='none',
                    )
                    try:
                        native = env.unwrapped
                        actors = native._repick_level_actors
                        self.assertIs(native.target_cube_1, actors[goal])
                        self.assertEqual([a.name for a in actors[:4]],
                                         ['bin_0', 'bin_1', 'bin_2', 'bin_3'])
                    finally:
                        env.close()
            np.testing.assert_array_equal(layouts[0], layouts[1])


if __name__ == '__main__':
    unittest.main()
