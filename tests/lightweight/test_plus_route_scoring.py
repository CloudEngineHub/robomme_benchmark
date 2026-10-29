"""Adversarial checks for RouteStick's visible-trace and terminal scoring."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from robomme_plus.tasks.imitation.route_stick import _TaskScene, is_obj_swing_onto
from robomme_plus.tasks.imitation.utils.drawing import Drawing


def actor(x, y, z=.07):
    return SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[x, y, z]], dtype=torch.float64)))


def scene():
    return SimpleNamespace(
        current_task_demonstration=False, _gripper_xy_trace=[],
        agent=SimpleNamespace(tcp=actor(0, 0)), _stick_short=False,
        elapsed_steps=torch.tensor([0]), use_demonstrationwrapper=True,
        demonstration_record_traj=False, timestep=0,
        task_list=[dict(name='route', func=lambda: False)], _drawing=Drawing(),
    )


def sample(env, x, y, z=.07, *, advance=True):
    env.agent.tcp.pose.p[0] = torch.tensor([x, y, z], dtype=torch.float64)
    if advance:
        env.elapsed_steps += 1
    return _TaskScene.evaluate(env)


class RouteScoringTest(unittest.TestCase):
    def test_lifted_excursion_cannot_reverse_a_wrong_drawn_curve(self):
        env = scene()
        # A wrong-side curve with a lift/rejoin at its midpoint. The white
        # trace is continuous; spending time in the air must not change its side.
        for f in np.linspace(0, .5, 16):
            sample(env, .16 * f, -.1 * np.sin(np.pi * f))
        for _ in range(100):
            sample(env, .08, .3, z=.2)
        for f in np.linspace(.5, 1, 16):
            sample(env, .16 * f, -.1 * np.sin(np.pi * f))
        self.assertFalse(_TaskScene.direction_fail(env, [actor(.16, 0), actor(0, 0), 'clockwise']))
        self.assertEqual(env._failure_code, 'wrong_route_direction')

    def test_lifted_excursion_cannot_fail_a_correct_drawn_curve(self):
        env = scene()
        for f in np.linspace(0, .5, 16):
            sample(env, .16 * f, .1 * np.sin(np.pi * f))
        for _ in range(100):
            sample(env, .08, -.3, z=.2)
        for f in np.linspace(.5, 1, 16):
            sample(env, .16 * f, .1 * np.sin(np.pi * f))
        self.assertTrue(_TaskScene.direction_fail(env, [actor(.16, 0), actor(0, 0), 'clockwise']))

    def test_repeated_evaluation_does_not_reweight_the_direction_trace(self):
        env = scene()
        sample(env, .04, -.1)
        sample(env, .08, .01)
        for _ in range(100):
            sample(env, .08, .01, advance=False)
        sample(env, .12, -.1)
        self.assertEqual(len(env._gripper_xy_trace), 3)
        self.assertFalse(_TaskScene.direction_fail(env, [actor(.16, 0), actor(0, 0), 'clockwise']))

    def test_target_contact_uses_same_height_gate_as_white_dots(self):
        env = scene()
        target = actor(0, 0)
        for short in (False, True):
            env._stick_short = short
            env._stick_short_tip = actor(0, 0)
            physical = env._stick_short_tip if short else env.agent.tcp
            for z, valid in ((-.001, False), (.07, True), (.081, False), (float('nan'), False)):
                with self.subTest(short=short, z=z):
                    physical.pose.p[0, 2] = z
                    self.assertEqual(bool(is_obj_swing_onto(env, env.agent.tcp, target)), valid)

    def test_latched_failure_cannot_be_reported_as_success(self):
        env = scene()
        sample(env, .08, -.1)
        self.assertFalse(_TaskScene.direction_fail(env, [actor(.16, 0), actor(0, 0), 'clockwise']))
        env.timestep = len(env.task_list)
        flags = _TaskScene.evaluate(env)
        self.assertTrue(flags['fail'].item())
        self.assertFalse(flags['success'].item())


if __name__ == '__main__':
    unittest.main()
