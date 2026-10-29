"""Demonstration collection retains visible frames and hides reset motions."""

from types import SimpleNamespace
import unittest

import numpy as np

from robomme_plus.evaluation import batches
from robomme_plus.evaluation.demonstration import DemonstrationWrapper


class DemonstrationFramesTest(unittest.TestCase):
    def test_button_view_handoff_returns_home_at_rest_with_full_budget(self):
        from robomme_plus import BenchmarkEnvBuilder
        from robomme_plus.control.reset import get_reset_panda_param

        env = BenchmarkEnvBuilder('ButtonUnmask', action_space='joint_angle').make_env_for_episode(0)
        try:
            _, info = env.reset()
            robot = env.unwrapped.agent.robot
            np.testing.assert_allclose(robot.get_qpos().cpu().numpy()[0],
                                       get_reset_panda_param('qpos'), atol=1e-3)
            self.assertLess(float(robot.get_qvel().abs().max()), .2)
            self.assertEqual(info['execution_control_steps'], 0)
            self.assertEqual(info['remaining_steps'], info['max_steps'])
        finally:
            env.close()

    def test_recording_marker_survives_internal_step_collection(self):
        class Wrapper:
            step = DemonstrationWrapper.step
            execution_clock = SimpleNamespace(active=False)

            def _step_batch(self, record):
                return batches.to_step_batch([(
                    {'front_rgb_list': np.full((2, 2, 3), int(record), dtype=np.uint8)},
                    0.0, False, False, {'_record_frame': record, 'status': 'ongoing'},
                )])

        wrapper = Wrapper()
        planner = SimpleNamespace(env=wrapper)

        def run_demo():
            for record in (True, True, False, False, True):
                wrapper.step(record)

        collected = batches.to_step_batch(batches._collect_dense_steps(planner, run_demo))
        visible = DemonstrationWrapper._filter_no_record_from_step_batch(wrapper, collected)
        self.assertEqual(len(visible[0]['front_rgb_list']), 3)
        self.assertEqual(visible[4]['_record_frame'], [True, True, True])
        wrapper.execution_clock.active = True
        self.assertNotIn('_record_frame', wrapper.step(True)[4])


if __name__ == '__main__':
    unittest.main()
