"""Recorded demos are scene-bound and cannot replace policy execution."""
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robomme_plus.evaluation import demonstration_actions as assets


class DemonstrationActionsTests(unittest.TestCase):
    def archive(self, root, actions):
        scene = {'seed': 12, 'layouts': {'demonstration': 'frozen'}}
        digest = assets.scene_digest(scene)
        payload = io.BytesIO()
        np.savez_compressed(payload, actions=actions, scene_sha256=digest)
        resource = 'assets/demonstrations/MoveCube/test/12.npz'
        path = root / resource
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload.getvalue())
        return dict(seed=12, scene_parameters=scene, demonstration_actions=dict(
            resource=resource, scene_sha256=digest,
            sha256=hashlib.sha256(payload.getvalue()).hexdigest()))

    def test_loads_exact_controls_and_rejects_changed_scene_or_archive(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            commands = np.arange(45, dtype=float).reshape(3, 15)
            row = self.archive(root, commands)
            with patch.object(assets.resources, 'files', return_value=root):
                np.testing.assert_array_equal(assets.load_demonstration_actions(row), commands)
                row['scene_parameters']['seed'] = 13
                with self.assertRaisesRegex(ValueError, 'different scene'):
                    assets.load_demonstration_actions(row)
                row['scene_parameters']['seed'] = 12
                (root / row['demonstration_actions']['resource']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    assets.load_demonstration_actions(row)

    def test_rejects_wrong_width_empty_or_nonfinite_commands(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for actions in (np.zeros((2, 8)), np.zeros((0, 15)), np.full((2, 15), np.nan)):
                row = self.archive(root, actions)
                with patch.object(assets.resources, 'files', return_value=root):
                    with self.assertRaisesRegex(ValueError, '15D'):
                        assets.load_demonstration_actions(row)

    def test_resource_is_limited_to_the_selected_seed_and_split(self):
        with TemporaryDirectory() as folder:
            row = self.archive(Path(folder), np.zeros((1, 15)))
            for resource in ('/tmp/12.npz', 'assets/demonstrations/MoveCube/test/13.npz',
                             'assets/demonstrations/MoveCube/../12.npz'):
                row['demonstration_actions']['resource'] = resource
                with self.assertRaisesRegex(ValueError, 'resource'):
                    assets.load_demonstration_actions(row)
        self.assertIsNone(assets.load_demonstration_actions({}))


    def test_recorded_actions_apply_only_to_demonstration_and_still_require_success(self):
        from robomme_plus.tasks.imitation.move_cube import MoveCube
        from robomme_plus.tasks.imitation.move_cube_support import oracle, diversity
        commands = np.arange(30).reshape(2, 15)
        native = SimpleNamespace(_phase='demonstration', _demonstration_actions=commands,
            scene_parameters={'oracle_parameters': {'execution': {'chosen': True}}},
            result=Mock(return_value={'success': True}), seed=12)
        env, planner = SimpleNamespace(step=Mock()), object()
        with patch.object(diversity, 'validate_reference_loading_surface'), patch.object(oracle, 'solve') as solve:
            MoveCube._solve_reference_once(native, env, planner)
            np.testing.assert_array_equal([c.args[0] for c in env.step.call_args_list], commands)
            solve.assert_not_called()
            env.step.reset_mock()
            native._phase = 'execution'
            MoveCube._solve_reference_once(native, env, planner)
            env.step.assert_not_called()
            solve.assert_called_once_with(env, planner, {'chosen': True})
            native._phase = 'demonstration'
            native.result.return_value = {'success': False, 'criteria': {'missing': 'progress'}}
            with self.assertRaisesRegex(RuntimeError, 'recorded demonstration failed'):
                MoveCube._solve_reference_once(native, env, planner)


if __name__ == '__main__':
    unittest.main()
