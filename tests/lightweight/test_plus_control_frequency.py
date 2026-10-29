"""Frequency changes must not reinterpret old controls or execution budgets."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robomme_plus.evaluation import demonstration_actions as assets
from robomme_plus.evaluation.timing import episode_control_frequency


class ControlFrequencyTests(unittest.TestCase):

    def test_explicit_frequency_preserves_new_and_historical_time_bases(self):
        historical = {'scene_parameters': {'schema': 'robomme-ood-move-cube/3'}}
        self.assertEqual(episode_control_frequency('MoveCube', historical), 100)
        self.assertEqual(episode_control_frequency('MoveCube', {**historical, 'control_freq': 20}), 20)
        self.assertEqual(episode_control_frequency('MoveCube', {}), 20)
        for task in ('InsertPeg', 'PatternLock', 'RouteStick', 'BinFill'):
            self.assertEqual(episode_control_frequency(task, {}), 20)
        for bad in (True, 20., '20', 0, 25, 500):
            with self.assertRaisesRegex(ValueError, 'control_freq'):
                episode_control_frequency('MoveCube', {'control_freq': bad})

    def test_versioned_assets_are_bound_to_their_control_frequency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = {'seed': 12}
            digest = assets.scene_digest(scene)
            commands = np.arange(45, dtype=float).reshape(3, 15)
            resource = 'assets/demonstrations/MoveCube/test/20hz/12.npz'
            path = root / resource
            path.parent.mkdir(parents=True)
            record = dict(seed=12, scene_parameters=scene, control_freq=20,
                          demonstration_actions=dict(resource=resource, scene_sha256=digest, control_freq=20))
            def write(frequency):
                payload = io.BytesIO()
                np.savez_compressed(payload, actions=commands, scene_sha256=digest, control_freq=frequency)
                path.write_bytes(payload.getvalue())
                record['demonstration_actions']['sha256'] = hashlib.sha256(payload.getvalue()).hexdigest()
            with patch.object(assets.resources, 'files', return_value=root):
                write(20)
                np.testing.assert_array_equal(assets.load_demonstration_actions(record), commands)
                record['control_freq'] = 100
                with self.assertRaisesRegex(ValueError, 'frequency'):
                    assets.load_demonstration_actions(record)
                record['control_freq'] = 20
                write(100)
                with self.assertRaisesRegex(ValueError, 'frequency'):
                    assets.load_demonstration_actions(record)


if __name__ == '__main__':
    unittest.main()
