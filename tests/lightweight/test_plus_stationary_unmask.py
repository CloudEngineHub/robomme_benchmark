"""Regression checks for the restored stationary Unmask train/test splits."""
import json
from importlib import resources
import unittest

from robomme_plus.tasks.permanence.utils.tasks import task_config


class StationaryUnmaskTest(unittest.TestCase):
    def test_all_packaged_episodes_use_original_reveal_and_button_rules(self):
        metadata = resources.files('robomme_plus') / 'env_metadata'
        for task in ('ButtonUnmask', 'VideoUnmask'):
            for split, count in (('test', 50), ('train', 3)):
                rows = json.loads((metadata / split / f'record_dataset_{task}_metadata.json').read_text())['records']
                self.assertEqual(len(rows), count)
                self.assertEqual([r['episode'] for r in rows], list(range(count)))
                for row in rows:
                    with self.subTest(task=task, split=split, seed=row['seed']):
                        self.assertFalse(any(k.startswith('remove') for k in row))
                        config = task_config(task, row['difficulty'], row['seed'])
                        self.assertEqual(config['scenario_version'], 3)
                        self.assertEqual(config['required_button_presses'], int(task == 'ButtonUnmask'))
                        self.assertEqual(config['moving_steps'], 0)
                        self.assertEqual(config['extra_central_distractors'],
                                         2 if task == 'ButtonUnmask' and row['difficulty'] == 'hard' else 0)

    def test_original_train_seeds_are_restored_without_test_leakage(self):
        root = resources.files('robomme_plus') / 'env_metadata'
        splits = {s: [r for path in (root / s).iterdir() if path.name.endswith('.json')
                      for r in json.loads(path.read_text())['records']] for s in ('train', 'test')}
        train_seeds = [row['seed'] for row in splits['train']]
        test_seeds = {row['seed'] for row in splits['test']}
        self.assertEqual(len(train_seeds), 50)
        self.assertEqual(len(set(train_seeds)), 50)
        self.assertFalse(set(train_seeds) & test_seeds)
        expected = {'ButtonUnmask': [1567058032, 2043653783, 1878721550],
                    'VideoUnmask': [1656454887, 1751945273, 1497627596]}
        for task, seeds in expected.items():
            self.assertEqual([r['seed'] for r in splits['train'] if r['task'] == task], seeds)

if __name__ == '__main__':
    unittest.main()
