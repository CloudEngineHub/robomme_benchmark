"""The public evaluator stays usable without authoring or execution policies."""

import ast
from pathlib import Path
import unittest

from robomme_plus import BenchmarkEnvBuilder
from robomme_plus.evaluation.clock import ExecutionClock
from robomme_plus.evaluation.grounding import fill_grounding
from robomme_plus.evaluation.runtime import ACTION_SPACES, make_episode

import numpy as np


class EvaluationContractTest(unittest.TestCase):
    def test_highlight_metadata_preserves_old_cues_and_resolves_new_modes(self):
        from robomme_plus.evaluation.metadata import normalize_episode
        from robomme_plus.tasks.reference.utils.highlight_timing import resolve_highlight_mode

        record = dict(seed=17, difficulty="hard", oracle_policy_step=1000)
        self.assertEqual(normalize_episode("PickHighlight", record, 0)["highlight_mode"], "staggered")
        self.assertNotIn("highlight_mode", record)
        self.assertEqual(normalize_episode("PickHighlight", record, 0)["layout_version"], 1)
        self.assertFalse(normalize_episode("PickHighlight", record, 0)["local_adjustments"])
        self.assertTrue(normalize_episode("PickHighlight", {**record, "local_adjustments": True}, 0)["local_adjustments"])
        with self.assertRaisesRegex(ValueError, "local_adjustments"):
            normalize_episode("PickHighlight", {**record, "local_adjustments": "true"}, 0)
        for mode in ("simultaneous", "staggered", "random"):
            normalized = normalize_episode("PickHighlight", {**record, "highlight_mode": mode}, 0)
            self.assertEqual(normalized["highlight_mode"], resolve_highlight_mode(17, mode))
            self.assertEqual(normalize_episode("PickHighlight", normalized, 0), normalized)
        with self.assertRaisesRegex(ValueError, "highlight_mode"):
            normalize_episode("PickHighlight", {**record, "highlight_mode": "unknown"}, 0)
        for version in (2, 3, 4):
            updated = normalize_episode("PickHighlight", {**record, "layout_version": version}, 0)
            self.assertEqual(updated["layout_version"], version)
        for version in (0, 5, True, "2"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "layout_version"):
                normalize_episode("PickHighlight", {**record, "layout_version": version}, 0)

    def test_every_registered_task_provides_its_goal(self):
        import re
        from importlib import import_module
        from robomme_plus import register_envs, TASKS

        register_envs()
        for name, spec in TASKS.items():
            module = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            task_class = getattr(import_module(f"robomme_plus.tasks.{spec.suite}.{module}"), name)
            self.assertTrue(callable(getattr(task_class, 'language_goal', None)), name)

    def test_public_metadata_and_budgets_for_every_action_mode(self):
        for task in BenchmarkEnvBuilder.get_task_list():
            for mode in ACTION_SPACES:
                builder = BenchmarkEnvBuilder(task, action_space=mode)
                self.assertGreater(builder.get_episode_num(), 0)
                record = builder.get_episode_metadata(0)
                expected = (3 * record['oracle_policy_step'] + 1) // 2 + 200
                self.assertEqual(builder.episode_max_steps(0), expected)
                record['seed'] = -1
                self.assertGreaterEqual(builder.get_episode_metadata(0)['seed'], 0)

    def test_public_runtime_requires_supplied_metadata(self):
        for mode in ACTION_SPACES:
            with self.assertRaisesRegex(ValueError, 'supplied episode metadata'):
                make_episode('BinFill', seed=0, difficulty='hard', action_space=mode)

    def test_public_runtime_rejects_conflicting_episode_identity_before_building(self):
        from unittest.mock import patch

        for task in ('BinFill', 'PickHighlight', 'MoveCube'):
            row = BenchmarkEnvBuilder(task).get_episode_metadata(0)
            arguments = {key: row.get(key) for key in ('seed', 'difficulty', 'variant', 'native_difficulty')}
            with patch('gymnasium.make') as make:
                for field, changed in (('seed', row['seed'] + 1), ('difficulty', 'incorrect')):
                    with self.subTest(task=task, field=field), self.assertRaisesRegex(ValueError, field):
                        make_episode(task, **{**arguments, field: changed}, episode_metadata=row)
                make.assert_not_called()

    def test_retired_removal_metadata_cannot_silently_build_stationary_tasks(self):
        from robomme_plus.evaluation.metadata import normalize_episode

        for task in ('ButtonUnmask', 'VideoUnmask'):
            row = BenchmarkEnvBuilder(task).get_episode_metadata(0)
            for changes in ({'remove': True}, {'remove_version': 2}, {'remove_steps': 200}):
                with self.subTest(task=task, changes=changes), self.assertRaisesRegex(ValueError, 'retired'):
                    normalize_episode(task, {**row, **changes}, 0)
            self.assertEqual(row, normalize_episode(task, {**row, 'remove': False}, 0))

    def test_replay_identity_uses_runtime_settings_and_preserves_input(self):
        from robomme_plus.evaluation.metadata import replay_identity

        for task in BenchmarkEnvBuilder.get_task_list():
            row = BenchmarkEnvBuilder(task).get_episode_metadata(0)
            original = dict(row)
            self.assertEqual(replay_identity(task, row), replay_identity(task, {
                **row, 'oracle_wall_seconds': 999, 'max_steps': 0, 'original_seed': -1,
                'review_video': '/different/export/path',
            }))
            self.assertNotEqual(replay_identity(task, row),
                                replay_identity(task, {**row, 'oracle_policy_step': row['oracle_policy_step'] + 1}))
            self.assertEqual(row, original)

    def test_replay_identity_rejects_non_object_metadata(self):
        from robomme_plus.evaluation.metadata import replay_identity

        for record in (None, [], "invalid"):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, 'JSON object'):
                replay_identity('BinFill', record)

    def test_packaged_metadata_has_no_local_authoring_paths(self):
        import json
        from importlib import resources

        def check(value):
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            elif isinstance(value, str):
                self.assertFalse(value.startswith(('/home/', '/data/', '/nfs/', '/tmp/')), value)

        root = resources.files('robomme_plus') / 'env_metadata'
        for split in ('train', 'test'):
            for path in (root / split).iterdir():
                if path.name.endswith('.json'):
                    with self.subTest(split=split, file=path.name):
                        check(json.loads(path.read_text()))

    def test_clock_excludes_demonstration_and_expires_at_the_limit(self):
        clock = ExecutionClock({'max_steps': 3, 'oracle_policy_step': 1})
        for _ in range(20):
            self.assertFalse(clock.advance())
        self.assertEqual(clock.information()['execution_control_steps'], 0)
        clock.start()
        self.assertFalse(clock.advance())
        self.assertFalse(clock.advance())
        self.assertTrue(clock.advance())
        self.assertEqual(clock.information()['remaining_steps'], 0)
        clock.reset()
        self.assertFalse(clock.active)
        self.assertEqual(clock.steps, 0)

    def test_oracle_subgoal_grounding_tracks_visibility(self):
        from types import SimpleNamespace
        from robomme_plus.evaluation.demonstration import DemonstrationWrapper

        actor = object()
        segmentation = np.zeros((16, 16), dtype=int)
        native = SimpleNamespace(
            current_subgoal_segment='pick at <>', current_task_name='pick the object',
            current_segment=actor, segmentation_id_map={7: actor},
        )
        wrapper = SimpleNamespace(unwrapped=native)
        obs = {'sensor_data': {'base_camera': {'segmentation': segmentation}}}

        def current_grounding():
            return DemonstrationWrapper._compute_segmentation_and_fill_subgoal(wrapper, obs)[0]

        segmentation[2:6, 4:8] = 7
        self.assertEqual(
            fill_grounding('pick at <>', 'pick the object', [actor], segmentation, {7: actor}),
            'pick at <3, 5>',
        )
        self.assertEqual(current_grounding(), 'pick at <3, 5>')
        # The same subgoal and actor must get fresh coordinates on the next frame.
        segmentation.fill(0)
        segmentation[8:12, 10:14] = 7
        self.assertEqual(current_grounding(), 'pick at <9, 11>')
        segmentation.fill(0)
        self.assertEqual(
            fill_grounding('pick at <>', 'pick the object', [actor], segmentation, {7: actor}),
            'pick the object',
        )
        self.assertEqual(current_grounding(), 'pick the object')
        # Visibility returning must also restore current, rather than stale, coordinates.
        segmentation[1:5, 2:6] = 7
        self.assertEqual(current_grounding(), 'pick at <2, 3>')

    def test_public_source_has_no_private_package_imports(self):
        root = Path(__file__).resolve().parents[2] / 'src' / 'robomme_plus'
        for path in root.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or '']
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    self.assertNotEqual(module.split('.')[0], 'robomme_plus_private', str(path))


if __name__ == '__main__':
    unittest.main()
