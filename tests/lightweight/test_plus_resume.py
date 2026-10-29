"""Exercise saved runs with tiny simulated environments; no GPU/API/model calls."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/plus_eval'))
from evaluation import Evaluation, PolicyStop, PolicyUnavailable, summarize, write_json
import run_storage as storage


class Video:
    def __init__(self, path, **kwargs):
        Path(path).write_bytes(b'video fixture')
    def add(self, *args, **kwargs):
        pass
    def close(self):
        pass


class Builder:
    outcomes = {}
    started = []
    @staticmethod
    def get_task_list():
        return ['BinFill', 'StopCube']
    def __init__(self, task, **kwargs):
        self.task = task
    def get_episode_num(self):
        return 3
    def episode_max_steps(self, episode):
        return 3
    def get_episode_metadata(self, episode):
        return {'task': self.task, 'episode': episode, 'seed': 100 + episode,
                'difficulty': 'hard', 'oracle_policy_step': 1}
    def make_env_for_episode(self, episode, **kwargs):
        Builder.started.append((self.task, episode))
        outcome = self.outcomes.get((self.task, episode), 'success')
        frame = np.zeros((4, 4, 3), np.uint8)
        obs = {'front_rgb_list': [frame], 'wrist_rgb_list': [frame]}
        def reset():
            if outcome == 'error': raise RuntimeError('fixture setup error')
            if outcome == 'interrupted': raise KeyboardInterrupt()
            if outcome == 'fatal': raise PolicyUnavailable('fixture policy service failed')
            return obs, {'task_goal': ['goal']}
        return SimpleNamespace(reset=reset, close=lambda: None,
            step=lambda _: (obs, 0, True, False, {'status': outcome, 'execution_control_steps': 1}))


class Policy(Evaluation):
    method = 'fixture'
    setup_calls = 0
    def setup(self):
        Policy.setup_calls += 1
    def act(self, obs, info):
        return np.zeros(8)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'run'
        Builder.outcomes = {}
        Builder.started = []
        Policy.setup_calls = 0
        self.patch_builder = patch('robomme_plus.env_record_wrapper.BenchmarkEnvBuilder', Builder)
        self.patch_builder.start()
        self.addCleanup(self.patch_builder.stop)
        self.patch_video = patch('evaluation.VideoRecorder', Video)
        self.patch_video.start()
        self.addCleanup(self.patch_video.stop)

    def evaluate(self, **kwargs):
        options = dict(tasks=['BinFill'], episodes=None, output_dir=self.directory,
                       api_pricing_json=None, fail_fast=False, resume=False,
                       retry_errors=False, rerun=False, model='fixture-model')
        options.update(kwargs)
        return asyncio.run(Policy(SimpleNamespace(**options)).run())

    def summary(self):
        return storage.read(self.directory / 'summary.json')

    def test_resume_retries_error_keeps_fail_and_success_and_does_not_reload_when_done(self):
        Builder.outcomes = {('BinFill', 0): 'fail', ('BinFill', 1): 'error'}
        self.evaluate()
        first = (self.directory / 'BinFill_ep0/result.json').read_bytes()
        Builder.outcomes = {}
        Builder.started = []
        self.evaluate(resume=True, tasks=None)
        self.assertEqual(Builder.started, [('BinFill', 1)])
        self.assertEqual(first, (self.directory / 'BinFill_ep0/result.json').read_bytes())
        self.assertEqual(self.summary()['counts']['success'], 2)
        self.assertEqual(self.summary()['completed'], 3)
        self.assertEqual(len(list((self.directory / '.history').glob('*/BinFill_ep1'))), 1)
        self.assertEqual(Policy.setup_calls, 2)
        self.evaluate(resume=True, tasks=None)
        self.assertEqual(Policy.setup_calls, 2)

    def test_interruption_is_pending_and_resume_runs_only_unfinished(self):
        Builder.outcomes = {('BinFill', 1): 'interrupted'}
        with self.assertRaises(KeyboardInterrupt): self.evaluate()
        self.assertEqual((self.summary()['completed'], self.summary()['pending'], self.summary()['interrupted']), (1, 2, 1))
        self.assertEqual(self.summary()['counts']['error'], 0)
        Builder.outcomes = {}
        Builder.started = []
        self.evaluate(resume=True, tasks=None)
        self.assertEqual(Builder.started, [('BinFill', 1), ('BinFill', 2)])
        self.assertTrue(self.summary()['complete'])

    def test_subset_rerun_replaces_whole_rollout_and_preserves_other_episodes(self):
        self.evaluate()
        before = (self.directory / 'BinFill_ep0/result.json').read_bytes()
        (self.directory / 'BinFill_ep1/obsolete.txt').write_text('previous attempt')
        Builder.outcomes = {('BinFill', 1): 'fail'}
        Builder.started = []
        self.evaluate(rerun=True, tasks=None, episodes=[1])
        self.assertEqual(Builder.started, [('BinFill', 1)])
        self.assertFalse((self.directory / 'BinFill_ep1/obsolete.txt').exists())
        self.assertEqual(before, (self.directory / 'BinFill_ep0/result.json').read_bytes())
        self.assertEqual(self.summary()['completed'], 3)
        self.assertEqual(self.summary()['counts']['fail'], 1)
        self.assertEqual(len(list((self.directory / '.history').glob('*/BinFill_ep1/obsolete.txt'))), 1)

    def test_interrupted_rerun_keeps_original_but_resume_remembers_replacement_intent(self):
        self.evaluate()
        old = (self.directory / 'BinFill_ep1/result.json').read_bytes()
        Builder.outcomes = {('BinFill', 1): 'interrupted'}
        with self.assertRaises(KeyboardInterrupt): self.evaluate(rerun=True)
        self.assertEqual(old, (self.directory / 'BinFill_ep1/result.json').read_bytes())
        self.assertEqual(self.summary()['pending_replacements'], [{'task': 'BinFill', 'episode': 1}, {'task': 'BinFill', 'episode': 2}])
        Builder.outcomes = {}
        Builder.started = []
        self.evaluate(resume=True, tasks=None)
        self.assertEqual(Builder.started, [('BinFill', 1), ('BinFill', 2)])
        self.assertEqual(self.summary()['pending_replacements'], [])

    def test_missing_artifact_is_retried_and_only_errors_does_not_run_missing(self):
        Builder.outcomes = {('BinFill', 0): 'error'}
        self.evaluate(episodes=[0, 1])
        Builder.outcomes = {}
        Builder.started = []
        self.evaluate(retry_errors=True, episodes=[0, 1, 2])
        self.assertEqual(Builder.started, [('BinFill', 0)])
        (self.directory / 'BinFill_ep1/episode.mp4').unlink()
        Builder.started = []
        self.evaluate(resume=True, tasks=None)
        self.assertEqual(Builder.started, [('BinFill', 1), ('BinFill', 2)])

    def test_zero_action_policy_stop_is_a_valid_failure_not_an_unfinished_episode(self):
        with patch.object(Policy, 'act', side_effect=PolicyStop('model_answer:done')):
            self.evaluate()
        self.assertEqual(self.summary()['counts']['fail'], 3)
        Builder.started = []
        self.evaluate(resume=True)
        self.assertEqual(Builder.started, [])

    def test_task_selection_can_extend_existing_run_without_losing_old_cases(self):
        self.evaluate(episodes=[0])
        self.evaluate(resume=True, tasks=['StopCube'], episodes=[2])
        self.assertEqual({(r['task'], r['episode']) for r in self.summary()['results']}, {('BinFill', 0), ('StopCube', 2)})

    def test_changed_model_and_concurrent_writer_are_rejected(self):
        self.evaluate(episodes=[0])
        with self.assertRaisesRegex(ValueError, 'Inference setting changed'):
            self.evaluate(resume=True, model='other-model')
        with storage.run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, 'Another worker'):
                self.evaluate(resume=True)

    def test_resume_replaces_changed_seed_and_does_not_reuse_missing_metadata(self):
        self.evaluate()
        path = self.directory / 'BinFill_ep0/metadata.json'
        saved = storage.read(path)
        saved['seed'] += 1
        write_json(path, saved)
        (self.directory / 'BinFill_ep1/metadata.json').unlink()
        Builder.started = []
        self.evaluate(resume=True)
        self.assertEqual(Builder.started, [('BinFill', 0), ('BinFill', 1)])

    def test_resume_ignores_recording_times_but_rejects_changed_budget(self):
        self.evaluate()
        for episode, changes in ((0, {'oracle_wall_seconds': 123.45}),
                                 (1, {'oracle_policy_step': 2})):
            path = self.directory / f'BinFill_ep{episode}/metadata.json'
            write_json(path, {**storage.read(path), **changes})
        Builder.started = []
        self.evaluate(resume=True)
        self.assertEqual(Builder.started, [('BinFill', 1)])

    def test_changed_benchmark_requires_explicit_rerun_and_keeps_episode_provenance(self):
        self.evaluate()
        path = self.directory / 'config.json'
        config = storage.read(path)
        retained = dict(config['case_provenance'])
        config['benchmark_source_sha256']['src/robomme_plus/registry.py'] = 'old source'
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, 'Benchmark source changed'):
            self.evaluate(resume=True)
        self.assertEqual(storage.read(path), config)
        self.evaluate(rerun=True, episodes=[1])
        current = storage.read(path)
        self.assertNotEqual(current['case_provenance']['BinFill_ep0'],
                            current['case_provenance']['BinFill_ep1'])
        for episode in (0, 2):
            self.assertEqual(current['case_provenance'][f'BinFill_ep{episode}'],
                             retained[f'BinFill_ep{episode}'])
        replacement = storage.read(self.directory / current['case_provenance']['BinFill_ep1'])
        self.assertEqual(replacement['benchmark_source_sha256'], current['benchmark_source_sha256'])
        archived = storage.read(self.directory / '.attempts' / current['attempt_id'] / 'previous_config.json')
        self.assertEqual(archived, config)

    def test_resume_checks_retained_episode_source_after_a_partial_rerun(self):
        self.evaluate()
        config = storage.read(self.directory / 'config.json')
        snapshot = self.directory / config['case_provenance']['BinFill_ep0']
        old = storage.read(snapshot)
        old['benchmark_source_sha256']['src/robomme_plus/registry.py'] = 'previous revision'
        write_json(snapshot, old)
        self.evaluate(rerun=True, episodes=[1])
        Builder.started = []
        self.evaluate(resume=True)
        self.assertEqual(Builder.started, [('BinFill', 0), ('BinFill', 2)])

    def test_private_checkpoint_identity_is_compared_without_saving_personal_path(self):
        self.evaluate(episodes=[0], checkpoint=Path('/home/private-person/checkpoints/model'))
        for path in self.directory.rglob('*.json'):
            self.assertNotIn('private-person', path.read_text())
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.evaluate(resume=True, checkpoint=Path('/home/private-person/checkpoints/other-model'))

    def test_fatal_policy_error_stops_before_cascading_to_other_episodes(self):
        Builder.outcomes = {('BinFill', 0): 'fatal'}
        with self.assertRaisesRegex(RuntimeError, 'Stopped after evaluation error'):
            self.evaluate()
        self.assertEqual(Builder.started, [('BinFill', 0)])
        self.assertEqual(self.summary()['pending'], 2)
        self.assertTrue(self.summary()['results'][0]['fatal_error'])

    def test_replacement_journal_recovers_gap_and_rebuild_ignores_archives(self):
        self.evaluate(episodes=[0])
        target = self.directory / 'BinFill_ep0'
        archive = self.directory / '.history/crash/BinFill_ep0'
        archive.parent.mkdir(parents=True)
        target.rename(archive)
        write_json(self.directory / '.replacement.json', {'target': 'BinFill_ep0',
            'source': '.attempts/crash/BinFill_ep0', 'archive': '.history/crash/BinFill_ep0'})
        storage.recover_replacement(self.directory)
        self.assertTrue(target.exists())
        (self.directory / 'summary.json').write_text('{}')
        self.assertEqual(storage.rebuild_task(self.directory)['completed'], 1)

    def test_published_episode_is_not_retried_if_process_died_before_progress_write(self):
        self.evaluate(episodes=[0])
        result = storage.saved_result(self.directory, {'task': 'BinFill', 'episode': 0})
        write_json(self.directory / 'run_state.json', {'attempt_id': result['attempt_id'],
            'pending': [{'task': 'BinFill', 'episode': 0}], 'status': 'running'})
        Builder.started = []
        self.evaluate(resume=True, tasks=None)
        self.assertEqual(Builder.started, [])

    def test_summary_deduplicates_episode_keys(self):
        case = {'task': 'BinFill', 'episode': 0}
        report = summarize([case], [{**case, 'status': 'fail', 'success': False}, {**case, 'status': 'success', 'success': True}])
        self.assertEqual((report['completed'], report['counts']['success']), (1, 1))






if __name__ == '__main__':
    unittest.main()
