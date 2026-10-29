"""Unified artifacts, accounting, stopping and sequential failure recovery."""
import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/plus_eval"))
from evaluation import Evaluation, EpisodeRecorder, VideoRecorder, PolicyStop, api_usage, estimate_cost, summarize, compare_runs


class ProtocolTest(unittest.TestCase):
    def test_execution_seconds_use_environment_frequency_independently_of_video_fps(self):
        frame = np.zeros((4, 4, 3), np.uint8)
        obs = {'front_rgb_list': [frame], 'wrist_rgb_list': [frame]}
        class Dummy(Evaluation):
            method = 'test'
            def act(self, obs, info):
                return np.zeros(8)
        for hz in (20, 100):
            env = SimpleNamespace(control_freq=hz, reset=lambda: (obs, {'task_goal': ['goal']}),
                                  step=lambda _: (obs, 0, True, False, {'status': 'success', 'execution_control_steps': 100}),
                                  close=lambda: None)
            builder = SimpleNamespace(make_env_for_episode=lambda *a, **kw: env,
                                      episode_max_steps=lambda ep: 200)
            with tempfile.TemporaryDirectory() as temporary, patch('evaluation.VideoRecorder'):
                result = asyncio.run(Dummy(SimpleNamespace()).run_episode(builder, 'MoveCube', 0, Path(temporary)/'ep'))
            self.assertEqual(result['control_freq'], hz)
            self.assertEqual(result['metrics']['simulated_seconds'], 100 / hz)

    def test_policy_stop_can_use_one_task_granted_retry(self):
        frame = np.zeros((4, 4, 3), np.uint8)
        obs = {"front_rgb_list": [frame], "wrist_rgb_list": [frame]}
        class Environment:
            trial = 1
            reset_count = 0
            def reset(self):
                self.reset_count += 1
                return obs, {"task_goal": ["goal"]}
            def get_wrapper_attr(self, name):
                if name != "try_execution_retry":
                    raise AttributeError(name)
                return self.retry
            def retry(self, status):
                if self.trial == 2:
                    return None
                self.trial = 2
                return obs, 0., False, False, {"status": "ongoing", "trial_reset": True,
                    "trial_index": 2, "execution_control_steps": 0, "execution_total_control_steps": 0}
            def step(self, action):
                return obs, 0., True, False, {"status": "success", "trial_index": 2,
                    "execution_control_steps": 1, "execution_total_control_steps": 1}
            def close(self):
                pass
        class Dummy(Evaluation):
            method = "test"
            calls = 0
            def act(self, obs, info):
                self.calls += 1
                if self.calls == 1 or self.stop_again:
                    raise PolicyStop("model_answer:done")
                return np.zeros(8)
        for stop_again in (False, True):
            env, policy = Environment(), Dummy(SimpleNamespace())
            policy.stop_again = stop_again
            builder = SimpleNamespace(make_env_for_episode=lambda *a, **kw: env,
                                      episode_max_steps=lambda ep: 3)
            with tempfile.TemporaryDirectory() as tmp, patch("evaluation.VideoRecorder"):
                result = asyncio.run(policy.run_episode(builder, "MoveCube", 0, Path(tmp)/"ep"))
            self.assertEqual(result["success"], not stop_again)
            self.assertEqual(policy.calls, 2)
            self.assertEqual(env.reset_count, 1)

    def test_internal_retry_uses_fresh_budget_and_counts_both_trials(self):
        class RetryEnvironment:
            count = 0
            def reset(self):
                return {}, {"task_goal": ["move cube"]}
            def step(self, action):
                self.count += 1
                retry = self.count == 1
                frame = np.zeros((4, 4, 3), np.uint8)
                return ({"front_rgb_list": [frame], "wrist_rgb_list": [frame]}, 0., not retry, False,
                        {"status": "ongoing" if retry else "success", "trial_index": 2,
                         "execution_control_steps": 0 if retry else 2,
                         "execution_total_control_steps": 3 if retry else 5,
                         **({"trial_reset": True, "message": "Try again"} if retry else {})})
            def close(self):
                pass
        class Dummy(Evaluation):
            method = "test"
            def act(self, obs, info):
                return np.zeros(8)
        env = RetryEnvironment()
        frame = np.zeros((4, 4, 3), np.uint8)
        env.reset = lambda: ({"front_rgb_list": [frame], "wrist_rgb_list": [frame]}, {"task_goal": ["move cube"]})
        builder = SimpleNamespace(make_env_for_episode=lambda *a, **kw: env, episode_max_steps=lambda ep: 3)
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.VideoRecorder"):
            result = asyncio.run(Dummy(SimpleNamespace()).run_episode(builder, "MoveCube", 0, Path(tmp)/"ep"))
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 5)
        self.assertEqual(result["trial_steps"], 2)
        self.assertEqual(env.count, 2)

    def test_video_clean_geometry_all_waypoint_frames_and_demo_alignment(self):
        raw = np.full((32, 32, 3), 99, np.uint8)
        with tempfile.TemporaryDirectory() as tmp, patch("imageio.v2.get_writer") as writer:
            rec = EpisodeRecorder(Path(tmp) / "ep")
            rec.frames({"front_rgb_list": [raw] * 3, "wrist_rgb_list": [raw] * 3}, initial=True)
            rec.step = 4
            rec.frames({"front_rgb_list": [raw] * 4, "wrist_rgb_list": [raw] * 4})
            rec.video.close()
            frames = [c.args[0] for c in writer.return_value.append_data.call_args_list]
            self.assertEqual(len(frames), 7)
            self.assertEqual(frames[0].shape, (32, 64, 3))
            np.testing.assert_array_equal(frames[0][0, 0], [255, 0, 0])
            for frame in frames[2:]:
                self.assertTrue((frame == 99).all())
            self.assertTrue((raw == 99).all())
            rows = [json.loads(s) for s in (rec.directory / "frames.jsonl").read_text().splitlines()]
            self.assertEqual([r["control_step"] for r in rows], [None, None, 0, 1, 2, 3, 4])
            self.assertEqual([r["phase"] for r in rows[:3]], ["demonstration", "demonstration", "execution"])

    def test_api_media_raw_usage_cost_and_failures(self):
        rates = {"source": "test fixture", "as_of": "2026-01-01", "models": {
            "fixture-model": {"input": 10, "cached_input": 1, "cache_write": 12.5, "output": 50}}}
        response = {"model": "fixture-model", "usage": {"input_tokens": 100, "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 5}}, "output": [{"text": "hello"}]}
        blob = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(blob, format="PNG")
        image = "data:image/png;base64," + base64.b64encode(blob.getvalue()).decode()
        with tempfile.TemporaryDirectory() as tmp:
            rec = EpisodeRecorder(Path(tmp) / "ep", rates, api_applicable=True)
            self.assertEqual(rec.metrics()["api"]["estimated_cost_usd"], 0)  # No requests yet.
            rec.call("astra_api", lambda: response, kind="api", model="fixture-model",
                     inputs={"authorization": "secret", "image": image}, prompt="prompt sk-test-secret")
            event = rec.events[-1]
            self.assertAlmostEqual(event["estimated_cost_usd"], (60*10 + 30 + 10*12.5 + 20*50) / 1e6)
            call = rec.directory / event["directory"]
            request = json.loads((call / "input.json").read_text())
            self.assertEqual(request["authorization"], "[REDACTED]")
            self.assertEqual((rec.directory / request["image"]["path"]).read_bytes(), blob.getvalue())
            self.assertNotIn("sk-test", (call / "prompt.txt").read_text())
            with self.assertRaisesRegex(RuntimeError, "transport"):
                rec.call("astra_api", Mock(side_effect=RuntimeError("transport")), kind="api", model="fixture-model")
            api = rec.metrics()["api"]
            self.assertEqual(api["attempts"], 2)
            self.assertEqual(api["priced_attempts"], 1)
            self.assertEqual(api["usage"]["reasoning_tokens"], 5)
            self.assertIsNone(api["estimated_cost_usd"])
            self.assertGreater(api["known_cost_usd"], 0)
        self.assertIsNone(estimate_cost(api_usage(response), "unknown-version", rates))
        self.assertIsNone(estimate_cost(api_usage({}), "fixture-model", rates))

    def test_model_calls_tensors_and_annotations_are_externalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = EpisodeRecorder(Path(tmp) / "ep")
            image = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
            rec.call("ponder", lambda: {"subgoal": "pick", "cognition": np.ones((100, 10))},
                     inputs={"images": [image], "tokens": np.arange(1000)}, prompt="Task: pick")
            rec.annotation(subgoal="pick")
            out = json.loads((rec.directory / "calls/000000_ponder/output.json").read_text())
            self.assertEqual(np.load(rec.directory / out["cognition"]["path"]).shape, (100, 10))
            row = json.loads((rec.directory / "annotations.jsonl").read_text())
            self.assertEqual(row["call_id"], "000000_ponder")
            self.assertEqual(rec.metrics()["api"]["estimated_cost_usd"], 0)
            original_image = rec.artifact(image)["path"]
            image.putpixel((0, 0), (255, 0, 0))
            updated_image = rec.artifact(image)["path"]
            self.assertNotEqual(original_image, updated_image)
            with Image.open(rec.directory / original_image) as saved:
                self.assertEqual(saved.getpixel((0, 0)), (0, 0, 0))

    def test_errors_and_pending_cases_are_not_dropped(self):
        cases = [{"task": "BinFill", "episode": i} for i in range(3)]
        results = [{**cases[0], "success": True, "status": "success"},
                   {**cases[1], "success": False, "status": "error"}]
        report = summarize(cases, results, "test")
        self.assertEqual(report["success_rate"], 1/3)
        self.assertEqual(report["pending"], 1)
        self.assertEqual(report["counts"]["error"], 1)

    def test_every_selected_episode_runs_sequentially_even_after_error(self):
        class Builder:
            active = 0
            order = []
            @staticmethod
            def get_task_list():
                return ["BinFill"]
            def __init__(self, *args, **kwargs):
                pass
            def get_episode_num(self):
                return 3
            def episode_max_steps(self, ep):
                return 10
            def get_episode_metadata(self, ep):
                return {"task": "BinFill", "episode": ep, "seed": 100 + ep,
                        "difficulty": "hard", "oracle_policy_step": 1}
            def make_env_for_episode(self, ep, **kwargs):
                self.assert_no_overlap = Builder.active == 0
                if not self.assert_no_overlap:
                    raise AssertionError("parallel environment")
                Builder.active += 1
                Builder.order.append(ep)
                frame = np.zeros((4, 4, 3), np.uint8)
                obs = {"front_rgb_list": [frame], "wrist_rgb_list": [frame]}
                def reset():
                    if ep == 1:
                        raise RuntimeError("reset fixture failure")
                    return obs, {"task_goal": ["goal"]}
                def close():
                    Builder.active -= 1
                return SimpleNamespace(reset=reset, close=close,
                    step=lambda a: (obs, 0, True, False, {"status": "success", "execution_control_steps": 1}))
        class Dummy(Evaluation):
            method = "test"
            def act(self, obs, info):
                return np.zeros(8)
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.VideoRecorder"), \
                patch("robomme_plus.env_record_wrapper.BenchmarkEnvBuilder", Builder):
            directory = Path(tmp) / "run"
            args = SimpleNamespace(tasks=None, episodes=None, api_pricing_json=None, output_dir=directory, fail_fast=False)
            results = asyncio.run(Dummy(args).run())
            self.assertEqual([r["status"] for r in results], ["success", "error", "success"])
            self.assertEqual(Builder.order, [0, 1, 2])
            self.assertEqual(Builder.active, 0)
            report = json.loads((directory / "summary.json").read_text())
            self.assertEqual(report["success_rate"], 2/3)
            self.assertTrue(report["complete"])
            rows = compare_runs([directory], Path(tmp) / "comparison")
            self.assertEqual(len(rows), 3)
            self.assertTrue((Path(tmp) / "comparison/comparison.csv").is_file())
            self.assertIn("reset fixture failure", (directory / "BinFill_ep1/episode.log").read_text())


if __name__ == "__main__":
    unittest.main()
