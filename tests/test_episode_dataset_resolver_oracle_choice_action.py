import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

import h5py


def _load_episode_dataset_resolver_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/robomme/env_record_wrapper/episode_dataset_resolver.py"
    )
    spec = importlib.util.spec_from_file_location("episode_dataset_resolver", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_RESOLVER_MODULE = _load_episode_dataset_resolver_module()
EpisodeDatasetResolver = _RESOLVER_MODULE.EpisodeDatasetResolver


def _write_timestep(
    episode_group: h5py.Group,
    idx: int,
    *,
    is_video_demo: bool = False,
    choice_action: str = None,
    grounded_subgoal: str = None,
) -> None:
    ts = episode_group.create_group(f"timestep_{idx}")
    info = ts.create_group("info")
    info.create_dataset("is_video_demo", data=is_video_demo)
    if grounded_subgoal is not None:
        info.create_dataset("grounded_subgoal", data=grounded_subgoal)
    action = ts.create_group("action")
    if choice_action is not None:
        action.create_dataset("choice_action", data=choice_action)


class TestEpisodeDatasetResolverOracleChoiceAction(unittest.TestCase):
    def test_oracle_choice_action_only_and_point_normalized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "record_dataset_TestEnv.h5"
            with h5py.File(h5_path, "w") as h5:
                ep = h5.create_group("episode_0")
                _write_timestep(
                    ep,
                    0,
                    choice_action=json.dumps(
                        {"action": "pick up the cube", "point": [120, 300], "serial_number": 1}
                    ),
                )
                _write_timestep(
                    ep,
                    1,
                    choice_action=json.dumps(
                        {"action": "pick up the cube", "point": [999, 888], "serial_number": 1}
                    ),
                )
                _write_timestep(
                    ep,
                    2,
                    grounded_subgoal="press the button at <1,2>",
                )
                _write_timestep(
                    ep,
                    3,
                    choice_action=json.dumps({"action": "press the button", "serial_number": 2}),
                )

            resolver = EpisodeDatasetResolver(
                env_id="TestEnv",
                episode=0,
                dataset_directory=h5_path,
            )
            try:
                step0 = resolver.get_step("oracle_planner", 0)
                self.assertEqual(step0["action"], "pick up the cube")
                self.assertEqual(step0["point"], [300, 120])
                self.assertEqual(step0["serial_number"], 1)

                step1 = resolver.get_step("oracle_planner", 1)
                self.assertEqual(step1["action"], "press the button")
                self.assertEqual(step1["serial_number"], 2)
                self.assertNotIn("point", step1)

                self.assertIsNone(resolver.get_step("oracle_planner", 2))
            finally:
                resolver.close()

    def test_oracle_dedup_by_action_point_without_serial(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = Path(tmp_dir) / "record_dataset_TestEnv.h5"
            with h5py.File(h5_path, "w") as h5:
                ep = h5.create_group("episode_0")
                _write_timestep(
                    ep,
                    0,
                    choice_action=json.dumps({"action": "pick up the cube", "point": [1, 2]}),
                )
                _write_timestep(
                    ep,
                    1,
                    choice_action=json.dumps({"action": "pick up the cube", "point": [1, 2]}),
                )
                _write_timestep(
                    ep,
                    2,
                    choice_action=json.dumps({"action": "pick up the cube", "point": [3, 4]}),
                )
                _write_timestep(
                    ep,
                    3,
                    choice_action="{invalid_json}",
                )

            resolver = EpisodeDatasetResolver(
                env_id="TestEnv",
                episode=0,
                dataset_directory=h5_path,
            )
            try:
                step0 = resolver.get_step("oracle_planner", 0)
                self.assertEqual(step0, {"action": "pick up the cube", "point": [2, 1]})

                step1 = resolver.get_step("oracle_planner", 1)
                self.assertEqual(step1, {"action": "pick up the cube", "point": [4, 3]})

                self.assertIsNone(resolver.get_step("oracle_planner", 2))
            finally:
                resolver.close()


if __name__ == "__main__":
    unittest.main()
