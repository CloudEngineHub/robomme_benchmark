from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.dataset

_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.robomme_env import *  # noqa: F401,F403,E402
from robomme.robomme_env.utils import *  # noqa: F401,F403,E402
from robomme.env_record_wrapper import BenchmarkEnvBuilder, EpisodeDatasetResolver  # noqa: E402


TEST_ENV_ID = "VideoUnmaskSwap"
TEST_EPISODE = 0
MAX_STEPS_ENV = 1000

ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]
ACTION_SPACES: list[ActionSpaceType] = [
    "joint_angle",
    "ee_pose",
    "waypoint",
    "multi_choice",
]


def _to_bool_flag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(np.reshape(value, -1)[0].item())
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except Exception:
            pass
    return bool(value)


def _task_goal_non_empty(task_goal: Any) -> bool:
    if task_goal is None:
        return False
    if isinstance(task_goal, str):
        return bool(task_goal.strip())
    if isinstance(task_goal, list):
        return len(task_goal) > 0
    return True


def _make_env(action_space: ActionSpaceType):
    builder = BenchmarkEnvBuilder(
        env_id=TEST_ENV_ID,
        dataset="train",
        action_space=action_space,
        gui_render=False,
    )
    return builder.make_env_for_episode(
        TEST_EPISODE,
        max_steps=MAX_STEPS_ENV,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )


def _replay_one_mode(action_space: ActionSpaceType, dataset_root: Path) -> dict[str, Any]:
    env = None
    resolver = None

    steps_executed = 0
    terminated_flag = False
    truncated_flag = False
    ended_by_no_action = False
    last_status = None
    task_goal_at_reset = None

    try:
        env = _make_env(action_space)
        resolver = EpisodeDatasetResolver(
            env_id=TEST_ENV_ID,
            episode=TEST_EPISODE,
            dataset_directory=str(dataset_root),
        )

        _, info = env.reset()
        task_goal_at_reset = info.get("task_goal")
        last_status = info.get("status")

        step_idx = 0
        while True:
            action = resolver.get_step(action_space, step_idx)
            if action is None:
                ended_by_no_action = True
                break

            try:
                _, _, terminated, truncated, info = env.step(action)
            except Exception as exc:
                raise AssertionError(
                    f"[{action_space}] env.step failed at replay step {step_idx}: {exc}"
                ) from exc

            steps_executed += 1
            terminated_flag = _to_bool_flag(terminated)
            truncated_flag = _to_bool_flag(truncated)
            last_status = info.get("status")

            step_idx += 1
            if terminated_flag or truncated_flag:
                break
    finally:
        if resolver is not None:
            resolver.close()
        if env is not None:
            env.close()

    return {
        "action_space": action_space,
        "steps_executed": steps_executed,
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "status": last_status,
        "task_goal_at_reset": task_goal_at_reset,
        "ended_by_no_action": ended_by_no_action,
    }


def test_replay_videounmaskswap_four_modes_consistent(video_unmaskswap_train_ep0_dataset) -> None:
    results: dict[str, dict[str, Any]] = {}
    dataset_root = video_unmaskswap_train_ep0_dataset.resolver_dataset_dir

    for action_space in ACTION_SPACES:
        summary = _replay_one_mode(action_space, dataset_root)
        results[action_space] = summary
        print(f"[replay summary] {action_space}: {summary}")

        assert summary["steps_executed"] >= 1, (
            f"[{action_space}] replay executed 0 steps. summary={summary}"
        )
        assert _task_goal_non_empty(summary["task_goal_at_reset"]), (
            f"[{action_space}] task_goal at reset is empty. summary={summary}"
        )

    baseline = results["joint_angle"]
    for action_space in ACTION_SPACES[1:]:
        current = results[action_space]
        assert current["terminated"] == baseline["terminated"], (
            f"[{action_space}] terminated mismatch vs joint_angle. "
            f"baseline={baseline}, current={current}"
        )
        assert current["truncated"] == baseline["truncated"], (
            f"[{action_space}] truncated mismatch vs joint_angle. "
            f"baseline={baseline}, current={current}"
        )
        assert current["status"] == baseline["status"], (
            f"[{action_space}] status mismatch vs joint_angle. "
            f"baseline={baseline}, current={current}"
        )

