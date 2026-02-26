"""
test_TaskGoalIsList.py

直接创建真实 Gymnasium 环境（包裹 DemonstrationWrapper），
调用 env.reset()，验证 info["task_goal"] 是 list 且非空。

全部 16 个 env 均覆盖。

运行：
    uv run python -m pytest tests/lightweight/test_TaskGoalI_isList.py -v -s
"""

import gymnasium as gym
import pytest
from typing import Literal

from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from robomme.env_record_wrapper.EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper
from robomme.env_record_wrapper.MultiStepDemonstrationWrapper import MultiStepDemonstrationWrapper
from robomme.env_record_wrapper.OraclePlannerDemonstrationWrapper import OraclePlannerDemonstrationWrapper

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

# ── 全部 16 个 env_id ──────────────────────────────────────────────────────────
ALL_ENV_IDS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]

# ── 四种 ActionSpaceType ───────────────────────────────────────────────────────
ACTION_SPACES = ["joint_angle", "ee_pose", "waypoint", "multi_choice"]


def _make_env(env_id: str, action_space: str):
    """创建并返回包裹了相应 Wrapper 的真实环境。"""
    env = gym.make(
        env_id,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    # 基础 Wrapper
    env = DemonstrationWrapper(
        env,
        max_steps_without_demonstration=10002,
        gui_render=False,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )

    # 根据 action_space 应用额外的 Wrapper
    if action_space == "joint_angle":
        pass
    elif action_space == "ee_pose":
        env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")
    elif action_space == "waypoint":
        env = MultiStepDemonstrationWrapper(env, gui_render=False, vis=False)
    elif action_space == "multi_choice":
        env = OraclePlannerDemonstrationWrapper(env, env_id=env_id, gui_render=False)
    else:
        raise ValueError(f"Unsupported action_space: {action_space}")

    return env


@pytest.mark.parametrize(
    "env_id, action_space",
    [(env, action) for env in ALL_ENV_IDS for action in ACTION_SPACES],
)
def test_task_goal_is_list(env_id: str, action_space: str):
    """
    对每个 env_id 连续测试四种 action_space：
    1. 创建真实环境（含相应的 Wrapper）
    2. 调用 reset()
    3. 断言 info["task_goal"] 是 list 且非空
    """
    print(f"\nTesting [{env_id}] with action_space={action_space!r}")
    env = _make_env(env_id, action_space)
    try:
        _, info = env.reset()
    finally:
        env.close()

    task_goal = info["task_goal"]
    print(f"[{env_id} | {action_space}] task_goal = {task_goal!r}")

    assert isinstance(task_goal, list), (
        f"[{env_id} | {action_space}] info['task_goal'] 应为 list，实际为 {type(task_goal).__name__!r}: {task_goal!r}"
    )
    assert len(task_goal) >= 1, (
        f"[{env_id} | {action_space}] info['task_goal'] 不应为空 list"
    )
    for i, item in enumerate(task_goal):
        assert isinstance(item, str), (
            f"[{env_id} | {action_space}] task_goal[{i}] 应为 str，实际为 {type(item).__name__!r}: {item!r}"
        )
