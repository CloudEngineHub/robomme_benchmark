# 从 dataset_generate 读取数据集并回放轨迹（跳过 demonstration 阶段，只执行非演示动作）

import os
import sys
import re
from pathlib import Path
from typing import Any, List, Optional, cast

# 将包根目录加入 path，便于作为脚本直接运行
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import gymnasium as gym
import h5py
import torch

from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import *

# 数据集根目录（包含 record_dataset_*_metadata.json 与 record_dataset_*.h5）
DATASET_ROOT = "/data/hongzefu/dataset_generate"


def discover_env_ids(dataset_root: str) -> List[str]:
    """
    从数据集根目录发现所有环境 ID。
    仅当存在对应的 record_dataset_{env_id}_metadata.json 时，才将 env_id 加入列表。
    """
    dataset_path = Path(dataset_root)
    if not dataset_path.is_dir():
        return []
    prefix = "record_dataset_"
    suffix = ".h5"
    env_ids = []
    for entry in sorted(dataset_path.iterdir()):
        if not entry.is_file() or not entry.name.startswith(prefix) or not entry.name.endswith(suffix):
            continue
        env_id = entry.name[len(prefix) : -len(suffix)]
        metadata_file = dataset_path / f"record_dataset_{env_id}_metadata.json"
        if metadata_file.exists():
            env_ids.append(env_id)
    return env_ids


def _as_bool(value) -> bool:
    """将 h5 标量 / tensor / 数组 / None 安全转换为 bool。"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        flat = value.detach().cpu().reshape(-1)
        return bool(flat[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(np.reshape(value, -1)[0].item())
    if hasattr(value, "decode"):
        value = value.decode("utf-8") if isinstance(value, bytes) else value
    return bool(value) if value is not None else False


def _action_to_8d(action) -> np.ndarray:
    """将动作统一为 8 维（7 关节 + 夹爪）；若为 7 维则末尾补 -1。"""
    action = np.asarray(action, dtype=np.float64).flatten()
    if action.size == 7:
        action = np.concatenate([action, [-1.0]])
    elif action.size < 8:
        action = np.pad(action, (0, 8 - action.size), constant_values=-1.0)
    return action[:8].astype(np.float64)


def main():
    """
    从 DATASET_ROOT 读取数据集，回放所有 episode；
    每个 episode 仅回放非 demonstration 的动作（跳过 demonstration 步）。
    """
    num_episodes_limit = None  # None 表示回放全部 episode；设为整数可限制数量
    gui_render = True
    max_steps = 3000

    render_mode = "human" if gui_render else "rgb_array"

    # 确定要回放的环境：env_id_filter 为 None 时使用发现到的全部环境，否则使用过滤列表
    env_id_filter: Optional[List[str]] = ["VideoRepick"]  # None = 使用发现到的所有环境
    env_id_list = discover_env_ids(DATASET_ROOT) if env_id_filter is None else env_id_filter
    if not env_id_list:
        print("没有可运行的环境 ID（发现列表或过滤列表为空）。")
        return

    for env_id in env_id_list:
        metadata_path = f"{DATASET_ROOT}/record_dataset_{env_id}_metadata.json"
        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"
        if not Path(metadata_path).exists():
            print(f"元数据不存在: {metadata_path}，跳过环境 {env_id}。")
            continue
        if not Path(h5_path).exists():
            print(f"H5 文件不存在: {h5_path}，跳过环境 {env_id}。")
            continue

        resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )

        # 从 h5 中读取该环境下的所有 episode 索引
        with h5py.File(h5_path, "r") as h5:
            env_group_key = f"env_{env_id}"
            if env_group_key not in h5:
                print(f"h5 中缺少组 '{env_group_key}'，跳过环境 {env_id}。")
                continue
            env_group = cast(h5py.Group, h5[env_group_key])
            episode_indices = sorted(
                int(k.split("_")[1])
                for k in env_group.keys()
                if k.startswith("episode_") and re.match(r"episode_\d+", k)
            )
            if not episode_indices:
                print(f"h5 中无 episode，跳过环境 {env_id}。")
                continue
            if num_episodes_limit is not None:
                episode_indices = episode_indices[:num_episodes_limit]

        for episode in episode_indices:
            env, seed, difficulty = resolver.make_env_for_episode(episode)

            with h5py.File(h5_path, "r") as h5:
                env_group = cast(h5py.Group, h5[f"env_{env_id}"])
                episode_key = f"episode_{episode}"
                if episode_key not in env_group:
                    print(f"h5 中无 {episode_key}，跳过。")
                    env.close()
                    continue
                episode_dataset = cast(h5py.Group, env_group[episode_key])

                obs, info = env.reset()

                # 本 episode 内所有 record_timestep_* 的步数索引（按顺序）
                timestep_indexes = sorted(
                    int(m.group(1))
                    for k in episode_dataset.keys()
                    if (m := re.match(r"record_timestep_(\d+)$", k))
                )

                for step in timestep_indexes:
                    timestep_group = cast(Any, episode_dataset[f"record_timestep_{step}"])
                    # 演示步不执行，只回放非演示动作
                    if _as_bool(timestep_group["demonstration"][()]):
                        continue
                    raw_action = timestep_group["action"][()]
                    if hasattr(raw_action, "decode"):
                        raw_action = raw_action.decode("utf-8") if isinstance(raw_action, bytes) else raw_action
                    if raw_action is None or (isinstance(raw_action, str) and raw_action.strip().lower() == "none"):
                        continue
                    action = _action_to_8d(raw_action)

                    obs, reward, terminated, truncated, info = env.step(action)

                    if gui_render:
                        env.render()
                    if truncated:
                        print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                        break
                    if terminated.any():
                        if info.get("success") == torch.tensor([True]) or (
                            isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                        ):
                            print(f"[{env_id}] episode {episode} 成功。")
                        elif info.get("fail", False):
                            print(f"[{env_id}] episode {episode} 失败。")
                        break

            env.close()

if __name__ == "__main__":
    main()
