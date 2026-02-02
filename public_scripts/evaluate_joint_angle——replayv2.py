# 从 dataset_generate 读取数据集并回放轨迹（跳过 demonstration 阶段，只执行非演示动作）

import os
import sys
from pathlib import Path
from typing import List, Optional

# 将包根目录加入 path，便于作为脚本直接运行
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
for _path in (_PARENT, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gymnasium as gym
import torch

from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
    list_episode_indices,
)

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

        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )

        try:
            episode_indices = list_episode_indices(env_id, DATASET_ROOT)
        except (FileNotFoundError, KeyError) as e:
            print(f"跳过环境 {env_id}: {e}")
            continue
        if not episode_indices:
            print(f"h5 中无 episode，跳过环境 {env_id}。")
            continue
        if num_episodes_limit is not None:
            episode_indices = episode_indices[:num_episodes_limit]

        for episode in episode_indices:
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)
            try:
                with EpisodeDatasetResolver(env_id, episode, DATASET_ROOT) as dataset_resolver:
                    obs, info = env.reset()
                    for step, action, is_demo in dataset_resolver.iter_timesteps():
                        if is_demo:
                            continue
                        if action is None:
                            continue
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
            except (FileNotFoundError, KeyError) as e:
                print(f"h5 中无 episode_{episode} 或读取失败: {e}，跳过。")
            finally:
                env.close()

if __name__ == "__main__":
    main()
