"""
核心理念：将“整个 Episode”视为一个原子操作。仅保留与环境交互部分，无 query model。

阶段 A 初始化 -> 阶段 B 步骤循环 (step_before -> command_dict -> step_after) -> 阶段 C 保存状态。
若发生异常：env.close(), del env, gc.collect()，记录 sim_error 后继续下一 episode；无 episode 重试。
"""
import os
import sys

# 添加项目根目录到 Python 路径，以便正确导入模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import json
import shutil
import torch
import gc
import re

from historybench.env_record_wrapper import EpisodeConfigResolver, EpisodeDatasetResolver
from pathlib import Path
from save_reset_video import save_listStep_video


def _flatten_column(batch_dict, key):
    out = []
    for item in (batch_dict or {}).get(key, []) or []:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            out.extend([x for x in item if x is not None])
        else:
            out.append(item)
    return out


def _last_info(info_batch, n):
    if n <= 0:
        return {}
    idx = n - 1
    return {k: v[idx] for k, v in (info_batch or {}).items() if len(v) > idx and v[idx] is not None}


def main():
    # 数据集根目录
    dataset_root = Path("/data/hongzefu/dataset_generate")
    
    env_id_list = [
        # "PickXtimes",
        # "StopCube",
    #"SwingXtimes",
      #  "BinFill",

    #     "VideoUnmaskSwap",
    #     "VideoUnmask",
    #     "ButtonUnmaskSwap",
    #     "ButtonUnmask",

         "VideoRepick",
    #     "VideoPlaceButton",
    #      "VideoPlaceOrder",
    #     "PickHighlight",

    #     "InsertPeg",
    #     'MoveCube',
    #     "PatternLock",
    #     "RouteStick"
    ]

    for env_id in env_id_list:

    # 为每个环境初始化配置解析器
        metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
        resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode="human",
            gui_render=True,
            max_steps_without_demonstration=1000,
            action_space="oracle_planner"
        )

        # 如需全量回放可改为：for episode in range(num_episodes):
        for episode in range(1):
            # 若只想调试某个 episode，可启用以下过滤：
            # if episode != 1:
            #     continue
            
            model_name = "env_only"
            save_dir = '/data/hongzefu/dataset_generate/videos'
            
            # 使用 EpisodeConfigResolver
            env, seed, difficulty = resolver.make_env_for_episode(episode)
            
            h5_path = dataset_root / f"record_dataset_{env_id}.h5"
            dataset_resolver = EpisodeDatasetResolver(env_id, episode, h5_path)

            obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

              # ---------- 从每个 obs 读取 frame 等，构建列表 ----------
            frames = []
            wrist_frames = []
            actions = []
            states = []
            velocity = []
            frames.extend(_flatten_column(obs_batch, "frames"))
            wrist_frames.extend(_flatten_column(obs_batch, "wrist_frames"))
            actions.extend(_flatten_column(obs_batch, "actions"))
            states.extend(_flatten_column(obs_batch, "states"))
            velocity.extend(_flatten_column(obs_batch, "velocity"))
            language_goal_list = (obs_batch or {}).get("language_goal", [])
            language_goal = language_goal_list[0] if language_goal_list else None

            # ---------- 从每个 info 读取子目标等 ----------
            subgoal = []
            subgoal_grounded = []
            subgoal.extend(_flatten_column(info_batch, "subgoal"))
            subgoal_grounded.extend(_flatten_column(info_batch, "subgoal_grounded"))

            
            success = "fail"

            video_dir = dataset_root / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            out_video_path = video_dir / f"replay_op_{env_id}_ep{episode}.mp4"

            reset_captioned_path = video_dir / f"replay_op_{env_id}_ep{episode}_reset_captioned.mp4"
            save_listStep_video(obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch, str(reset_captioned_path))
            os.makedirs(save_dir, exist_ok=True)

            step_idx = 0
            max_query_times = 10

            # 阶段 B：执行步骤循环
            while True:
                if step_idx >= max_query_times:
                    print(f"Max query times ({max_query_times}) reached, stopping.")
                    break



                # 每次获得的 obs_batch 都保存为带字幕视频（save_listStep_video，与关键点回放一致）
                step_pre_captioned_path = video_dir / f"replay_op_{env_id}_ep{episode}_step{step_idx}_pre_captioned.mp4"
                save_listStep_video(obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch, str(step_pre_captioned_path))

                subgoal_text = dataset_resolver.get_grounded_subgoal(step_idx)
                if subgoal_text is None:
                    print(f"No more subgoals at step {step_idx}. Exiting loop.")
                    break
                
                point = None
                match = re.search(r'<(\d+),\s*(\d+)>', subgoal_text)
                if match:
                    point = [int(match.group(1)), int(match.group(2))]

                command_dict = {"action": subgoal_text, "point": point}

                if command_dict["point"] is not None:
                    command_dict["point"] = command_dict["point"][::-1]

                print(f"\nCommand: {command_dict}")

                step_idx += 1

                # 步骤 2：执行 step（相当于 step_after + step_before）
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(command_dict)

                # 从每个 obs 读取 frame 等，构建列表
                frames = []
                wrist_frames = []
                actions = []
                states = []
                velocity = []
                frames.extend(_flatten_column(obs_batch, "frames"))
                wrist_frames.extend(_flatten_column(obs_batch, "wrist_frames"))
                actions.extend(_flatten_column(obs_batch, "actions"))
                states.extend(_flatten_column(obs_batch, "states"))
                velocity.extend(_flatten_column(obs_batch, "velocity"))
                language_goal_list = (obs_batch or {}).get("language_goal", [])
                language_goal = language_goal_list[0] if language_goal_list else None

                # 从每个 info 读取
                subgoal = []
                subgoal_grounded = []
                subgoal.extend(_flatten_column(info_batch, "subgoal"))
                subgoal_grounded.extend(_flatten_column(info_batch, "subgoal_grounded"))

                # 保存当前 step 的带字幕视频（与关键点回放一致）
                step_captioned_path = video_dir / f"replay_op_{env_id}_ep{episode}_step{step_idx}_captioned.mp4"
                save_listStep_video(obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch, str(step_captioned_path))

                # 用最后一步的 terminated/truncated/info 做循环判断
                n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
                terminated = bool(terminated_batch[-1].item()) if n > 0 else False
                truncated = bool(truncated_batch[-1].item()) if n > 0 else False
                info = _last_info(info_batch, n)


                # 步数达到上限
                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限，步 {step_idx}。")
                    break
                # 任务结束（成功或失败）
                if terminated:
                    if info.get("success") == torch.tensor([True]) or (
                        isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break


            

            # 保存回放视频（frames + subgoal_grounded）
            env.save_video(str(out_video_path))
            print(f"Saved video: {out_video_path}")



            # 阶段 C：成功标记与保存
            env.close()
            dataset_resolver.close()
                      
    # oracle_resolver.close()  # 当前逻辑为每轮循环内创建解析器，无需额外关闭
    
if __name__ == "__main__":
    main()
