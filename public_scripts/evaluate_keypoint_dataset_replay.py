import os
import sys
import json
import numpy as np
from pathlib import Path

# 将上级目录与 scripts 目录加入 Python 路径
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "scripts"))
import gymnasium as gym
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
)
from historybench.HistoryBench_env import *
from save_reset_video import save_listStep_video

import torch

OUTPUT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/data/hongzefu/dataset_generate")


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


def read_metadata(metadata_path):
    """
    从 metadata JSON 文件读取所有 episode 配置。

    参数:
        metadata_path: metadata JSON 文件路径。

    返回:
        list: episode 记录列表，每条记录包含 task、episode、seed、difficulty。
    """
    if not Path(metadata_path).exists():
        print(f"Metadata file not found: {metadata_path}")
        return []

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        episode_records = metadata.get('records', [])
        return episode_records


def main():
    """
    主函数：使用数据集中的 keypoint 执行回放。
    每个 episode 通过 EpisodeDatasetResolver.get_keypoint(step) 逐步取动作，
    整体流程与 evaluate_endeffector_dataset_replay 的结构保持一致。
    """

    env_id_list = [
        "RouteStick",
    ]

    gui_render = True
    render_mode = "human" if gui_render else "rgb_array"
    max_steps_without_demonstration = 200

    for env_id in env_id_list:
        metadata_path = DATASET_ROOT / f"record_dataset_{env_id}_metadata.json"
        h5_path = DATASET_ROOT / f"record_dataset_{env_id}.h5"

        episode_records = read_metadata(metadata_path)
        if not episode_records:
            print(f"No episode records found for {env_id}; skipping")
            continue

        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=str(metadata_path),
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps_without_demonstration,
            action_space="keypoint",
        )

        for episode_record in episode_records:
            episode = episode_record["episode"]
            if episode != 0:
                continue
            seed = episode_record.get("seed")
            difficulty = episode_record.get("difficulty")

            print(f"--- Running simulation for episode:{episode}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")

            env, resolved_seed, resolved_difficulty = config_resolver.make_env_for_episode(episode)
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_path=h5_path,
            )
            obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()

              # ---------- 从每个 obs 读取 frame 等，构建列表 ----------
            maniskill_obs = (obs_batch or {}).get("maniskill_obs", [])
            image = []
            wrist_image = []
            base_camera_depth = []
            base_camera_segmentation = []
            wrist_camera_depth = []
            base_camera_extrinsic_opencv = []
            base_camera_intrinsic_opencv = []
            base_camera_cam2world_opengl = []
            wrist_camera_extrinsic_opencv = []
            wrist_camera_intrinsic_opencv = []
            wrist_camera_cam2world_opengl = []
            robot_endeffector_p = []
            robot_endeffector_q = []
            actions = []
            states = []
            velocity = []
            image.extend(_flatten_column(obs_batch, "image"))
            wrist_image.extend(_flatten_column(obs_batch, "wrist_image"))
            base_camera_depth.extend(_flatten_column(obs_batch, "base_camera_depth"))
            base_camera_segmentation.extend(_flatten_column(obs_batch, "base_camera_segmentation"))
            wrist_camera_depth.extend(_flatten_column(obs_batch, "wrist_camera_depth"))
            base_camera_extrinsic_opencv.extend(_flatten_column(obs_batch, "base_camera_extrinsic_opencv"))
            base_camera_intrinsic_opencv.extend(_flatten_column(obs_batch, "base_camera_intrinsic_opencv"))
            base_camera_cam2world_opengl.extend(_flatten_column(obs_batch, "base_camera_cam2world_opengl"))
            wrist_camera_extrinsic_opencv.extend(_flatten_column(obs_batch, "wrist_camera_extrinsic_opencv"))
            wrist_camera_intrinsic_opencv.extend(_flatten_column(obs_batch, "wrist_camera_intrinsic_opencv"))
            wrist_camera_cam2world_opengl.extend(_flatten_column(obs_batch, "wrist_camera_cam2world_opengl"))
            robot_endeffector_p.extend(_flatten_column(obs_batch, "robot_endeffector_p"))
            robot_endeffector_q.extend(_flatten_column(obs_batch, "robot_endeffector_q"))
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
            n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
            info = _last_info(info_batch, n)
            terminated = bool(terminated_batch[-1].item()) if n > 0 else False
            truncated = bool(truncated_batch[-1].item()) if n > 0 else False

            # 用 reset 后的 image 和 subgoal_grounded 直接保存为带字幕视频
            out_video_dir = DATASET_ROOT / "videos"
            os.makedirs(out_video_dir, exist_ok=True)
            reset_captioned_path = os.path.join(out_video_dir, f"replay_ee_{env_id}_ep{episode}_reset_captioned.mp4")
            save_listStep_video(obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch, reset_captioned_path)





            video_dir = DATASET_ROOT / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            out_video_path = video_dir / f"replay_kp_{env_id}_ep{episode}.mp4"
            fps = 20
            episode_success = False
            replay_frames = []
            replay_subgoal_grounded = []

            step = 0
            while True:
                action = dataset_resolver.get_keypoint(step)
                if action is None:
                    break

                print(f"  Executing keypoint {step+1}: keypoint_p: {action[:3]}")
                action = action.astype(np.float32)

                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

                # 从每个 obs 读取 frame 等，构建列表
                maniskill_obs = (obs_batch or {}).get("maniskill_obs", [])
                image = []
                wrist_image = []
                base_camera_depth = []
                base_camera_segmentation = []
                wrist_camera_depth = []
                base_camera_extrinsic_opencv = []
                base_camera_intrinsic_opencv = []
                base_camera_cam2world_opengl = []
                wrist_camera_extrinsic_opencv = []
                wrist_camera_intrinsic_opencv = []
                wrist_camera_cam2world_opengl = []
                robot_endeffector_p = []
                robot_endeffector_q = []
                actions = []
                states = []
                velocity = []
                image.extend(_flatten_column(obs_batch, "image"))
                wrist_image.extend(_flatten_column(obs_batch, "wrist_image"))
                base_camera_depth.extend(_flatten_column(obs_batch, "base_camera_depth"))
                base_camera_segmentation.extend(_flatten_column(obs_batch, "base_camera_segmentation"))
                wrist_camera_depth.extend(_flatten_column(obs_batch, "wrist_camera_depth"))
                base_camera_extrinsic_opencv.extend(_flatten_column(obs_batch, "base_camera_extrinsic_opencv"))
                base_camera_intrinsic_opencv.extend(_flatten_column(obs_batch, "base_camera_intrinsic_opencv"))
                base_camera_cam2world_opengl.extend(_flatten_column(obs_batch, "base_camera_cam2world_opengl"))
                wrist_camera_extrinsic_opencv.extend(_flatten_column(obs_batch, "wrist_camera_extrinsic_opencv"))
                wrist_camera_intrinsic_opencv.extend(_flatten_column(obs_batch, "wrist_camera_intrinsic_opencv"))
                wrist_camera_cam2world_opengl.extend(_flatten_column(obs_batch, "wrist_camera_cam2world_opengl"))
                robot_endeffector_p.extend(_flatten_column(obs_batch, "robot_endeffector_p"))
                robot_endeffector_q.extend(_flatten_column(obs_batch, "robot_endeffector_q"))
                actions.extend(_flatten_column(obs_batch, "actions"))
                states.extend(_flatten_column(obs_batch, "states"))
                velocity.extend(_flatten_column(obs_batch, "velocity"))
                language_goal_list = (obs_batch or {}).get("language_goal", [])
                language_goal = language_goal_list[-1] if language_goal_list else None

                # 从每个 info 读取
                subgoal = []
                subgoal_grounded = []
                subgoal.extend(_flatten_column(info_batch, "subgoal"))
                subgoal_grounded.extend(_flatten_column(info_batch, "subgoal_grounded"))
                if image:
                    replay_frames.append(np.asarray(image[-1]).copy())
                if subgoal_grounded:
                    replay_subgoal_grounded.append(subgoal_grounded[-1])

                # 用最后一步的 terminated/truncated/info 做循环判断
                n = int(reward_batch.numel()) if hasattr(reward_batch, "numel") else 0
                terminated = bool(terminated_batch[-1].item()) if n > 0 else False
                truncated = bool(truncated_batch[-1].item()) if n > 0 else False
                info = _last_info(info_batch, n)

                # 保存当前步骤的带字幕视频
                kp_captioned_path = video_dir / f"replay_kp_{env_id}_ep{episode}_kp{step}_captioned.mp4"
                save_listStep_video(obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch, str(kp_captioned_path), fps=fps)



                if gui_render:
                    env.render()
                step += 1

                if truncated:
                    print(f"[{env_id}] episode {episode} 步数超限。")
                    break
                if terminated:
                    if info.get("success") == torch.tensor([True]) or (
                        isinstance(info.get("success"), torch.Tensor) and info.get("success").item()
                    ):
                        print(f"[{env_id}] episode {episode} 成功。")
                        episode_success = True
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} 失败。")
                    break

            if replay_frames and replay_subgoal_grounded:
                obs_video = {"image": replay_frames}
                info_video = {"subgoal_grounded": replay_subgoal_grounded}
                save_listStep_video(obs_video, reward_batch, terminated_batch, truncated_batch, info_video, str(out_video_path), fps=fps)
            print(f"Saved video: {out_video_path}")

            dataset_resolver.close()
            env.close()
            print(f"--- Finished Running simulation for episode:{episode}, env: {env_id} ---")


if __name__ == "__main__":
    main()
