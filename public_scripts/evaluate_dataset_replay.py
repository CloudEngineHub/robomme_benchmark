# -*- coding: utf-8 -*-
# 脚本功能：统一 dataset replay 入口，支持 joint_angle / ee_pose / ee_quat / keypoint / oracle_planner 五种 action_space。
# 与 evaluate.py 的主循环与调试字段保持一致；差异在于动作来自 EpisodeDatasetResolver。

import os
import re
import sys
from typing import Any, Optional

# 将包根目录、上级目录及 scripts 加入 sys.path，便于作为脚本直接运行（不依赖 PYTHONPATH）
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
_SCRIPTS = os.path.join(_PARENT, "scripts")
for _path in (_PARENT, _ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
import cv2

from robomme.robomme_env import *
from robomme.robomme_env.util import *
from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
    EpisodeDatasetResolver,
)
from robomme.robomme_env.util.save_reset_video import save_robomme_video

# 只启用一个 ACTION_SPACE；其他选项保留在注释中供手动切换
ACTION_SPACE = "joint_angle"
#ACTION_SPACE = "ee_pose"
#ACTION_SPACE = "ee_quat"
#ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = False
MAX_STEPS = 3000
DATASET_ROOT = "/data/hongzefu/dataset_0211"
#OVERRIDE_METADATA_PATH = "/data/hongzefu/dataset_generate-b4"   

DEFAULT_ENV_IDS = [
    # "PickXtimes",
    # "StopCube",
    # "SwingXtimes",
    #"BinFill",
    # "VideoUnmaskSwap",
    # "VideoUnmask",
    # "ButtonUnmaskSwap",
    # "ButtonUnmask",
     "VideoRepick",
    # "VideoPlaceButton",
    # "VideoPlaceOrder",
   # "PickHighlight",
    # "InsertPeg",
    # "MoveCube",
    # "PatternLock",
    # "RouteStick",
]

# ######## 视频保存变量（输出目录）开始 ########
# 视频输出目录：独立固定写死，不与 h5 路径或 env_id 对齐
OUT_VIDEO_DIR = "/data/hongzefu/dataset_replay"
# ######## 视频保存变量（输出目录）结束 ########

def _parse_oracle_command(subgoal_text: Optional[str]) -> Optional[dict[str, Any]]:
    if not subgoal_text:
        return None
    point = None
    match = re.search(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>", subgoal_text)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        # 数据集文本通常是 <x, y>，Oracle wrapper 期望 [row, col]，即 [y, x]
        point = [y, x]
    return {"action": subgoal_text, "point": point}


def main():
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    

    #for env_id in env_id_list:
    for env_id in DEFAULT_ENV_IDS:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
            #override_metadata_path=OVERRIDE_METADATA_PATH,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        for episode in range(episode_count):

            env = None
            dataset_resolver = None
            try:
                env, seed, difficulty = env_builder.make_env_for_episode(episode)
                dataset_resolver = EpisodeDatasetResolver(
                    env_id=env_id,
                    episode=episode,
                    dataset_directory=DATASET_ROOT,
                    
                )

                # obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()
                obs_batch, info_batch = env.reset()

                # 保持 evaluate.py 中的调试变量语义
                maniskill_obs = obs_batch["maniskill_obs"]
                front_camera = obs_batch["front_camera"]
                wrist_camera = obs_batch["wrist_camera"]
                front_camera_depth = obs_batch["front_camera_depth"]
                front_camera_segmentation = obs_batch["front_camera_segmentation"]
                wrist_camera_depth = obs_batch["wrist_camera_depth"]
                front_camera_extrinsic_opencv = obs_batch["front_camera_extrinsic_opencv"]
                front_camera_intrinsic_opencv = obs_batch["front_camera_intrinsic_opencv"]
                front_camera_cam2world_opengl = obs_batch["front_camera_cam2world_opengl"]
                wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
                wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
                wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
                end_effector_pose_raw = obs_batch["end_effector_pose_raw"]
                end_effector_pose = obs_batch["end_effector_pose"]
                joint_states = obs_batch["joint_states"]
                velocity = obs_batch["velocity"]
                language_goal_list = info_batch["language_goal"]
                language_goal = language_goal_list[0] if language_goal_list else None

                subgoal = info_batch["subgoal"]
                subgoal_grounded = info_batch["subgoal_grounded"]
                available_options = info_batch["available_options"]

                info = {k: v[-1] for k, v in info_batch.items()}
                # terminated = bool(terminated_batch[-1].item())
                # truncated = bool(truncated_batch[-1].item())

                # #todo：保存最后两张front camera为图片 左右拼接加上注释
                # if len(front_camera) >= 2:
                #     def _tensor_to_numpy_img(f):
                #         img = torch.as_tensor(f).detach().cpu().numpy()
                #         if img.dtype != np.uint8:
                #             # 假设是[0,1] float，转为[0,255] uint8
                #             if img.max() <= 1.0:
                #                 img = (img * 255).astype(np.uint8)
                #             else:
                #                 img = img.astype(np.uint8)
                #         return img.copy()  # Ensure writable copy

                #     def _draw_text_with_wrap(img, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(0, 255, 0), thickness=1):
                #         """绘制文本，支持自动换行"""
                #         if not text:
                #             return img
                        
                #         img_h, img_w = img.shape[:2]
                #         x, y = position
                #         line_height = int(30 * font_scale) + 5
                        
                #         words = text.split(' ')
                #         current_line = ""
                        
                #         # 简单的逐词换行逻辑
                #         for word in words:
                #             test_line = current_line + word + " "
                #             (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                #             if x + w > img_w - 10:  # 留出右侧 margin
                #                 # 绘制当前行
                #                 cv2.putText(img, current_line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                #                 # 重置新行
                #                 current_line = word + " "
                #                 y += line_height
                #             else:
                #                 current_line = test_line
                        
                #         # 绘制最后一行
                #         if current_line:
                #             cv2.putText(img, current_line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                        
                #         return img

                #     img_prev = _tensor_to_numpy_img(front_camera[-2])
                #     img_curr = _tensor_to_numpy_img(front_camera[-1])

                #     # 为两张图分别添加对应的 subgoal
                #     # 注意：subgoal_grounded 的长度可能与 front_camera 一致，取倒数第二个和最后一个
                #     subgoal_text_prev = str(subgoal_grounded[-2]) if len(subgoal_grounded) >= 2 else "No Subgoal"
                #     subgoal_text_curr = str(subgoal_grounded[-1]) if subgoal_grounded else "No Subgoal"
                    
                #     # 绘制文字
                #     _draw_text_with_wrap(img_prev, f"Prev: {subgoal_text_prev}")
                #     _draw_text_with_wrap(img_curr, f"Curr: {subgoal_text_curr}")

                #     # 水平拼接
                #     concat_img = np.hstack((img_prev, img_curr))
                    
                #     # 转换为 BGR 用于保存
                #     concat_img_bgr = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
                    
                #     save_path = os.path.join(OUT_VIDEO_DIR, f"{env_id}-{episode}-reset-comparison.png")
                #     os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
                #     cv2.imwrite(save_path, concat_img_bgr)
                #     print(f"[{env_id}] episode {episode} reset comparison image saved to {save_path}")
                

                # ######## 视频保存变量准备（reset 阶段）开始 ########
                reset_base_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera]
                reset_wrist_frames = [torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera]
                reset_subgoal_grounded = subgoal_grounded
                # ######## 视频保存变量准备（reset 阶段）结束 ########

                # ######## 视频保存变量初始化开始 ########
                step = 0
                episode_success = False
                rollout_base_frames: list[np.ndarray] = []
                rollout_wrist_frames: list[np.ndarray] = []
                rollout_subgoal_grounded: list[Any] = []
                # ######## 视频保存变量初始化结束 ########

                while step < MAX_STEPS:
                    replay_key = ACTION_SPACE
                    action = dataset_resolver.get_step(replay_key, step)
                    if ACTION_SPACE == "oracle_planner":
                        action = _parse_oracle_command(action)
                    if action is None:
                        break

                    obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)

                    # 保持 evaluate.py 中的调试变量语义
                    maniskill_obs = obs_batch["maniskill_obs"]
                    front_camera = obs_batch["front_camera"]
                    wrist_camera = obs_batch["wrist_camera"]
                    front_camera_depth = obs_batch["front_camera_depth"]
                    front_camera_segmentation = obs_batch["front_camera_segmentation"]
                    wrist_camera_depth = obs_batch["wrist_camera_depth"]
                    front_camera_extrinsic_opencv = obs_batch["front_camera_extrinsic_opencv"]
                    front_camera_intrinsic_opencv = obs_batch["front_camera_intrinsic_opencv"]
                    front_camera_cam2world_opengl = obs_batch["front_camera_cam2world_opengl"]
                    wrist_camera_extrinsic_opencv = obs_batch["wrist_camera_extrinsic_opencv"]
                    wrist_camera_intrinsic_opencv = obs_batch["wrist_camera_intrinsic_opencv"]
                    wrist_camera_cam2world_opengl = obs_batch["wrist_camera_cam2world_opengl"]
                    end_effector_pose_raw = obs_batch["end_effector_pose_raw"]
                    end_effector_pose = obs_batch["end_effector_pose"]
                    joint_states = obs_batch["joint_states"]
                    velocity = obs_batch["velocity"]



                    language_goal_list = info_batch["language_goal"]
                    subgoal = info_batch["subgoal"]
                    subgoal_grounded = info_batch["subgoal_grounded"]
                    available_options = info_batch["available_options"]

                    # ######## 视频保存变量准备（replay 阶段）开始 ########
                    rollout_base_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in front_camera)
                    rollout_wrist_frames.extend(torch.as_tensor(f).detach().cpu().numpy().copy() for f in wrist_camera)
                    rollout_subgoal_grounded.extend(subgoal_grounded)
                    # ######## 视频保存变量准备（replay 阶段）结束 ########

                    info = {k: v[-1] for k, v in info_batch.items()}
                    terminated = bool(terminated_batch[-1].item())
                    truncated = bool(truncated_batch[-1].item())

                    step += 1
                    if GUI_RENDER:
                        env.render()
                    if truncated:
                        print(f"[{env_id}] episode {episode} 步数超限，步 {step}。")
                        break
                    if terminated:
                        succ = info.get("success")
                        if succ == torch.tensor([True]) or (
                            isinstance(succ, torch.Tensor) and succ.item()
                        ):
                            print(f"[{env_id}] episode {episode} 成功。")
                            episode_success = True
                        elif info.get("fail", False):
                            print(f"[{env_id}] episode {episode} 失败。")
                        break

                # ######## 视频保存部分开始 ########
                save_robomme_video(
                    reset_base_frames=reset_base_frames,
                    reset_wrist_frames=reset_wrist_frames,
                    rollout_base_frames=rollout_base_frames,
                    rollout_wrist_frames=rollout_wrist_frames,
                    reset_subgoal_grounded=reset_subgoal_grounded,
                    rollout_subgoal_grounded=rollout_subgoal_grounded,
                    out_video_dir=OUT_VIDEO_DIR,
                    action_space=ACTION_SPACE,
                    env_id=env_id,
                    episode=episode,
                    episode_success=episode_success,
                )
                # ######## 视频保存部分结束 ########

            except (FileNotFoundError, KeyError) as exc:
                print(f"[{env_id}] episode {episode} 数据缺失，跳过。{exc}")
                continue
            except Exception as exc:
                print(f"[{env_id}] episode {episode} 回放异常，跳过。{exc}")
                continue
            finally:
                if dataset_resolver is not None:
                    dataset_resolver.close()
                if env is not None:
                    env.close()


if __name__ == "__main__":
    main()
