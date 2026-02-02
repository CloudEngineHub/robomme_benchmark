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

import numpy as np
import gymnasium as gym
import torch
from PIL import Image

from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import *
from historybench.env_record_wrapper import (
    EpisodeConfigResolver,
    EpisodeDatasetResolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

# 数据集根目录（包含 record_dataset_*_metadata.json 与 record_dataset_*.h5）
DATASET_ROOT = "/data/hongzefu/dataset_generate"





def main():
    """
    从 DATASET_ROOT 读取数据集，回放所有 episode；
    每个 episode 仅回放非 demonstration 的动作（跳过 demonstration 步）。
    """
    gui_render = True
    max_steps = 3000

    render_mode = "human" if gui_render else "rgb_array"

    env_id_list =["VideoRepick"]

    for env_id in env_id_list:
        metadata_path = f"{DATASET_ROOT}/record_dataset_{env_id}_metadata.json"

        config_resolver = EpisodeConfigResolver(
            env_id=env_id,
            metadata_path=metadata_path,
            render_mode=render_mode,
            gui_render=gui_render,
            max_steps_without_demonstration=max_steps,
        )

      

        h5_path = f"{DATASET_ROOT}/record_dataset_{env_id}.h5"
        for episode in range(10):
            env, seed, difficulty = config_resolver.make_env_for_episode(episode)
            
            planner = PandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
            
            dataset_resolver = EpisodeDatasetResolver(
                env_id=env_id,
                episode=episode,
                dataset_path=h5_path,
            )
            obs, info = env.reset()

            # 从 obs 读取
            frames = obs.get('frames', []) if obs else []
            wrist_frames = obs.get('wrist_frames', []) if obs else []
            actions = obs.get('actions', []) if obs else []
            states = obs.get('states', []) if obs else []
            velocity = obs.get('velocity', []) if obs else []
            language_goal = obs.get('language_goal') if obs else None

            # 从 info 读取
            subgoal = info.get('subgoal_history', []) if info else []
            subgoal_grounded = info.get('subgoal_grounded_history', []) if info else []

            # 保存最后一张frame和wrist_frame 左右拼接成一张图片
            image = np.concatenate([frames[-1], wrist_frames[-1]], axis=1)
            image = Image.fromarray(image)
            image.save(f"last_frame_{env_id}_{episode}.png")

            step = 0
            while True:
                action_original = dataset_resolver.get_action(step)
                ee_p, ee_q = dataset_resolver.get_ee_pose(step)
                
                if ee_p is None or ee_q is None:
                    break
                
                # 使用 planner 的 IK 将 end-effector pose 转换为 joint angles
                # 1. 构造 goal pose (world frame)
                ee_p = ee_p.flatten()
                ee_q = ee_q.flatten()
                goal_world = np.concatenate([ee_p, ee_q])
                
                # 2. 转换到 base frame
                goal_base = planner.planner.transform_goal_to_wrt_base(goal_world)
                
                # 3. 获取当前 joint angles 作为 IK 初始值
                current_qpos = planner.robot.get_qpos().cpu().numpy()[0]
                
                # 4. 执行 IK
                ik_status, ik_solutions = planner.planner.IK(
                    goal_base,
                    current_qpos,
                )
                
                if ik_status == "Success" and len(ik_solutions) > 0:
                    qpos = ik_solutions[0]
                    # Panda arm has 7 joints, but mplib might return 9 (including gripper).
                    # Slice to take only the first 7 joints.
                    qpos = qpos[:7]

                    # 5. 构造 action
                    # 从原始 action 中获取 gripper 状态 (通常是最后一个维度)
                    gripper = action_original[-1] if action_original is not None else -1
                    
                    if planner.control_mode == "pd_joint_pos_vel":
                        qvel = np.zeros_like(qpos)
                        action = np.hstack([qpos, qvel, gripper])
                    else:
                        action = np.hstack([qpos, gripper])
                        
                    obs, reward, terminated, truncated, info = env.step(action)
                else:
                    print(f"[{env_id}] episode {episode} step {step}: IK failed ({ik_status}). Using original action.")
                    if action_original is not None:
                        obs, reward, terminated, truncated, info = env.step(action_original)
                    else:
                        break

                # 从 obs 读取
                image = obs.get('frames', [])[-1] if obs.get('frames') else None
                wrist_image = obs.get('wrist_frames', [])[-1] if obs.get('wrist_frames') else None
                last_action = obs.get('actions', [])[-1] if obs.get('actions') else None
                state = obs.get('states', [])[-1] if obs.get('states') else None
                velocity = obs.get('velocity', [])[-1] if obs.get('velocity') else None
                language_goal = obs.get('language_goal') if obs else None
                # 从 info 读取
                subgoal = info.get('subgoal_history', []) if info else []
                subgoal_grounded = info.get('subgoal_grounded_history', []) if info else []

                step += 1
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
            dataset_resolver.close()
            planner.close()
            env.close()

if __name__ == "__main__":
    main()
