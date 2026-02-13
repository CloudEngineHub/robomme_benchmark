# -*- coding: utf-8 -*-
# Script function: Unified evaluation entry point, supporting 4 action spaces: joint_angle / ee_pose / keypoint / oracle_planner.

import os
import sys

# Add package root, parent dir, and scripts to sys.path for direct execution (no PYTHONPATH needed)
_ROOT = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.dirname(_ROOT)
_SCRIPTS = os.path.join(_PARENT, "scripts")
for _path in (_PARENT, _ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

# Robomme environment and tools (consistent import with existing evaluation scripts)
from robomme.robomme_env import *
from robomme.robomme_env.util import *
from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
)

# Only enable one ACTION_SPACE; others are commented out for manual switching
ACTION_SPACE = "joint_angle"
#ACTION_SPACE = "ee_pose"
#ACTION_SPACE = "keypoint"
#ACTION_SPACE = "oracle_planner"

GUI_RENDER = False
MAX_STEPS = 3000




def _get_dummy_action(action_space):
    
    if action_space == "joint_angle":
        return np.array(
            [0.0, 0.0, 0.0, -1.5707964, 0.0, 1.5707964, 0.7853982, 1.0],
            dtype=np.float32,
        )
    if action_space == "ee_pose":
        return np.array(
            [
                -6.0499899e-02,  # pose x
                -2.8136521e-08,  # pose y
                5.2110010e-01,   # pose z
                0.0,             # roll
                0.0,             # pitch
                0.0,             # yaw
                1.0,             # gripper
            ],
            dtype=np.float32,
        )
    if action_space == "keypoint":
        return np.array(
            [-0.120827354, 0.17769682, 0.15, 0.0, 0.972572, 0.23260213, 0.0, 1.0],
            dtype=np.float32,
        )
    if action_space == "oracle_planner":
        return {
            "action": "pick up the cube",
            "point": [0, 0],
        }
    raise ValueError(f"Unsupported ACTION_SPACE: {action_space}")


def main():
    env_id_list = BenchmarkEnvBuilder.get_task_list()
    print(f"Running envs: {env_id_list}")
    print(f"Using action_space: {ACTION_SPACE}")

    for env_id in env_id_list:
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="train",
            action_space=ACTION_SPACE,
            gui_render=GUI_RENDER,
        )
        episode_count = env_builder.get_episode_num()
        print(f"[{env_id}] episode_count from metadata: {episode_count}")

        for episode in range(episode_count):
            env, seed, difficulty = env_builder.make_env_for_episode(episode)
            # obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.reset()
            obs_batch, info_batch = env.reset()

            # Keep debug variable semantics from original 4 evaluation scripts
            maniskill_obs = obs_batch["maniskill_obs"]
            front_camera = obs_batch["front_camera"]
            wrist_camera = obs_batch["wrist_camera"]
            front_camera_depth = obs_batch["front_camera_depth"]
            wrist_camera_depth = obs_batch["wrist_camera_depth"]
            end_effector_pose = obs_batch["end_effector_pose"]
            joint_states = obs_batch["joint_states"]
            velocity = obs_batch["velocity"]
            language_goal_list = info_batch["language_goal"]

            language_goal = language_goal_list[0] if language_goal_list else None
            subgoal = info_batch["subgoal"]
            subgoal_grounded = info_batch["subgoal_grounded"]
            available_options = info_batch["available_options"]
            front_camera_extrinsic_opencv = info_batch["front_camera_extrinsic_opencv"]
            front_camera_intrinsic_opencv = info_batch["front_camera_intrinsic_opencv"]
            wrist_camera_extrinsic_opencv = info_batch["wrist_camera_extrinsic_opencv"]
            wrist_camera_intrinsic_opencv = info_batch["wrist_camera_intrinsic_opencv"]

         
            info ={k: v[-1] for k, v in info_batch.items()}
            # terminated = bool(terminated_batch[-1].item())
            # truncated = bool(truncated_batch[-1].item())

            episode_success = False
            step = 0
            while step < MAX_STEPS:
                dummy_action = _get_dummy_action(ACTION_SPACE)
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(dummy_action)
                print("dummy_action: ", dummy_action)

                # Keep debug variable semantics from original 4 evaluation scripts
                maniskill_obs = obs_batch["maniskill_obs"]
                front_camera = obs_batch["front_camera"]
                wrist_camera = obs_batch["wrist_camera"]
                front_camera_depth = obs_batch["front_camera_depth"]
                wrist_camera_depth = obs_batch["wrist_camera_depth"]
                end_effector_pose = obs_batch["end_effector_pose"]
                joint_states = obs_batch["joint_states"]
                velocity = obs_batch["velocity"]

                language_goal_list = info_batch["language_goal"]
                subgoal = info_batch["subgoal"]
                subgoal_grounded = info_batch["subgoal_grounded"]
                available_options = info_batch["available_options"]
                front_camera_extrinsic_opencv = info_batch["front_camera_extrinsic_opencv"]
                front_camera_intrinsic_opencv = info_batch["front_camera_intrinsic_opencv"]
                wrist_camera_extrinsic_opencv = info_batch["wrist_camera_extrinsic_opencv"]
                wrist_camera_intrinsic_opencv = info_batch["wrist_camera_intrinsic_opencv"]

         
                info ={k: v[-1] for k, v in info_batch.items()}
                terminated = bool(terminated_batch[-1].item()) 
                truncated = bool(truncated_batch[-1].item()) 

                step += 1
                if GUI_RENDER:
                    env.render()
                if truncated:
                    print(f"[{env_id}] episode {episode} steps exceeded, step {step}.")
                    break
                if terminated:
                    succ = info.get("success")
                    if succ == torch.tensor([True]) or (
                        isinstance(succ, torch.Tensor) and succ.item()
                    ):
                        print(f"[{env_id}] episode {episode} success.")
                        episode_success = True
                    elif info.get("fail", False):
                        print(f"[{env_id}] episode {episode} failed.")
                    break


            env.close()


if __name__ == "__main__":
    main()
