
import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs
from robomme.robomme_env import MoveCube  # Import to register envs

def main():
    env_id = "MoveCube"
    print(f"Creating environment {env_id}")
    env = gym.make(env_id, robot_uids="panda", obs_mode="rgbd")
    
    print("Resetting environment...")
    obs, _ = env.reset()
    
    print(f"Observation type: {type(obs)}")

    # Check depth type in observation
    if isinstance(obs, dict):
        if 'sensor_data' in obs and 'base_camera' in obs['sensor_data']:
            depth = obs['sensor_data']['base_camera']['depth']
            print(f"Depth data type: {depth.dtype}")
            print(f"Depth min: {depth.min()}, max: {depth.max()}")
            if isinstance(depth, torch.Tensor):
                print("Depth is a torch Tensor")
            else:
                print("Depth is numpy array")
        else:
             print("Structure 'sensor_data.base_camera' not found in observation.")
             # Try to print keys to debug
             print(f"Observation keys: {obs.keys()}")
    else:
        print(f"Observation is not a dict, it is {type(obs)}")
            
    env.close()

if __name__ == "__main__":
    main()
