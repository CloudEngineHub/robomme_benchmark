"""
评估数据集关键点V2版本脚本

该脚本用于通过SAM2ACT Agent API评估HistoryBench数据集中的任务。
主要功能包括：
1. 从metadata文件读取episode配置
2. 创建仿真环境并初始化
3. 通过HTTP API调用SAM2ACT模型获取动作
4. 使用运动规划器执行动作
5. 评估任务完成情况

使用场景：
- 评估模型在特定任务上的表现
- 批量测试多个episode
- 记录评估结果用于分析
"""

import os
import sys
import copy
import json  # 用于JSON文件读写
import h5py  # 用于HDF5文件操作（虽然本脚本中未直接使用）
import numpy as np  # 数值计算库
import sapien  # SAPIEN物理仿真引擎
import requests  # HTTP请求库，用于调用API
import argparse  # 命令行参数解析
from pathlib import Path  # 路径操作
from PIL import Image  # 图片处理库

# 将父目录添加到Python路径中，以便导入项目模块
# 这样可以导入historybench等自定义模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym  # Gymnasium强化学习环境库
from gymnasium.utils.save_video import save_video  # 视频保存工具（本脚本中未使用）

# 导入HistoryBench相关模块
from historybench.env_record_wrapper import HistoryBenchRecordWrapper, EpisodeConfigResolver
from historybench.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from historybench.HistoryBench_env import *
from historybench.HistoryBench_env.util import task_goal  # 任务目标语言描述工具
from historybench.HistoryBench_env.util import reset_panda

# 导入ManiSkill运动规划工具函数
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,  # 通过OBB（定向包围盒）计算抓取信息
    get_actor_obb,  # 获取actor的OBB
)

import torch  # PyTorch深度学习框架

# 导入自定义的运动规划器，包含失败处理机制
from planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,  # Panda机械臂运动规划求解器（带失败感知）
    FailAwarePandaStickMotionPlanningSolver,  # Panda机械臂+棍子运动规划求解器（带失败感知）
    ScrewPlanFailure,  # 螺旋运动规划失败异常
)

# 输出根目录：脚本所在目录的父目录
OUTPUT_ROOT = Path(__file__).resolve().parents[1]


class CustomDemonstrationWrapper(DemonstrationWrapper):
    def __init__(self, env, max_steps_without_demonstration, gui_render, save_video=False):
        super().__init__(env, max_steps_without_demonstration, gui_render, save_video)
        self.full_obs_history = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_success = False
        self.full_obs_history = [copy.deepcopy(obs)]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.full_obs_history.append(copy.deepcopy(obs))
        return obs, reward, terminated, truncated, info



class NumpyEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理NumPy数据类型
    
    由于标准JSON编码器不支持NumPy的数据类型（如np.integer、np.floating、np.ndarray），
    需要自定义编码器将这些类型转换为Python原生类型，以便进行JSON序列化。
    
    转换规则：
    - np.integer -> int: NumPy整数类型转换为Python整数
    - np.floating -> float: NumPy浮点数类型转换为Python浮点数
    - np.ndarray -> list: NumPy数组转换为Python列表
    """
    def default(self, o):
        # 处理NumPy整数类型
        if isinstance(o, np.integer):
            return int(o)
        # 处理NumPy浮点数类型
        elif isinstance(o, np.floating):
            return float(o)
        # 处理NumPy数组类型
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # 其他类型使用默认的JSON编码器处理
        return json.JSONEncoder.default(self, o)


def get_model_input(env, obs, timestep, lang_goal):
    """
    从环境观测数据构建模型API所需的输入字典
    
    该函数将HistoryBench环境的观测数据转换为SAM2ACT模型API期望的格式。
    主要转换包括：
    1. 相机数据（RGB、深度）- 映射到image/wrist_image, base_camera_depth/wrist_camera_depth
    2. 机器人状态（关节位置、夹爪状态、末端执行器位姿）- 映射到robot_endeffector_p/q等
    3. 相机参数（内参、外参）- 映射到base_camera_intrinsic_opencv等
    
    Args:
        env: 环境实例
        obs: 当前时刻的环境观测数据字典
        timestep: 当前时间步
        lang_goal: 语言目标描述字符串
        
    Returns:
        dict: 包含SAM2ACT API所需的所有键值的字典
    """
    
    # 从观测数据中提取传感器数据、传感器参数和额外信息
    sensor_data = obs['sensor_data']  # 传感器数据：相机RGB、深度等
    sensor_param = obs['sensor_param']  # 传感器参数：相机内参、外参等
    agent = env.unwrapped.agent
    
    # 初始化输出字典
    obs_obj = {}
    
    # ========== 相机数据映射 ==========
    # Base camera -> image, base_camera_depth
    if 'base_camera' in sensor_data:
        obs_obj['image'] = sensor_data['base_camera']['rgb'].cpu().numpy()[0]
        # 深度数据除以1000
        obs_obj['base_camera_depth'] = sensor_data['base_camera']['depth'].cpu().numpy()[0] / 1000.0
    
    # Hand camera -> wrist_image, wrist_camera_depth
    if 'hand_camera' in sensor_data:
        obs_obj['wrist_image'] = sensor_data['hand_camera']['rgb'].cpu().numpy()[0]
        # 深度数据除以1000
        obs_obj['wrist_camera_depth'] = sensor_data['hand_camera']['depth'].cpu().numpy()[0] / 1000.0
    
    # ========== 机器人状态提取 ==========
    # 获取关节位置
    qpos = obs['agent']['qpos'].cpu().numpy()[0]
    
    # 提取机械臂关节位置（前7个）
    if len(qpos) >= 7:
        obs_obj['joint_positions'] = qpos[:7]
        
    # 提取夹爪关节位置（第8和第9个）
    if len(qpos) >= 9:
        obs_obj['gripper_joint_positions'] = qpos[7:9]
    
    # 计算夹爪开合状态
    gripper_width = np.sum(obs_obj.get('gripper_joint_positions', [0, 0]))
    obs_obj['gripper_open'] = 1.0 if gripper_width > 0.002 else 0.0
    
    # 末端执行器位姿
    position = agent.tcp.pose.p.cpu().numpy().flatten()
    # SAPIEN returns quaternion as [w, x, y, z], but model expects [x, y, z, w]
    quaternion_wxyz = agent.tcp.pose.q.cpu().numpy().flatten()
    quaternion_xyzw = np.array([
        quaternion_wxyz[1], # x
        quaternion_wxyz[2], # y
        quaternion_wxyz[3], # z
        quaternion_wxyz[0]  # w
    ])
    obs_obj['robot_endeffector_p'] = position
    obs_obj['robot_endeffector_q'] = quaternion_xyzw
    
    # 忽略碰撞标志
    obs_obj['ignore_collisions'] = 1
    
    # ========== 相机参数映射 ==========
    # 前置相机参数
    if 'base_camera' in sensor_param:
        obs_obj['base_camera_intrinsic_opencv'] = sensor_param['base_camera']['intrinsic_cv'].cpu().numpy()[0]
        obs_obj['base_camera_extrinsic_opencv'] = sensor_param['base_camera']['extrinsic_cv'].cpu().numpy()[0]
        
    # 腕部相机参数
    if 'hand_camera' in sensor_param:
        obs_obj['wrist_camera_intrinsic_opencv'] = sensor_param['hand_camera']['intrinsic_cv'].cpu().numpy()[0]
        obs_obj['wrist_camera_extrinsic_opencv'] = sensor_param['hand_camera']['extrinsic_cv'].cpu().numpy()[0]
        
    # 保留misc字典（如果需要兼容性，虽然内容已经提取到了顶层）
    obs_obj['misc'] = {}
    
    return obs_obj


def read_metadata(metadata_path):
    """
    从metadata JSON文件读取所有episode配置信息
    
    该函数读取HistoryBench数据集的metadata文件，该文件包含了所有需要评估的episode的配置信息。
    每个episode记录通常包含任务类型、episode编号、随机种子、难度等级等信息。
    
    Args:
        metadata_path: metadata JSON文件的路径（字符串或Path对象）
            文件格式示例：
            {
                "records": [
                    {
                        "task": "BinFill",
                        "episode": 0,
                        "seed": 42,
                        "difficulty": "easy"
                    },
                    ...
                ]
            }
        
    Returns:
        list: 包含所有episode记录的列表
            每个记录是一个字典，通常包含以下键：
            - task: 任务类型（如"BinFill"）
            - episode: episode编号（整数）
            - seed: 随机种子（整数，可选）
            - difficulty: 难度等级（字符串，如"easy"、"medium"、"hard"，可选）
            如果文件不存在或读取失败，返回空列表
    """
    # 检查文件是否存在
    if not Path(metadata_path).exists():
        print(f"Metadata file not found: {metadata_path}")
        return []
    
    # 读取JSON文件
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        # 从metadata中提取records字段，如果不存在则返回空列表
        episode_records = metadata.get('records', [])
        return episode_records


def main():
    """
    主函数：使用SAM2ACT Agent API运行仿真评估
    
    该函数是脚本的入口点，主要执行以下步骤：
    1. 解析命令行参数（API地址、最大步数等）
    2. 遍历指定的环境任务列表
    3. 读取每个任务的metadata文件，获取所有episode配置
    4. 对每个episode：
       a. 创建并初始化环境
       b. 初始化运动规划器
       c. 执行评估循环（调用API获取动作并执行）
       d. 评估任务完成情况
    5. 输出评估结果
    
    工作流程：
    - 每个episode最多执行max_steps步
    - 每一步：获取观测 -> 调用API获取动作 -> 执行动作 -> 检查是否成功
    - 如果任务成功完成，提前结束episode
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description='Run evaluation using SAM2ACT Agent API')
    # API服务器地址，默认本地8000端口
    parser.add_argument('--api_url', type=str, default='http://141.212.48.176:8001', help='API URL')
    # 每个episode的最大执行步数，默认25步
    parser.add_argument('--max_steps', type=int, default=40, help='Max steps per episode')
    args = parser.parse_args()
    
    # 提取参数值
    api_url = args.api_url  # API服务器URL
    max_steps = args.max_steps  # 每个episode的最大步数
    
    # ========== 配置参数 ==========
    # 要评估的环境任务列表（可以包含多个任务，如["BinFill", "PatternLock"]）
    env_id_list =[
#     "PickXtimes",
#     "StopCube",
#     "SwingXtimes",
   # "BinFill",

     "VideoUnmaskSwap",
    "VideoUnmask",
#     "ButtonUnmaskSwap",
#     "ButtonUnmask",

     "VideoRepick",
    "VideoPlaceButton",
    #"VideoPlaceOrder",
    #     "PickHighlight",

#      "InsertPeg",expo
#      'MoveCube',
#     "PatternLock",
#    "RouteStick"
    ]

    #env_id_list = ["BinFill"]
    # 数据集根目录路径，包含metadata文件和episode数据
    dataset_root = Path("/data/hongzefu/historybench-v5.7.6-sam2act7-full-dataset2-annotate/dataset_json")
    
    # ========== 遍历所有环境任务 ==========
    for env_id in env_id_list:
        # 构建metadata文件路径
        # 格式：record_dataset_{环境ID}_metadata.json
        metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
        
        # 读取metadata文件，获取所有episode配置
        episode_records = read_metadata(metadata_path)
        if not episode_records:
            print(f"No episode records found for {env_id}; skipping")
            continue  # 如果没有episode记录，跳过该环境
            
        print(f"Found {len(episode_records)} episodes for {env_id}")
        
        # ========== 初始化Episode配置解析器 ==========
        # EpisodeConfigResolver用于根据episode配置创建对应的环境实例
        resolver = EpisodeConfigResolver(
            env_id=env_id,  # 环境ID（任务类型）
            dataset=None,  # 数据集对象（None表示从metadata文件读取）
            metadata_path=metadata_path,  # metadata文件路径
            render_mode="rgb_array",  # 渲染模式："rgb_array"表示无头渲染（不创建窗口）
            gui_render=False,  # 是否启用GUI渲染（False表示不显示图形界面）
            max_steps_without_demonstration=200,  # 没有演示时的最大步数（用于安全限制）
            save_video=True,  # 是否保存视频
        )
        
        # ========== 遍历所有episode ==========
        for episode_record in episode_records:
            # 从episode记录中提取信息
            episode = episode_record['episode']  # episode编号
            seed = episode_record.get('seed')  # 随机种子（可选）
            difficulty = episode_record.get('difficulty')  # 难度等级（可选）

            #only run episode 0-10
            if episode > 50:
                continue    
            
            print(f"--- Running simulation for episode:{episode}, env: {env_id}, seed: {seed}, difficulty: {difficulty} ---")
            
            # ========== 创建环境并重置 (Manual with Custom Wrapper) ==========
            seed, difficulty_hint, episode_dataset = resolver.resolve_episode(episode)
            resolved_seed = seed
            resolved_difficulty = difficulty_hint
            
            env_kwargs = dict(
                obs_mode="rgb+depth+segmentation",
                control_mode="pd_joint_pos",
                render_mode=resolver.render_mode,
                reward_mode="dense",
                max_episode_steps=99999,
            )
            if seed is not None:
                env_kwargs["HistoryBench_seed"] = seed
            if difficulty_hint:
                env_kwargs["HistoryBench_difficulty"] = difficulty_hint
            seed_desc = seed if seed is not None else "default"
            difficulty_str = f", difficulty={difficulty_hint}" if difficulty_hint else ""
            print(f"[{env_id}] Episode {episode}: seed={seed_desc}{difficulty_str}")
            
            env = gym.make(env_id, **env_kwargs)
            env = CustomDemonstrationWrapper(
                env,
                max_steps_without_demonstration=resolver.max_steps_without_demonstration,
                gui_render=resolver.gui_render,
                save_video=False,
            )
            
            obs, info = env.reset()
            
            # ========== 初始化运动规划器 (Moved Up) ==========
            if env_id in ("PatternLock", "RouteStick"):
                planner = FailAwarePandaStickMotionPlanningSolver(
                    env, debug=False, vis=False,
                    base_pose=env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=False, print_env_info=False, joint_vel_limits=0.3,
                )
            else:
                planner = FailAwarePandaArmMotionPlanningSolver(
                    env, debug=False, vis=False,
                    base_pose=env.unwrapped.agent.robot.pose,
                    visualize_target_grasp_pose=True, print_env_info=False,
                )

            # ========== Manual Demonstration Execution ==========
            print("Executing manual demonstration...")
            tasks = getattr(env, 'task_list', [])
            demonstration_tasks = [task for task in tasks if task.get("demonstration", False)]
            
            # 获取语言目标描述（用于模型查询）
            lang_goal = task_goal.get_language_goal(env, env_id)
            print(f"Language Goal: {lang_goal}")
            
            # 在执行demonstration之前，先调用API重置记忆
            try:
                reset_response = requests.post(f"{api_url}/reset_memory")
                if reset_response.status_code == 200:
                    print(f"Memory reset successful")
                else:
                    print(f"Memory reset failed: {reset_response.text}")
            except Exception as e:
                print(f"Error calling reset_memory: {e}")
            
            demo_step = 0
            for idx, task_entry in enumerate(demonstration_tasks):
                env.unwrapped.demonstration_record_traj = True
                task_name = task_entry.get("name", f"Task {idx}")
                print(f"Executing task {idx+1}/{len(demonstration_tasks)}: {task_name}")
                solve_callable = task_entry.get("solve")
                if callable(solve_callable):
                    evaluation = env.evaluate(solve_complete_eval=True)
                    
                    # 执行预设的demonstration动作
                    solve_callable(env, planner)
                    demo_step += 1
                    
                    evaluation = env.evaluate(solve_complete_eval=True)
            
            env.unwrapped.demonstration_record_traj = False
            
            # === Post-Demonstration Action (from reset) ===
            if env.unwrapped.spec.id=="PatternLock" or env.unwrapped.spec.id == "RouteStick":
                 gripper="stick"
            else:
                 gripper=None
            if env.unwrapped.spec.id=="PatternLock" or env.unwrapped.spec.id == "RouteStick": 
                action=env.unwrapped.swing_qpos
            else:
                action=reset_panda.get_reset_panda_param("action",gripper=gripper)
            
            obs, _, _, _, info = env.step(action)

            language_goal_demo = task_goal.get_language_goal(env, env.unwrapped.spec.id)
            
            env.demonstration_data = {
                'frames': env.frames,
                'wrist_frames': env.wrist_frames,
                'actions': env.actions,
                'states': env.states,
                'velocity': env.velocity,
                'subgoal_history': env.subgoal,
                'subgoal_grounded_history': env.subgoal_grounded,
                'language goal': language_goal_demo,
            }
            
            # ========== 获取语言目标描述 ==========
            # 从任务配置中获取自然语言描述的任务目标
            # 例如："将红色方块放入蓝色盒子中"
            # 注意：lang_goal 已经在 demonstration execution 部分获取过了，这里再次获取以确保一致性
            lang_goal = task_goal.get_language_goal(env, env_id)
            print(f"Language Goal: {lang_goal}")

            # ========== Post-Demonstration Model Update ==========
            # 从 env.full_obs_history 中均匀采样 (间隔 100) 并更新模型
            sampled_history = env.full_obs_history[::20]
            print(f"Feeding {len(sampled_history)} sampled frames to model (interval 20)...")

            # ========== 保存采样的历史图片 ==========
            try:
                # 路径: /data/hongzefu/dataset_generate/replay_videos/<env_id>/<episode>
                images_output_dir = Path("/data/hongzefu/dataset_generate/replay_videos") / env_id / f"episode_{episode}"
                images_output_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Saving {len(sampled_history)} sampled images to {images_output_dir}...")
                
                for idx, hist_obs in enumerate(sampled_history):
                    if 'sensor_data' in hist_obs and 'base_camera' in hist_obs['sensor_data']:
                        # 获取RGB数据，处理Batch维度 [B, H, W, C] -> [H, W, C]
                        rgb_tensor = hist_obs['sensor_data']['base_camera']['rgb']
                        # 转换为numpy，取第一个元素
                        rgb_np = rgb_tensor.cpu().numpy()[0]
                        
                        # 转换为uint8
                        if rgb_np.dtype != np.uint8:
                             if rgb_np.max() <= 1.0:
                                rgb_np = (rgb_np * 255).astype(np.uint8)
                             else:
                                rgb_np = rgb_np.astype(np.uint8)
                        
                        # 保存图片
                        img = Image.fromarray(rgb_np)
                        img.save(images_output_dir / f"{idx}.png")
            except Exception as e:
                print(f"Error saving sampled images: {e}")
            
            for idx, hist_obs in enumerate(sampled_history):
                try:
                    obs_dict = get_model_input(env, hist_obs, idx, lang_goal)
                    payload = {
                        "obs_obj": obs_dict,
                        "curr_idx": idx,
                        "lang_goal": lang_goal,
                        "episode_length": max_steps
                    }
                    response = requests.post(
                        f"{api_url}/act",
                        data=json.dumps(payload, cls=NumpyEncoder),
                        headers={'Content-Type': 'application/json'}
                    )
                    if response.status_code != 200:
                        print(f"  [History Feed {idx}] Failed: {response.status_code} - {response.text}")
                except Exception as e:
                    print(f"  [History Feed {idx}] Error: {e}")
            
            model_timestep_count = len(sampled_history)
            print(f"Model context updated. Resuming from timestep {model_timestep_count}")
            
            # ========== 执行循环 ==========
            # 在每个episode中，最多执行max_steps步
            for step in range(model_timestep_count, max_steps):
                print(f"Step {step}/{max_steps}")
                
                # ========== 准备API输入数据 ==========
                try:
                    # 将环境观测数据转换为模型API期望的格式
                    obs_dict = get_model_input(env, obs, step, lang_goal)
                    
                    # ========== 调用SAM2ACT API获取动作 ==========
                    # 构建API请求的payload（载荷）
                    payload = {
                        "obs_obj": obs_dict,  # 观测数据字典（包含相机、机器人状态等）
                        "curr_idx": step,  # 当前时间步索引
                        "lang_goal": lang_goal,  # 语言目标描述
                        "episode_length": max_steps  # episode总长度
                    }
                    
                    # 发送POST请求到API服务器
                    # 使用NumpyEncoder处理numpy类型数据，确保可以正确序列化为JSON
                    response = requests.post(
                        f"{api_url}/act",  # API端点：/act（获取动作）
                        data=json.dumps(payload, cls=NumpyEncoder),  # 将payload序列化为JSON字符串
                        headers={'Content-Type': 'application/json'}  # 设置请求头
                    )
                    
                    # 检查API响应状态
                    if response.status_code != 200:
                        print(f"API Error: {response.status_code} - {response.text}")
                        break  # API调用失败，退出循环
                        
                    # 解析API返回的JSON响应
                    result = response.json()
                    action = result.get('action')  # 提取动作数据
                    
                    # 检查是否成功获取动作
                    if action is None:
                        print("No action returned from API")
                        break  # 没有返回动作，退出循环
                        
                    # ========== 执行动作 ==========
                    # 动作格式：[x, y, z, qx, qy, qz, qw, grip, coll]
                    # 前7个元素：末端执行器目标位姿（位置x,y,z + 四元数qx,qy,qz,qw）
                    # 第8个元素：夹爪动作（>0.5表示打开，<=0.5表示闭合）
                    # 第9个元素：碰撞相关（本脚本中未使用）
                    pose_list = action[:7]  # 提取位姿部分（位置+四元数）
                    gripper_action = action[7]  # 提取夹爪动作
                    
                    # 构建SAPIEN Pose对象
                    # p: 位置向量 [x, y, z]
                    # 模型返回的四元数通常是 [x, y, z, w]
                    pos = pose_list[:3]
                    quat_xyzw = pose_list[3:7] # [qx, qy, qz, qw]
                    
                    # SAPIEN 需要 [w, x, y, z]
                    quat_wxyz = [
                        quat_xyzw[3], # w
                        quat_xyzw[0], # x
                        quat_xyzw[1], # y
                        quat_xyzw[2]  # z
                    ]
                    
                    pose = sapien.Pose(p=pos, q=quat_wxyz)
                    print(f"pose: {pos}, quat(wxyz): {quat_wxyz} gripper_action: {gripper_action}")
                    
                 

                        
                    # ========== 执行运动规划并移动到目标位姿 ==========
                    try:
                        # 使用螺旋运动（screw motion）规划并执行移动到目标位姿
                        # 螺旋运动是一种平滑的轨迹规划方法，适合机械臂运动
                        planner.move_to_pose_with_RRTStar(pose)
                        
                        # 如果动作指示需要闭合夹爪，在移动完成后闭合
                        if gripper_action <= 0.5:
                            planner.close_gripper()
                        else:
                            planner.open_gripper()
                            
                    except ScrewPlanFailure as exc:
                        # 捕获螺旋运动规划失败异常
                        # 这通常发生在目标位姿不可达或存在碰撞时
                        print(f"    Screw plan failure: {exc}")
                        # 注意：这里可以选择break（退出）或continue（继续下一步）
                        # 当前实现是继续执行，但可以考虑改为break以提高效率
                    except Exception as exc:
                        # 捕获其他执行错误
                        print(f"    Error executing action: {exc}")
                        break  # 发生错误，退出循环
                        
                    # ========== 更新观测数据 ==========
                    # 执行动作后，获取新的环境观测
                    obs = env.unwrapped.get_obs()
                    
                    # ========== 检查任务是否成功完成 ==========
                    # 评估当前状态是否满足任务目标
                    # solve_complete_eval=True表示进行完整的成功评估
                    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                    # 从评估结果中提取成功标志
                    # 如果evaluation中没有"success"键，默认返回False
                    success = evaluation.get("success", torch.tensor([False])).item()
                    # 从评估结果中提取失败标志
                    # 如果evaluation中没有"fail"键，默认返回False
                    fail = evaluation.get("fail", torch.tensor([False])).item()
                    
                    # 如果任务失败，提前结束episode
                    if fail:
                        print(f"Episode {episode} Failed!")
                        break  # 任务失败，退出循环
                    
                    # 如果任务成功完成，提前结束episode
                    if success:
                        print(f"Episode {episode} Success!")
                        break  # 任务完成，退出循环
                        
                except Exception as e:
                    # 捕获执行循环中的任何其他异常
                    print(f"Error in step loop: {e}")
                    import traceback
                    traceback.print_exc()  # 打印完整的错误堆栈信息，便于调试
                    break  # 发生异常，退出循环
            
            # ========== 最终评估 ==========
            # 无论episode是否提前结束，都进行最终评估
            # 这可以确保即使任务未成功完成，也能获得完整的评估结果
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            print(f"Final evaluation for episode {episode}: {evaluation}")

            # ========== 保存视频 ==========
            try:
                # 检查是否存在frames数据
                if hasattr(env, "frames") and len(env.frames) > 0:
                    video_output_dir = "/data/hongzefu/dataset_generate/replay_videos"
                    os.makedirs(video_output_dir, exist_ok=True)
                    
                    # 获取成功/失败状态
                    is_success = evaluation.get("success", torch.tensor([False])).item()
                    result_status = "success" if is_success else "fail"
                    
                    # 处理文件名中的特殊字符
                    safe_lang_goal = lang_goal.replace(" ", "_").replace("/", "-")
                    
                    # 构建文件名: ep几 success/fail envid seed几 languagegoal
                    video_filename = f"ep{episode}_{result_status}_{env_id}_seed{seed}_{safe_lang_goal}"
                    
                    print(f"Saving video to {video_output_dir}/{video_filename}.mp4")
                    
                    # 使用gymnasium的save_video函数保存视频
                    save_video(
                        env.frames,
                        video_folder=video_output_dir,
                        fps=30,
                        name_prefix=video_filename,
                        episode_index=0
                    )
                    
                    # save_video会自动添加-episode-0后缀，将其重命名为目标文件名
                    generated_path = os.path.join(video_output_dir, f"{video_filename}-episode-0.mp4")
                    final_path = os.path.join(video_output_dir, f"{video_filename}.mp4")
                    
                    if os.path.exists(generated_path):
                        os.rename(generated_path, final_path)
                        print(f"Video renamed to {final_path}")
                else:
                    print("No frames captured, skipping video save.")
            except Exception as e:
                print(f"Error saving video: {e}")
            
            # ========== 清理资源 ==========
            # 关闭环境，释放资源（包括图形界面、物理引擎等）
            env.close()
            print(f"--- Finished Running simulation for episode:{episode}, env: {env_id} ---")


# ========== 脚本入口 ==========
# 当脚本被直接运行时（而非作为模块导入），执行main函数
if __name__ == "__main__":
    main()
