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
import json  # 用于JSON文件读写
import h5py  # 用于HDF5文件操作（虽然本脚本中未直接使用）
import numpy as np  # 数值计算库
import sapien  # SAPIEN物理仿真引擎
import requests  # HTTP请求库，用于调用API
import argparse  # 命令行参数解析
from pathlib import Path  # 路径操作
from concurrent.futures import ProcessPoolExecutor, as_completed  # 并行处理
import multiprocessing
from multiprocessing import Manager

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
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_success = False
        return obs, info



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


def evaluate_single_episode(episode_record, env_id, api_url_queue, max_steps, metadata_path, gpu_id=0):
    """
    评估单个Episode。
    gpu_id: 本进程使用的 GPU 编号（0 或 1），用于交替使用 GPU 0/1。
    """
    # 在进程内尽早限制可见 GPU，实现 GPU 0/1 交替运行
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    api_url = None
    try:
        # 获取一个可用的API URL
        api_url = api_url_queue.get()

        # 从episode记录中提取信息
        episode = episode_record['episode']  # episode编号
        seed = episode_record.get('seed')  # 随机种子（可选）
        difficulty = episode_record.get('difficulty')  # 难度等级（可选）

        print(f"[{env_id}] --- Running simulation for episode:{episode}, seed: {seed}, difficulty: {difficulty} with API: {api_url} [GPU {gpu_id}] ---")
        
        # ========== 初始化Episode配置解析器 ==========
        resolver = EpisodeConfigResolver(
            env_id=env_id,
            dataset=None,
            metadata_path=metadata_path,
            render_mode="rgb_array",
            gui_render=False,
            max_steps_without_demonstration=200,
            save_video=True,
        )

        # ========== 创建环境并重置 (Manual with Custom Wrapper) ==========
        seed, difficulty_hint, episode_dataset = resolver.resolve_episode(episode)
        
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
            save_video=resolver.save_video,
        )
        
        obs, info = env.reset()
        
        # ========== 初始化运动规划器 ==========
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
        print(f"[{env_id}] Executing manual demonstration...")
        tasks = getattr(env, 'task_list', [])
        demonstration_tasks = [task for task in tasks if task.get("demonstration", False)]
        
        # 获取语言目标描述（用于模型查询）
        lang_goal = task_goal.get_language_goal(env, env_id)
        
        # 在执行demonstration之前，先调用API重置记忆
        try:
            reset_response = requests.post(f"{api_url}/reset_memory")
            if reset_response.status_code == 200:
                print(f"[{env_id}] Memory reset successful")
            else:
                print(f"[{env_id}] Memory reset failed: {reset_response.text}")
        except Exception as e:
            print(f"[{env_id}] Error calling reset_memory: {e}")
        
        demo_step = 0
        for idx, task_entry in enumerate(demonstration_tasks):
            env.unwrapped.demonstration_record_traj = True
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"[{env_id}] Executing task {idx+1}/{len(demonstration_tasks)}: {task_name}")
            solve_callable = task_entry.get("solve")
            if callable(solve_callable):
                evaluation = env.evaluate(solve_complete_eval=True)
                
                # 在执行demonstration之前，获取当前观测并query model
                current_obs = env.unwrapped.get_obs()
                try:
                    obs_dict = get_model_input(env, current_obs, demo_step, lang_goal)
                    payload = {
                        "obs_obj": obs_dict,
                        "curr_idx": demo_step,
                        "lang_goal": lang_goal,
                        "episode_length": max_steps
                    }
                    response = requests.post(
                        f"{api_url}/act",
                        data=json.dumps(payload, cls=NumpyEncoder),
                        headers={'Content-Type': 'application/json'}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        model_action = result.get('action')
                        print(f"[{env_id}]   [Demo Step {demo_step}] Model queried (action not executed, using preset action)")
                    else:
                        print(f"[{env_id}]   [Demo Step {demo_step}] Model query failed: {response.status_code} - {response.text}")
                except Exception as e:
                    print(f"[{env_id}]   [Demo Step {demo_step}] Error querying model: {e}")
                
                # 执行预设的demonstration动作
                solve_callable(env, planner)
                demo_step += 1
                
                # 在执行demonstration之后，再次获取观测并query model
                current_obs = env.unwrapped.get_obs()
                try:
                    obs_dict = get_model_input(env, current_obs, demo_step, lang_goal)
                    payload = {
                        "obs_obj": obs_dict,
                        "curr_idx": demo_step,
                        "lang_goal": lang_goal,
                        "episode_length": max_steps
                    }
                    response = requests.post(
                        f"{api_url}/act",
                        data=json.dumps(payload, cls=NumpyEncoder),
                        headers={'Content-Type': 'application/json'}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        model_action = result.get('action')
                        print(f"[{env_id}]   [Demo Step {demo_step}] Model queried (action not executed, using preset action)")
                    else:
                        print(f"[{env_id}]   [Demo Step {demo_step}] Model query failed: {response.status_code} - {response.text}")
                except Exception as e:
                    print(f"[{env_id}]   [Demo Step {demo_step}] Error querying model: {e}")
                
                evaluation = env.evaluate(solve_complete_eval=True)
        
        env.unwrapped.demonstration_record_traj = False
        
        # === Post-Demonstration Action (from reset) ===
        if env_id == "PatternLock" or env_id == "RouteStick":
                gripper="stick"
        else:
                gripper=None
        if env_id == "PatternLock" or env_id == "RouteStick": 
            action=env.unwrapped.swing_qpos
        else:
            action=reset_panda.get_reset_panda_param("action",gripper=gripper)
        
        obs, _, _, _, info = env.step(action)

        language_goal_demo = task_goal.get_language_goal(env, env_id)
        
        env.demonstration_data = {
            'frames': env.frames,
            'wrist_frames': env.wrist_frames,
            'actions': env.actions,
            'states': env.states,
            'velocity': env.velocity,
            'subgoal': env.subgoal,
            'subgoal_grounded': env.subgoal_grounded,
            'language goal': language_goal_demo,
        }
        
        # ========== 获取语言目标描述 ==========
        lang_goal = task_goal.get_language_goal(env, env_id)
        print(f"[{env_id}] Language Goal: {lang_goal}")
        
        # ========== 执行循环 ==========
        for step in range(max_steps):
            print(f"[{env_id}] Episode {episode} Step {step}/{max_steps}")
            
            # ========== 准备API输入数据 ==========
            try:
                obs_dict = get_model_input(env, obs, step, lang_goal)
                
                # ========== 调用SAM2ACT API获取动作 ==========
                payload = {
                    "obs_obj": obs_dict,
                    "curr_idx": step,
                    "lang_goal": lang_goal,
                    "episode_length": max_steps
                }
                
                response = requests.post(
                    f"{api_url}/act",
                    data=json.dumps(payload, cls=NumpyEncoder),
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code != 200:
                    print(f"[{env_id}] API Error: {response.status_code} - {response.text}")
                    break
                    
                result = response.json()
                action = result.get('action')
                
                if action is None:
                    print(f"[{env_id}] No action returned from API")
                    break
                    
                # ========== 执行动作 ==========
                pose_list = action[:7]
                gripper_action = action[7]
                
                pos = pose_list[:3]
                quat_xyzw = pose_list[3:7] # [qx, qy, qz, qw]
                
                quat_wxyz = [
                    quat_xyzw[3], # w
                    quat_xyzw[0], # x
                    quat_xyzw[1], # y
                    quat_xyzw[2]  # z
                ]
                
                pose = sapien.Pose(p=pos, q=quat_wxyz)
                # print(f"[{env_id}] pose: {pos}, quat(wxyz): {quat_wxyz} gripper_action: {gripper_action}")
                
                # ========== 执行运动规划并移动到目标位姿 ==========
                try:
                    planner.move_to_pose_with_RRTStar(pose)
                    
                    if gripper_action <= 0.5:
                        planner.close_gripper()
                    else:
                        planner.open_gripper()
                        
                except ScrewPlanFailure as exc:
                    print(f"[{env_id}]     Screw plan failure: {exc}")
                except Exception as exc:
                    print(f"[{env_id}]     Error executing action: {exc}")
                    break
                    
                obs = env.unwrapped.get_obs()
                
                # ========== 检查任务是否成功完成 ==========
                evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
                success = evaluation.get("success", torch.tensor([False])).item()
                fail = evaluation.get("fail", torch.tensor([False])).item()
                
                if fail:
                    print(f"[{env_id}] Episode {episode} Failed!")
                    break
                
                if success:
                    print(f"[{env_id}] Episode {episode} Success!")
                    break
                    
            except Exception as e:
                print(f"[{env_id}] Error in step loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # ========== 最终评估 ==========
        evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
        print(f"[{env_id}] Final evaluation for episode {episode}: {evaluation}")
        
        env.close()
        print(f"[{env_id}] --- Finished Running simulation for episode:{episode} ---")
        return {"env_id": env_id, "episode": episode, "status": "completed"}
        
    except Exception as e:
        print(f"[{env_id}] Error in evaluate_single_episode {episode_record.get('episode')}: {e}")
        import traceback
        traceback.print_exc()
        return {"env_id": env_id, "episode": episode_record.get('episode'), "status": "failed", "error": str(e)}
        
    finally:
        # 归还API URL
        if api_url:
            api_url_queue.put(api_url)


def process_env_parallel(env_id, api_url_queue, max_steps, dataset_root, max_episodes_per_env):
    """
    并行处理单个环境的所有episodes
    """
    print(f"[{env_id}] Starting parallel evaluation...")
    
    metadata_path = dataset_root / f"record_dataset_{env_id}_metadata.json"
    episode_records = read_metadata(metadata_path)
    
    if not episode_records:
        print(f"[{env_id}] No episode records found; skipping")
        return {"env_id": env_id, "status": "skipped", "reason": "no_episodes"}
        
    print(f"[{env_id}] Found {len(episode_records)} episodes")
    
    # 过滤掉不需要的episodes (比如只跑前20个)；episode 为 0-based，故用 < 保证恰好 max_episodes_per_env 个
    filtered_records = [r for r in episode_records if r['episode'] < max_episodes_per_env]
    print(f"[{env_id}] Processing {len(filtered_records)} episodes after filtering (max_episodes_per_env={max_episodes_per_env})")
    
    results = []
    # 使用ProcessPoolExecutor并行执行
    # 注意：这里的workers数量取决于有多少个可用的API端口，但我们可以设置得大一点，
    # 因为受限于api_url_queue.get()，实际并发数会被API端口数限制
    num_workers = min(len(filtered_records), 32) # 设置一个合理的上限
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, record in enumerate(filtered_records):
            gpu_id = idx % 2  # GPU 0 与 GPU 1 交替
            future = executor.submit(
                evaluate_single_episode,
                record, env_id, api_url_queue, max_steps, metadata_path, gpu_id,
            )
            futures.append(future)
            
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[{env_id}] Exception in worker: {e}")
                
    return {"env_id": env_id, "status": "completed", "episodes_processed": len(results)}


def main():
    """
    主函数：使用SAM2ACT Agent API并行运行多个环境的仿真评估
    """
    parser = argparse.ArgumentParser(description='Run evaluation using SAM2ACT Agent API with parallel processing')
    parser.add_argument('--base_url', type=str, default='http://141.212.48.176', help='Base API Host URL (without port)')
    parser.add_argument('--start_port', type=int, default=8001, help='Starting port number')
    parser.add_argument('--num_ports', type=int, default=8, help='Number of API ports available')
    parser.add_argument('--max_steps', type=int, default=40, help='Max steps per episode')
    parser.add_argument('--max_episodes_per_env', type=int, default=50, help='Max number of episodes to run per environment')
    args = parser.parse_args()
    
    base_url = args.base_url
    if base_url.endswith('/'):
        base_url = base_url[:-1]
        
    start_port = args.start_port
    num_ports = args.num_ports
    max_steps = args.max_steps
    max_episodes_per_env = args.max_episodes_per_env
    
    # 初始化端口队列
    manager = Manager()
    api_url_queue = manager.Queue()
    
    print(f"[Main] Initializing API URL queue with {num_ports} ports starting from {start_port}")
    for i in range(num_ports):
        port = start_port + i
        url = f"{base_url}:{port}"
        api_url_queue.put(url)
        print(f"  Added {url}")
        
    env_id_list =[
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",

    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",

    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",

     "InsertPeg",
     "MoveCube",
    "PatternLock",
   "RouteStick"
    ]
    
 
    dataset_root = Path("/data/hongzefu/historybench-v5.7.6-sam2act7-full-dataset2-annotate/dataset_json")
    
    # 顺序处理每个环境
    for env_id in env_id_list:
        print(f"\n[Main] Processing environment: {env_id}")
        process_env_parallel(env_id, api_url_queue, max_steps, dataset_root, max_episodes_per_env)
        
    print("\n[Main] All evaluations completed!")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # 使用spawn模式，避免fork带来的问题
    main()
