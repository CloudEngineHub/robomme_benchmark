import sys
from pathlib import Path

import numpy as np
import cv2
import imageio
import os
import json
import shutil
import colorsys
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chat_api.api import *
from chat_api.prompts import *
from historybench.HistoryBench_env.util.vqa_options import get_vqa_options
from scripts.evaluate_oracle_planner_gui import EpisodeConfigResolverForOraclePlanner


# NLP 语义匹配（可选）
_NLP_MODEL = None
_ST_UTIL = None
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    print("Loading NLP Model (all-MiniLM-L6-v2)...")
    _NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    _ST_UTIL = st_util
    print("NLP Model loaded.")
except ImportError:
    print("Warning: sentence-transformers not found. NLP matching will fail.")
except Exception as e:
    print(f"Error loading NLP model: {e}")


# =============================================================================
# 辅助函数
# =============================================================================

def _generate_color_map(n=10000, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
    """生成颜色映射表用于分割可视化"""
    phi = 0.6180339887498948
    color_map = {}
    for i in range(1, n + 1):
        h = (i * phi) % 1.0
        s = s_min + (s_max - s_min) * ((i % 7) / 6)
        v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_map[i] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
    return color_map


def _sync_table_color(env, color_map):
    """同步桌面颜色为黑色"""
    seg_id_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if not isinstance(seg_id_map, dict):
        return
    for obj_id, obj in seg_id_map.items():
        if getattr(obj, "name", None) == "table-workspace":
            color_map[obj_id] = [0, 0, 0]


def _tensor_to_bool(value):
    """将 tensor 或其他类型转换为布尔值"""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _prepare_frame(frame):
    """预处理帧数据为 uint8 格式"""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        max_val = float(np.max(frame)) if frame.size else 0.0
        if max_val <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame


def _prepare_segmentation_visual(segmentation, color_map, target_hw):
    """将分割数据转换为可视化图像"""
    if segmentation is None:
        return None, None

    seg = segmentation
    if hasattr(seg, "cpu"):
        seg = seg.cpu().numpy()
    seg = np.asarray(seg)
    if seg.ndim > 2:
        seg = seg[0]
    seg_2d = seg.squeeze().astype(np.int64)

    h, w = seg_2d.shape[:2]
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(seg_2d)
    for seg_id in unique_ids:
        if seg_id <= 0:
            continue
        color = color_map.get(int(seg_id))
        if color is None:
            continue
        seg_rgb[seg_2d == seg_id] = color
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    target_h, target_w = target_hw
    if seg_bgr.shape[:2] != (target_h, target_w):
        seg_bgr = cv2.resize(seg_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return seg_bgr, seg_2d


def _fetch_segmentation(env):
    """从环境获取分割数据"""
    obs = env.unwrapped.get_obs(unflattened=True)
    return obs["sensor_data"]["base_camera"]["segmentation"]


def _build_solve_options(env, planner, selected_target, env_id):
    """构建可用的动作选项"""
    return get_vqa_options(env, planner, selected_target, env_id)


def _find_best_semantic_match(user_query, options):
    """使用 NLP 语义匹配找到最佳选项"""
    if _NLP_MODEL is None or _ST_UTIL is None:
        return -1, 0.0
    
    if not options:
        return -1, 0.0

    labels = [opt.get("label", "") for opt in options]
    query_text = str(user_query or "").strip()

    try:
        query_embedding = _NLP_MODEL.encode(query_text, convert_to_tensor=True)
        corpus_embeddings = _NLP_MODEL.encode(labels, convert_to_tensor=True)
        cos_scores = _ST_UTIL.cos_sim(query_embedding, corpus_embeddings)[0]
        best_idx = int(torch.argmax(cos_scores).item())
        best_score = float(cos_scores[best_idx].item())
    except Exception as exc:
        print(f"  [NLP] Semantic match failed ({exc}); defaulting to option 1.")
        return 0, 0.0

    print(f"  [NLP] Closest Match: '{query_text}' -> '{labels[best_idx]}' (Score: {best_score:.4f})")
    
    return best_idx, best_score


# =============================================================================
# 核心函数: step_before 和 step_after
# =============================================================================

def step_before(env, planner, env_id, color_map, use_segmentation=False):
    """
    在执行动作之前调用，获取当前环境状态和可用选项。
    
    Args:
        env: 环境对象
        planner: 规划器对象
        env_id: 环境 ID
        color_map: 颜色映射表
        use_segmentation: 是否使用分割可视化
    
    Returns:
        seg_vis: 分割可视化图像 (BGR)
        seg_raw: 原始分割数据
        base_frames: 基础相机帧列表
        wrist_frames: 腕部相机帧列表
        available_options: 可用动作选项列表
    """
    # 1. 获取帧数据
    base_frames = getattr(env, "frames", [])
    if not base_frames:
        base_frames = getattr(env.unwrapped, "frames", []) or []
        
    wrist_frames = getattr(env, "wrist_frames", [])
    if not wrist_frames:
        wrist_frames = getattr(env.unwrapped, "wrist_frames", []) or []

    # 2. 获取分割数据
    seg_data = _fetch_segmentation(env)
    
    # 3. 确定分辨率
    seg_hw = (255, 255)  # 默认
    if base_frames and len(base_frames) > 0:
        seg_hw = base_frames[-1].shape[:2]
    elif seg_data is not None:
        try:
            temp = seg_data
            if hasattr(temp, "cpu"):
                temp = temp.cpu().numpy()
            temp = np.asarray(temp)
            if temp.ndim > 2:
                temp = temp[0]
            seg_hw = temp.shape[:2]
        except Exception:
            pass

    # 4. 处理分割可视化
    seg_vis = None
    seg_raw = None

    if use_segmentation:
        seg_vis, seg_raw = _prepare_segmentation_visual(seg_data, color_map, seg_hw)
    else:
        _, seg_raw = (_prepare_segmentation_visual(seg_data, color_map, seg_hw) 
                      if seg_data is not None else (None, None))
        if base_frames:
            vis_frame = _prepare_frame(base_frames[-1])
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            if vis_frame.shape[:2] != seg_hw:
                vis_frame = cv2.resize(vis_frame, (seg_hw[1], seg_hw[0]), interpolation=cv2.INTER_LINEAR)
            seg_vis = vis_frame
    
    if seg_vis is None:
        seg_vis = np.zeros((seg_hw[0], seg_hw[1], 3), dtype=np.uint8)

    # 5. 构建可用选项
    dummy_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    raw_options = _build_solve_options(env, planner, dummy_target, env_id)
    available_options = [{"action": opt.get("label", "Unknown"), "need_parameter": bool(opt.get("available"))} 
                         for opt in raw_options]

    return seg_vis, seg_raw, base_frames, wrist_frames, available_options


def step_after(env, planner, env_id, seg_vis, seg_raw, base_frames, wrist_frames, command_dict):
    """
    在收到命令后执行动作并返回评估结果。
    
    Args:
        env: 环境对象
        planner: 规划器对象
        env_id: 环境 ID
        seg_vis: 分割可视化图像
        seg_raw: 原始分割数据
        base_frames: 基础相机帧列表
        wrist_frames: 腕部相机帧列表
        command_dict: 命令字典，包含 'action' 和 'point'
    
    Returns:
        evaluation: 评估结果字典
    """
    # 1. 构建选项
    selected_target = {"obj": None, "name": None, "seg_id": None, "click_point": None, "centroid_point": None}
    solve_options = _build_solve_options(env, planner, selected_target, env_id)
    
    target_action = command_dict.get("action")
    target_param = command_dict.get("point")

    if "action" not in command_dict:
        return None
    if target_action is None:
        return None

    # 2. 查找匹配的动作选项
    found_idx = -1
    for i, opt in enumerate(solve_options):
        if opt.get("label") == target_action or str(i + 1) == str(target_action):
            found_idx = i
            break
    
    # 3. 如果精确匹配失败，尝试语义匹配
    if found_idx == -1 and isinstance(target_action, str) and not target_action.isdigit():
        print(f"Attempting semantic match for: '{target_action}'")
        found_idx, score = _find_best_semantic_match(target_action, solve_options)
    
    if found_idx == -1:
        print(f"Error: Action '{target_action}' not found in current options.")
        return None

    # 4. 处理点击坐标，解析目标对象
    if target_param is not None and seg_raw is not None:
        cx, cy = target_param
        h, w = seg_raw.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        
        seg_id_map = getattr(env.unwrapped, "segmentation_id_map", {}) or {}
        
        # 收集可用候选对象
        candidates = []
        def _collect(item):
            if isinstance(item, (list, tuple)):
                for x in item:
                    _collect(x)
            elif isinstance(item, dict):
                for x in item.values():
                    _collect(x)
            else:
                if item:
                    candidates.append(item)
        
        avail = solve_options[found_idx].get("available")
        if avail:
            _collect(avail)
            best_cand = None
            min_dist = float('inf')
            for actor in candidates:
                target_ids = [sid for sid, obj in seg_id_map.items() if obj is actor]
                for tid in target_ids:
                    tid = int(tid)
                    mask = (seg_raw == tid)
                    if np.any(mask):
                        ys, xs = np.nonzero(mask)
                        center_x, center_y = xs.mean(), ys.mean()
                        dist = (center_x - cx) ** 2 + (center_y - cy) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_cand = {
                                "obj": actor,
                                "name": getattr(actor, "name", f"id_{tid}"),
                                "seg_id": tid,
                                "click_point": (int(cx), int(cy)),
                                "centroid_point": (int(center_x), int(center_y))
                            }
            if best_cand:
                selected_target.update(best_cand)
            else:
                selected_target["click_point"] = (int(cx), int(cy))
        else:
            selected_target["click_point"] = (int(cx), int(cy))

    # 5. 执行动作
    print(f"Executing Option: {found_idx + 1} - {solve_options[found_idx].get('label')}")
    solve_options[found_idx].get("solve")()

    # 6. 评估结果
    env.unwrapped.evaluate()
    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print(f"Evaluation: {evaluation}")
    return evaluation


TASK_WITH_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

def main():    
    # Initialization Wrapper
    oracle_resolver = EpisodeConfigResolverForOraclePlanner(
        gui_render=False,
        max_steps_without_demonstration=1000
    )
    
    env_id_list = [
        # "PickXtimes",
        # "StopCube",
        # "SwingXtimes",
        #"BinFill",

        #"VideoUnmaskSwap",
        #"VideoUnmask",
        # "ButtonUnmaskSwap",
        # "ButtonUnmask",

        # "VideoRepick",
        # "VideoPlaceButton",
         "VideoPlaceOrder",
        # "PickHighlight",

        # "InsertPeg",
        # 'MoveCube',
        # "PatternLock",
        # "RouteStick"
    ]

    for env_id in env_id_list:
        num_episodes = oracle_resolver.get_num_episodes(env_id)

        for episode in range(10):
            # if episode !=2:
            #     continue

            env, planner, color_map, language_goal = oracle_resolver.initialize_episode(env_id, episode)
            model_name = "gemini-2.5-flash"  # "gemini-2.5-pro" # "gpt-4o-mini", "gemini-er", "qwen-vl"
            success = "fail"
            save_dir = f"oracle_planning/{model_name}/{env_id}/ep{episode}"
                        
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            
            
            with open(os.path.join(save_dir, "language_goal.txt"), "w") as f:
                f.write(language_goal)
            
            if "gemini" in model_name:
                api = GeminiModel(save_dir=save_dir, task_id=env_id, model_name=model_name, task_goal=language_goal, subgoal_type="oracle_planner")
            elif "qwen" in model_name:
                api = QwenModel(save_dir=save_dir, task_id=env_id, model_name=model_name, task_goal=language_goal, subgoal_type="oracle_planner")
            else:
                api = OpenAIModel(save_dir=save_dir, task_id=env_id, model_name=model_name, task_goal=language_goal, subgoal_type="oracle_planner")


            step_idx = 0
            frame_idx = 0
            max_query_times = 10
            
            while True:
                if step_idx >= max_query_times:
                    print(f"Max query times ({max_query_times}) reached, stopping.")
                    break

                seg_vis, seg_raw, base_frames, wrist_frames, available_options = step_before(
                    env,
                    planner,
                    env_id,
                    color_map
                )
                print("num of base_frames", len(base_frames)-frame_idx)
                print("num of wrist_frames", len(wrist_frames)-frame_idx)
                print(available_options)
                
                # ------------------------ Call Gemini API ------------------------------------
            
                if step_idx == 0:
                    if env_id in TASK_WITH_DEMO:
                        if api.use_multi_images_as_video:
                            text_query = DEMO_TEXT_QUERY_multi_image.format(task_goal=language_goal)
                        else:
                            text_query = DEMO_TEXT_QUERY.format(task_goal=language_goal)
                    else:
                        text_query = IMAGE_TEXT_QUERY.format(task_goal=language_goal)
                else:
                    if api.use_multi_images_as_video:
                        text_query = VIDEO_TEXT_QUERY_multi_image.format(task_goal=language_goal)
                    else:
                        text_query = VIDEO_TEXT_QUERY.format(task_goal=language_goal)
                
                input_data = api.prepare_input_data(base_frames[frame_idx:], text_query, step_idx)

               #使用gui画出图
                # cv2.imshow("base_frames[-1]", base_frames[-1])
                # cv2.waitKey(0)
                # cv2.destroyWindow("base_frames[-1]")


                response, points = api.call(input_data)
                
                #points=[(255, 255)]#test

                if response is None:
                    print("Response is None, skipping this step")
                    break
                
                # Draw the points for debugging              
                if points and len(points) > 0:
                    anno_image = base_frames[-1].copy()
                    for point in points:
                        cv2.circle(anno_image, (point[1], point[0]), 5, (255, 255, 0), -1)
                    imageio.imwrite(os.path.join(save_dir, f"anno_step_{step_idx}_image.png"), anno_image)
                    api.add_frame_hold(anno_image)
                
                command_dict = response['subgoal']
                # TODO: will be fixed in the future
                if command_dict['point'] is not None:
                    command_dict['point'] = command_dict['point'][::-1]  
                
                print(f"\nResponse: {response}")              
                print(f"\nCommand: {command_dict}")
                
                                
                frame_idx = len(base_frames)
                step_idx += 1
                
                # ------------------------------------------------------------                
                evaluation = step_after(
                    env,
                    planner,
                    env_id,
                    seg_vis,
                    seg_raw,
                    base_frames,
                    wrist_frames,
                    command_dict
                )
                
                fail_flag = evaluation.get("fail", False)
                success_flag = evaluation.get("success", False)
                if _tensor_to_bool(fail_flag):
                    success = "fail"
                    print("Encountered failure condition; stopping task sequence.")
                    break

                if _tensor_to_bool(success_flag):
                    success = "success"
                    print("Task completed successfully.")
                    break
            
            
            if response is not None:
                api.prepare_input_data(base_frames[frame_idx:], text_query, step_idx)
            else:
                success = "api_error"
            
            api.save_conversation()
            api.save_final_video(os.path.join(os.path.dirname(save_dir), f"{success}_ep{episode}_{language_goal}.mp4"))
            api.clear_uploaded_files()
            del api
            #import pdb; pdb.set_trace()
                      
    oracle_resolver.close()
    
if __name__ == "__main__":
    main()
