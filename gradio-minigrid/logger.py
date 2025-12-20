import json
import threading
import os
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import cv2
from PIL import Image

# 线程锁，防止多用户同时写入时文件损坏
lock = threading.Lock()
# 使用基于 logger.py 文件位置的绝对路径，确保日志文件始终保存在 gradio-minigrid/data/ 目录下
BASE_DIR = Path(__file__).parent.absolute()
LOG_FILE = str(BASE_DIR / "data" / "experiment_logs.jsonl")
USER_ACTION_LOG_DIR = str(BASE_DIR / "data" / "user_action_logs")

def _get_current_attempt_index(f):
    """
    获取当前最新的 attempt 索引。
    
    Args:
        f: h5py.File 对象
    
    Returns:
        int: 当前 attempt 索引，如果没有则返回 -1
    """
    attempt_indices = []
    for key in f.keys():
        if key.startswith("attempt_"):
            try:
                idx = int(key.split("_")[1])
                attempt_indices.append(idx)
            except (ValueError, IndexError):
                pass
    
    if not attempt_indices:
        return -1
    return max(attempt_indices)

def _get_or_create_attempt(f, attempt_idx, username, env_id, episode_idx):
    """
    获取或创建指定索引的 attempt 组。
    
    Args:
        f: h5py.File 对象
        attempt_idx: attempt 索引（如果为 None，则创建新的）
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    
    Returns:
        h5py.Group: attempt 组对象
    """
    if attempt_idx is None:
        # 创建新的 attempt
        current_max = _get_current_attempt_index(f)
        attempt_idx = current_max + 1
    
    attempt_name = f"attempt_{attempt_idx}"
    
    if attempt_name not in f:
        # 创建新的 attempt 组
        attempt_group = f.create_group(attempt_name)
        
        # 创建 metadata 组
        metadata = attempt_group.create_group("metadata")
        metadata.create_dataset("username", data=username.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        metadata.create_dataset("env_id", data=env_id.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        metadata.create_dataset("episode_idx", data=np.int32(episode_idx))
        metadata.create_dataset("attempt_idx", data=np.int32(attempt_idx))
        created_at = datetime.now().isoformat()
        metadata.create_dataset("created_at", data=created_at.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        metadata.create_dataset("last_updated", data=created_at.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
        # 创建 actions 组
        attempt_group.create_group("actions")
    else:
        # 获取现有的 attempt 组
        attempt_group = f[attempt_name]
        
        # 更新最后更新时间
        if "metadata" in attempt_group:
            metadata_group = attempt_group["metadata"]
            try:
                if "last_updated" in metadata_group:
                    del metadata_group["last_updated"]
            except (KeyError, TypeError):
                pass
            metadata_group.create_dataset("last_updated", data=datetime.now().isoformat().encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
    
    return attempt_group

def _ensure_hdf5_file(username, env_id, episode_idx, create_new_attempt=False):
    """
    确保 HDF5 文件存在并初始化必要的组结构。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        create_new_attempt: 是否创建新的 attempt（用于 refresh）
    
    Returns:
        tuple: (h5py.File 对象, h5py.Group attempt_group, int attempt_idx) 或 (None, None, -1)（如果出错）
    """
    if not username or not env_id or episode_idx is None:
        return None, None, -1
    
    # 构建文件路径
    user_dir = os.path.join(USER_ACTION_LOG_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    hdf5_file = os.path.join(user_dir, f"{env_id}_{episode_idx}.h5")
    
    try:
        # 检查文件是否存在
        file_exists = os.path.exists(hdf5_file)
        
        # 以追加模式打开（如果不存在则创建）
        f = h5py.File(hdf5_file, "a")
        
        # 获取或创建 attempt
        if create_new_attempt:
            # 创建新的 attempt
            attempt_group = _get_or_create_attempt(f, None, username, env_id, episode_idx)
            attempt_idx = _get_current_attempt_index(f)
        else:
            # 获取当前最新的 attempt，如果不存在则创建 attempt_0
            current_attempt_idx = _get_current_attempt_index(f)
            if current_attempt_idx == -1:
                # 文件存在但没有 attempt，创建 attempt_0
                attempt_group = _get_or_create_attempt(f, 0, username, env_id, episode_idx)
                attempt_idx = 0
            else:
                # 使用当前最新的 attempt
                attempt_group = _get_or_create_attempt(f, current_attempt_idx, username, env_id, episode_idx)
                attempt_idx = current_attempt_idx
        
        return f, attempt_group, attempt_idx
    except Exception as e:
        print(f"Error ensuring HDF5 file {hdf5_file}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, -1

def _add_coordinate_click_to_hdf5(clicks_group, click_index, coordinates, coords_str, image_array, timestamp):
    """
    将坐标点击及其图片数组添加到 HDF5 文件的 coordinate_clicks 组中。
    
    Args:
        clicks_group: h5py.Group 对象，表示 coordinate_clicks 组
        click_index: 点击索引（用于生成唯一的 click 组名）
        coordinates: 坐标字典 {"x": x, "y": y}
        coords_str: 坐标字符串
        image_array: numpy array，形状为 [H, W, 3] 的 RGB 图片数组
        timestamp: ISO 格式的时间戳字符串
    """
    try:
        click_name = f"click_{click_index}"
        
        # 创建 click 组
        click_group = clicks_group.create_group(click_name)
        
        # 存储坐标（作为 dataset）
        click_group.create_dataset("coordinates", data=np.array([coordinates["x"], coordinates["y"]], dtype=np.int32))
        
        # 存储坐标字符串（作为 dataset）
        click_group.create_dataset("coords_str", data=coords_str.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
        # 存储图片数组（使用压缩）
        if image_array is not None:
            # 确保是 uint8 类型
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            
            # 确保是 RGB 格式 [H, W, 3]
            if len(image_array.shape) == 2:
                # 灰度图转 RGB
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA 转 RGB
                image_array = image_array[:, :, :3]
            
            # 在图片上画圈可视化点击位置
            try:
                # 确保数组在内存中是连续的，并且是副本以防副作用
                if not image_array.flags['C_CONTIGUOUS']:
                    image_array = np.ascontiguousarray(image_array)
                else:
                    image_array = image_array.copy()
                
                # 画红色圆圈: 中心点, 半径5, 颜色(255,0,0), 线宽2
                cv2.circle(image_array, (int(coordinates["x"]), int(coordinates["y"])), 5, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error drawing circle on coordinate click image: {e}")

            click_group.create_dataset(
                "image",
                data=image_array,
                compression="gzip",
                compression_opts=9,
                dtype=np.uint8
            )
            
            # 存储图片尺寸（作为 dataset）
            click_group.create_dataset("image_shape", data=np.array(image_array.shape, dtype=np.int32))
        
        # 存储时间戳
        click_group.create_dataset("timestamp", data=timestamp.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
    except Exception as e:
        print(f"Error adding coordinate click to HDF5: {e}")
        import traceback
        traceback.print_exc()

def _add_option_select_to_hdf5(option_selects_group, select_index, option_idx, option_label, timestamp):
    """
    将选项选择添加到 HDF5 文件的 option_selects 组中。
    
    Args:
        option_selects_group: h5py.Group 对象，表示 option_selects 组
        select_index: 选择索引（用于生成唯一的 select 组名）
        option_idx: 选项索引
        option_label: 选项标签
        timestamp: ISO 格式的时间戳字符串
    """
    try:
        select_name = f"select_{select_index}"
        
        # 创建 select 组
        select_group = option_selects_group.create_group(select_name)
        
        # 存储选项信息
        select_group.create_dataset("option_idx", data=np.int32(option_idx))
        if option_label is not None:
            select_group.create_dataset("option_label", data=str(option_label).encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        select_group.create_dataset("timestamp", data=timestamp.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
    except Exception as e:
        print(f"Error adding option select to HDF5: {e}")
        import traceback
        traceback.print_exc()

def _add_action_to_hdf5(attempt_group, action_index, action_data):
    """
    将操作记录添加到 HDF5 文件的 attempt 组中。
    
    新的数据格式：
    每个 action 组记录一次 execute action，包含 execute 之前所有的 click 和选择的 option。
    
    Args:
        attempt_group: h5py.Group 对象，表示 attempt 组
        action_index: 操作索引（用于生成唯一的 action 组名）
        action_data: 操作数据字典，包含：
            - option_idx: execute 时使用的选项索引（最后一次选择的）
            - option_label: execute 时使用的选项标签
            - final_coordinates: 最后执行的坐标 {"x": x, "y": y}（可选）
            - final_coords_str: 最后执行的坐标字符串（可选）
            - final_image_array: 最后执行时的图片数组（可选）
            - option_selects_before_execute: 列表，包含 execute 之前所有的选项选择
            - coordinate_clicks_before_execute: 列表，包含 execute 之前所有的坐标点击（每个元素包含 coordinates, coords_str, image_array, timestamp）
            - status: 执行状态
            - done: 是否完成
    """
    try:
        actions_group = attempt_group["actions"]
        action_name = f"action_{action_index}"
        
        # 创建 action 组
        action_group = actions_group.create_group(action_name)
        
        # 存储基本属性
        timestamp = action_data.get("timestamp", datetime.now().isoformat())
        action_group.create_dataset("timestamp", data=timestamp.encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
        # 存储 execute 时使用的选项信息（最后一次选择的）
        if "option_idx" in action_data and action_data["option_idx"] is not None:
            action_group.create_dataset("option_idx", data=np.int32(action_data["option_idx"]))
        
        if "option_label" in action_data and action_data["option_label"] is not None:
            action_group.create_dataset("option_label", data=str(action_data["option_label"]).encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
        # 记录最后执行的坐标和图片
        if "final_coordinates" in action_data and action_data["final_coordinates"] is not None:
            action_group.create_dataset("final_coordinates", data=np.array([action_data["final_coordinates"]["x"], action_data["final_coordinates"]["y"]], dtype=np.int32))
        
        if "final_coords_str" in action_data and action_data["final_coords_str"] is not None:
            action_group.create_dataset("final_coords_str", data=str(action_data["final_coords_str"]).encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
        if "final_image_array" in action_data and action_data["final_image_array"] is not None:
            image_array = action_data["final_image_array"]
            # 确保是 uint8 类型
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            
            # 确保是 RGB 格式 [H, W, 3]
            if len(image_array.shape) == 2:
                # 灰度图转 RGB
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA 转 RGB
                image_array = image_array[:, :, :3]
            
            # 在图片上画圈可视化点击位置（如果有坐标）
            if "final_coordinates" in action_data and action_data["final_coordinates"] is not None:
                try:
                    final_coords = action_data["final_coordinates"]
                    # 确保数组在内存中是连续的，并且是副本以防副作用
                    if not image_array.flags['C_CONTIGUOUS']:
                        image_array = np.ascontiguousarray(image_array)
                    else:
                        image_array = image_array.copy()
                    
                    cv2.circle(image_array, (int(final_coords["x"]), int(final_coords["y"])), 5, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Error drawing circle on action final image: {e}")

            action_group.create_dataset(
                "final_image",
                data=image_array,
                compression="gzip",
                compression_opts=9,
                dtype=np.uint8
            )
            
            # 存储图片尺寸
            action_group.create_dataset("final_image_shape", data=np.array(image_array.shape, dtype=np.int32))
        
        # 记录 execute 之前所有的选项选择
        option_selects = action_data.get("option_selects_before_execute", [])
        
        # 如果 option_selects 为空，但 execute 时使用了 option_idx，则至少记录最后一次选择的 option
        if not option_selects and action_data.get("option_idx") is not None:
            option_selects = [{
                "option_idx": action_data.get("option_idx"),
                "option_label": action_data.get("option_label"),
                "timestamp": action_data.get("timestamp", datetime.now().isoformat())
            }]
        
        if option_selects:
            # 创建 option_selects 组
            if "option_selects" not in action_group:
                action_group.create_group("option_selects")
            
            selects_group = action_group["option_selects"]
            
            for select_idx, select_data in enumerate(option_selects):
                option_idx = select_data.get("option_idx")
                option_label = select_data.get("option_label")
                timestamp = select_data.get("timestamp", datetime.now().isoformat())
                
                if option_idx is not None:
                    _add_option_select_to_hdf5(
                        selects_group,
                        select_idx,
                        option_idx,
                        option_label,
                        timestamp
                    )
        
        # 记录 execute 之前所有的坐标点击（圆圈选择点）
        coordinate_clicks = action_data.get("coordinate_clicks_before_execute", [])
        if coordinate_clicks:
            # 创建 coordinate_clicks 组
            if "coordinate_clicks" not in action_group:
                action_group.create_group("coordinate_clicks")
            
            clicks_group = action_group["coordinate_clicks"]
            for click_idx, click_data in enumerate(coordinate_clicks):
                coordinates = click_data.get("coordinates")
                coords_str = click_data.get("coords_str", "")
                image_array = click_data.get("image_array")  # numpy array
                timestamp = click_data.get("timestamp", datetime.now().isoformat())
                
                if coordinates:
                    _add_coordinate_click_to_hdf5(
                        clicks_group,
                        click_idx,
                        coordinates,
                        coords_str,
                        image_array,
                        timestamp
                    )
        
        # 记录 execute 的状态信息（可选）
        if "status" in action_data and action_data["status"] is not None:
            action_group.create_dataset("status", data=str(action_data["status"]).encode('utf-8'), dtype=h5py.string_dtype(encoding='utf-8'))
        
        if "done" in action_data:
            action_group.create_dataset("done", data=np.bool_(bool(action_data["done"])))
        
    except Exception as e:
        print(f"Error adding action to HDF5: {e}")
        import traceback
        traceback.print_exc()

def log_session(session_data):
    """
    将单个会话的数据追加写入到 JSONL 文件中。
    session_data 应该是一个字典。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # 添加写入时间戳
    session_data["logged_at"] = datetime.now().isoformat()
    
    with lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(session_data, ensure_ascii=False) + "\n")

def log_user_action_hdf5(username, env_id, episode_idx, action_data):
    """
    记录用户的详细操作到 HDF5 文件中。
    
    新的数据格式：
    每个 action 组记录一次 execute action，包含 execute 之前所有的 click 和选择的 option。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        action_data: 操作数据字典，包含：
            - option_idx: execute 时使用的选项索引（最后一次选择的）
            - option_label: execute 时使用的选项标签
            - final_coordinates: 最后执行的坐标 {"x": x, "y": y}（可选）
            - final_coords_str: 最后执行的坐标字符串（可选）
            - final_image_array: 最后执行时的图片数组（可选）
            - option_selects_before_execute: 列表，包含 execute 之前所有的选项选择
            - coordinate_clicks_before_execute: 列表，包含 execute 之前所有的坐标点击（每个元素包含 coordinates, coords_str, image_array, timestamp）
            - status: 执行状态
            - done: 是否完成
    
    文件路径: data/user_action_logs/{username}/{env_id}_{episode_idx}.h5
    文件格式: HDF5，包含 attempt_N 组，每个 attempt 包含 metadata 和 actions 组
    """
    if not username or not env_id or episode_idx is None:
        print(f"Warning: Missing required parameters for log_user_action_hdf5: username={username}, env_id={env_id}, episode_idx={episode_idx}")
        return
    
    # 添加时间戳
    action_data_with_timestamp = {
        **action_data,
        "timestamp": datetime.now().isoformat()
    }
    
    # 使用线程锁确保并发安全
    with lock:
        f, attempt_group, attempt_idx = _ensure_hdf5_file(username, env_id, episode_idx, create_new_attempt=False)
        if f is None or attempt_group is None:
            print(f"Error: Failed to open HDF5 file for {username}/{env_id}_{episode_idx}")
            return
        
        try:
            # 获取当前 attempt 的 actions 组
            actions_group = attempt_group["actions"]
            # 计算现有 action 的数量
            action_index = len(actions_group)  # type: ignore
            
            # 添加操作记录到当前 attempt
            _add_action_to_hdf5(attempt_group, action_index, action_data_with_timestamp)
            
            # 强制刷新以确保所有数据都被写入
            f.flush()
        except Exception as e:
            print(f"Error writing to HDF5 file: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if f is not None:
                try:
                    f.flush()  # 再次刷新以确保所有数据被写入
                except:
                    pass
                f.close()  # 关闭文件，这会自动保存所有更改

def has_existing_actions(username, env_id, episode_idx):
    """
    检查指定任务是否已有 actions 记录。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    
    Returns:
        bool: 如果存在 actions 则返回 True，否则返回 False
    """
    if not username or not env_id or episode_idx is None:
        return False
    
    user_dir = os.path.join(USER_ACTION_LOG_DIR, username)
    hdf5_file = os.path.join(user_dir, f"{env_id}_{episode_idx}.h5")
    
    if not os.path.exists(hdf5_file):
        return False
    
    try:
        with h5py.File(hdf5_file, "r") as f:
            current_attempt_idx = _get_current_attempt_index(f)
            if current_attempt_idx == -1:
                return False
            
            attempt_name = f"attempt_{current_attempt_idx}"
            if attempt_name not in f:
                return False
            
            attempt_group = f[attempt_name]
            if "actions" not in attempt_group:
                return False
            
            actions_group = attempt_group["actions"]
            return len(actions_group) > 0
    except Exception as e:
        print(f"Error checking existing actions: {e}")
        return False

def create_new_attempt(username, env_id, episode_idx):
    """
    为指定的任务创建新的 attempt。
    在 refresh 时调用此函数来创建新的 attempt。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
    
    Returns:
        int: 新创建的 attempt 索引，如果失败返回 -1
    """
    if not username or not env_id or episode_idx is None:
        return -1
    
    with lock:
        f, attempt_group, attempt_idx = _ensure_hdf5_file(username, env_id, episode_idx, create_new_attempt=True)
        if f is None or attempt_group is None:
            print(f"Error: Failed to create new attempt for {username}/{env_id}_{episode_idx}")
            return -1
        
        try:
            f.flush()
            print(f"Created new attempt_{attempt_idx} for {username}/{env_id}_{episode_idx}")
            return attempt_idx
        except Exception as e:
            print(f"Error creating new attempt: {e}")
            import traceback
            traceback.print_exc()
            return -1
        finally:
            if f is not None:
                try:
                    f.flush()
                except:
                    pass
                f.close()

def log_user_action(username, env_id, episode_idx, action_data):
    """
    记录用户的详细操作到 HDF5 文件中。
    
    新的数据格式：
    每个 action 组记录一次 execute action，包含 execute 之前所有的 click 和选择的 option。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        action_data: 操作数据字典，包含：
            - option_idx: execute 时使用的选项索引（最后一次选择的）
            - option_label: execute 时使用的选项标签
            - final_coordinates: 最后执行的坐标 {"x": x, "y": y}（可选）
            - final_coords_str: 最后执行的坐标字符串（可选）
            - final_image_array: 最后执行时的图片数组（可选）
            - option_selects_before_execute: 列表，包含 execute 之前所有的选项选择
            - coordinate_clicks_before_execute: 列表，包含 execute 之前所有的坐标点击（每个元素包含 coordinates, coords_str, image_array, timestamp）
            - status: 执行状态
            - done: 是否完成
    
    文件路径: data/user_action_logs/{username}/{env_id}_{episode_idx}.h5
    文件格式: HDF5，包含 actions 组，每个 action 记录一次 execute，包含 execute 之前所有的 option_select 和 coordinate_clicks
    """
    # 直接调用 HDF5 版本
    log_user_action_hdf5(username, env_id, episode_idx, action_data)
