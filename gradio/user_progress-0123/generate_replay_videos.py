#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Run_Peng 用户的所有错误任务生成回放视频
每个 action 显示最后一个 annotated image 和选择的 option
"""

import json
import os
import sys
from pathlib import Path

# 检查必要的依赖
try:
    import h5py
except ImportError:
    print("错误: 缺少 h5py 模块，请安装: pip install h5py")
    print(f"当前 Python 路径: {sys.executable}")
    print(f"Python 版本: {sys.version}")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("错误: 缺少 numpy 模块，请安装: pip install numpy")
    print(f"当前 Python 路径: {sys.executable}")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("错误: 缺少 Pillow 模块，请安装: pip install Pillow")
    print(f"当前 Python 路径: {sys.executable}")
    sys.exit(1)

try:
    import imageio
except ImportError:
    print("错误: 缺少 imageio 模块，请安装: pip install imageio imageio-ffmpeg")
    print(f"当前 Python 路径: {sys.executable}")
    sys.exit(1)

# 配置路径
BASE_DIR = Path(__file__).parent.absolute()
JSONL_FILE = BASE_DIR / "Run_Peng.jsonl"
USER_ACTION_LOG_DIR = BASE_DIR / "data" / "user_action_logs" / "Run_Peng"
OUTPUT_VIDEO_DIR = BASE_DIR / "data" / "generate_video"

# 视频参数
FPS = 1  # 每个 action 显示 1 秒


def load_failed_tasks():
    """读取 Run_Peng.jsonl 文件，筛选出所有失败的任务"""
    failed_tasks = []
    
    if not JSONL_FILE.exists():
        print(f"错误: 找不到文件 {JSONL_FILE}")
        return failed_tasks
    
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                status = record.get('status')
                
                if status == 'failed':
                    env_id = record.get('env_id')
                    episode_idx = record.get('episode_idx')
                    
                    if env_id is not None and episode_idx is not None:
                        failed_tasks.append({
                            'env_id': env_id,
                            'episode_idx': episode_idx,
                            'line_num': line_num
                        })
                        print(f"找到错误任务: {env_id}_{episode_idx} (行 {line_num})")
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    return failed_tasks


def get_action_indices(actions_group):
    """从 actions 组中提取所有 action 的索引，按数字顺序排序"""
    action_indices = []
    
    for key in actions_group.keys():
        if key.startswith('action_'):
            try:
                idx = int(key.split('_')[1])
                action_indices.append(idx)
            except (ValueError, IndexError):
                continue
    
    return sorted(action_indices)


def extract_action_data(action_group, option_list=None, default_image_size=(224, 224)):
    """从 action 组中提取图像和 option 信息
    
    Args:
        action_group: h5py Group 对象，表示 action 组
        option_list: 可选的 option_list，用于匹配 option_idx 到 label
        default_image_size: 当没有图像数据时，创建黑色图像的默认尺寸 (height, width)
    """
    try:
        last_image = None
        
        # 读取 click_history_annotated_image
        if 'click_history_annotated_image' in action_group:
            images = action_group['click_history_annotated_image']
            
            # 获取最后一个图像
            if images.shape[0] > 0:
                last_image = images[-1]  # 形状: [H, W, 3]
                
                # 确保是 uint8 格式
                if last_image.dtype != np.uint8:
                    if np.max(last_image) <= 1.0:
                        last_image = (last_image * 255).astype(np.uint8)
                    else:
                        last_image = last_image.clip(0, 255).astype(np.uint8)
        
        # 如果没有图像数据，创建黑色图像
        if last_image is None:
            h, w = default_image_size
            last_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 读取 final_choice 获取 option_idx
        option_idx = None
        option_label = None
        
        if 'final_choice' in action_group:
            final_choice = action_group['final_choice'][:]
            if len(final_choice) >= 1:
                option_idx = int(final_choice[0])
        
        # 优先从 option_list 中匹配 option_label
        if option_idx is not None and option_list is not None:
            try:
                if option_idx >= 0 and option_idx < len(option_list):
                    option_item = option_list[option_idx]
                    if isinstance(option_item, np.void):
                        # 结构化数组
                        label = option_item['label']
                        if label is not None:
                            # 解码字节字符串
                            if isinstance(label, bytes):
                                option_label = label.decode('utf-8')
                            else:
                                option_label = str(label)
            except Exception as e:
                print(f"    警告: 从 option_list 匹配失败: {e}")
        
        # 如果从 option_list 没有找到，尝试从 option_history 获取选项标签
        if option_label is None and 'option_history' in action_group:
            option_history = action_group['option_history']
            if len(option_history) > 0:
                # option_history 是结构化数组，包含 (idx, label) 对
                # 获取最后一个选项
                last_option = option_history[-1]
                if isinstance(last_option, np.void):
                    # 结构化数组
                    option_idx_from_history = last_option['idx']
                    option_label_from_history = last_option['label']
                    if option_label_from_history is not None:
                        # 解码字节字符串
                        if isinstance(option_label_from_history, bytes):
                            option_label = option_label_from_history.decode('utf-8')
                        else:
                            option_label = str(option_label_from_history)
        
        return last_image, {
            'option_idx': option_idx,
            'option_label': option_label
        }
    
    except Exception as e:
        print(f"  错误: 提取 action 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def draw_option_text(image, option_info):
    """在图像上绘制 option 信息（上方，自动换行，8号字体）"""
    # 转换为 PIL Image
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体（8号字体）
    font_size = 8
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    # 构建显示文本
    option_idx = option_info.get('option_idx')
    option_label = option_info.get('option_label')
    
    if option_label:
        text = f"Option: {option_label}"
    elif option_idx is not None:
        text = f"Option: {option_idx}"
    else:
        text = "Option: N/A"
    
    # 获取图像尺寸
    img_width, img_height = pil_image.size
    
    # 设置文本区域参数
    padding = 5
    max_width = img_width - 2 * padding  # 文本区域最大宽度
    
    # 自动换行处理
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # 测试添加单词后的宽度
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        
        if test_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # 计算总文本高度
    line_height = font_size + 2  # 行高（字体大小 + 行间距）
    total_text_height = len(lines) * line_height
    
    # 在图像上方绘制文本（带黑色背景）
    text_x = padding
    text_y = padding
    
    # 绘制黑色背景矩形（覆盖所有行）
    draw.rectangle(
        [(text_x - padding, text_y - padding),
         (text_x + max_width + padding, text_y + total_text_height + padding)],
        fill=(0, 0, 0)
    )
    
    # 逐行绘制白色文字
    for i, line in enumerate(lines):
        y_pos = text_y + i * line_height
        draw.text((text_x, y_pos), line, fill=(255, 255, 255), font=font)
    
    # 转换回 numpy 数组
    return np.array(pil_image)


def process_task(env_id, episode_idx):
    """处理单个错误任务，生成回放视频"""
    h5_filename = f"{env_id}_{episode_idx}.h5"
    h5_path = USER_ACTION_LOG_DIR / h5_filename
    
    if not h5_path.exists():
        print(f"  跳过: 文件不存在 {h5_path}")
        return False
    
    print(f"  处理: {h5_filename}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # 读取 attempt_0
            if 'attempt_0' not in f:
                print(f"  跳过: 找不到 attempt_0")
                return False
            
            attempt_group = f['attempt_0']
            
            # 读取 metadata 中的 option_list
            option_list = None
            if 'metadata' in attempt_group:
                metadata_group = attempt_group['metadata']
                if 'option_list' in metadata_group:
                    try:
                        option_list = metadata_group['option_list'][:]
                        print(f"  读取到 option_list，包含 {len(option_list)} 个选项")
                    except Exception as e:
                        print(f"  警告: 读取 option_list 失败: {e}")
            
            # 读取 actions
            if 'actions' not in attempt_group:
                print(f"  跳过: 找不到 actions 组")
                return False
            
            actions_group = attempt_group['actions']
            
            # 获取所有 action 索引
            action_indices = get_action_indices(actions_group)
            
            if not action_indices:
                print(f"  跳过: 没有找到任何 action")
                return False
            
            print(f"  找到 {len(action_indices)} 个 actions")
            
            # 提取所有 action 的图像
            frames = []
            
            for action_idx in action_indices:
                action_name = f"action_{action_idx}"
                
                if action_name not in actions_group:
                    print(f"  警告: 找不到 {action_name}")
                    continue
                
                action_group = actions_group[action_name]
                
                # 提取图像和 option 信息（传入 option_list 用于匹配）
                # 尝试从第一个有图像的 action 获取图像尺寸，如果没有则使用默认尺寸
                default_size = (224, 224)  # 默认图像尺寸
                if frames:
                    # 使用已有帧的尺寸作为默认尺寸
                    default_size = (frames[0].shape[0], frames[0].shape[1])
                elif 'click_history_annotated_image' in action_group:
                    # 尝试从当前 action 获取尺寸
                    try:
                        images = action_group['click_history_annotated_image']
                        if images.shape[0] > 0:
                            default_size = (images.shape[1], images.shape[2])
                    except:
                        pass
                
                image, option_info = extract_action_data(action_group, option_list=option_list, default_image_size=default_size)
                
                if image is None:
                    print(f"  警告: {action_name} 无法提取图像数据")
                    continue
                
                # 在图像上绘制 option 文本
                annotated_image = draw_option_text(image, option_info)
                frames.append(annotated_image)
                
                option_text = option_info.get('option_label') or option_info.get('option_idx') or 'N/A'
                print(f"    {action_name}: Option = {option_text}")
            
            if not frames:
                print(f"  跳过: 没有提取到任何有效帧")
                return False
            
            # 生成视频
            output_dir = OUTPUT_VIDEO_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_filename = f"{env_id}_{episode_idx}.mp4"
            video_path = output_dir / video_filename
            
            # 确保所有帧都是 uint8 格式
            processed_frames = []
            for frame in frames:
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                if frame.dtype != np.uint8:
                    if np.max(frame) <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.clip(0, 255).astype(np.uint8)
                processed_frames.append(frame)
            
            # 使用 imageio 生成视频
            imageio.mimwrite(
                str(video_path),
                processed_frames,
                fps=FPS,
                quality=8,
                macro_block_size=None
            )
            
            print(f"  成功: 生成视频 {video_path} ({len(frames)} 帧)")
            return True
    
    except Exception as e:
        print(f"  错误: 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("生成 Run_Peng 错误任务回放视频")
    print("=" * 80)
    
    # 读取错误任务列表
    print("\n1. 读取错误任务列表...")
    failed_tasks = load_failed_tasks()
    
    if not failed_tasks:
        print("没有找到任何错误任务")
        return
    
    print(f"找到 {len(failed_tasks)} 个错误任务\n")
    
    # 处理每个错误任务
    print("2. 处理错误任务...")
    success_count = 0
    fail_count = 0
    
    for i, task in enumerate(failed_tasks, 1):
        env_id = task['env_id']
        episode_idx = task['episode_idx']
        
        print(f"\n[{i}/{len(failed_tasks)}] {env_id}_{episode_idx}:")
        
        if process_task(env_id, episode_idx):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(failed_tasks)}")
    print(f"\n视频保存目录: {OUTPUT_VIDEO_DIR}")


if __name__ == "__main__":
    main()
