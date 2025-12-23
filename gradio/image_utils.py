"""
图像处理工具模块
无状态的图像处理函数
"""
import numpy as np
import tempfile
import os
import traceback
from PIL import Image, ImageDraw, ImageFont
import cv2
from config import VIDEO_PLAYBACK_FPS


def save_video(frames, suffix=""):
    """
    视频保存函数 - 使用imageio生成视频
    
    优化点：
    1. 使用imageio.mimwrite，不依赖FFmpeg编码器
    2. 直接处理RGB帧，无需颜色空间转换
    3. 自动处理编码，简单可靠
    """
    if not frames or len(frames) == 0:
        return None
    
    try:
        import imageio
        
        # 准备帧：确保是uint8格式的numpy数组
        processed_frames = []
        for f in frames:
            if not isinstance(f, np.ndarray):
                f = np.array(f)
            # 确保是uint8格式
            if f.dtype != np.uint8:
                if np.max(f) <= 1.0:
                    f = (f * 255).astype(np.uint8)
                else:
                    f = f.clip(0, 255).astype(np.uint8)
            # 处理灰度图
            if len(f.shape) == 2:
                f = np.stack([f] * 3, axis=-1)
            # imageio期望RGB格式，frames已经是RGB
            processed_frames.append(f)
        
        fd, path = tempfile.mkstemp(suffix=f"_{suffix}.mp4")
        os.close(fd)
        
        # imageio.mimwrite会自动处理编码
        imageio.mimwrite(path, processed_frames, fps=VIDEO_PLAYBACK_FPS, quality=8, macro_block_size=None)

        return path
    except ImportError:
        print("Error: imageio module not found. Please install it: pip install imageio imageio-ffmpeg")
        return None
    except Exception as e:
        print(f"Error in save_video: {e}")
        traceback.print_exc()
        return None


def concatenate_frames_horizontally(frames1, frames2):
    """
    将两个帧序列左右拼接成一个帧序列
    
    Args:
        frames1: 左侧视频帧列表（base frames）
        frames2: 右侧视频帧列表（wrist frames）
    
    Returns:
        拼接后的帧列表
    """
    if not frames1 and not frames2:
        return []
    
    # 确定最大帧数
    max_frames = max(len(frames1), len(frames2))
    concatenated_frames = []
    
    for i in range(max_frames):
        # 获取当前帧，如果某个序列较短，重复最后一帧
        frame1 = frames1[min(i, len(frames1) - 1)] if frames1 else None
        frame2 = frames2[min(i, len(frames2) - 1)] if frames2 else None
        
        # 转换为numpy数组并确保格式正确
        if frame1 is not None:
            if not isinstance(frame1, np.ndarray):
                frame1 = np.array(frame1)
            if frame1.dtype != np.uint8:
                if np.max(frame1) <= 1.0:
                    frame1 = (frame1 * 255).astype(np.uint8)
                else:
                    frame1 = frame1.clip(0, 255).astype(np.uint8)
            if len(frame1.shape) == 2:
                frame1 = np.stack([frame1] * 3, axis=-1)
        else:
            # 如果frame1为空，创建一个黑色帧
            if frame2 is not None:
                h, w = frame2.shape[:2]
                frame1 = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                continue
        
        if frame2 is not None:
            if not isinstance(frame2, np.ndarray):
                frame2 = np.array(frame2)
            if frame2.dtype != np.uint8:
                if np.max(frame2) <= 1.0:
                    frame2 = (frame2 * 255).astype(np.uint8)
                else:
                    frame2 = frame2.clip(0, 255).astype(np.uint8)
            if len(frame2.shape) == 2:
                frame2 = np.stack([frame2] * 3, axis=-1)
        else:
            # 如果frame2为空，创建一个黑色帧
            h, w = frame1.shape[:2]
            frame2 = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 确保两个帧的高度相同，如果不同则调整
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        if h1 != h2:
            # 调整到相同高度（使用较小的那个）
            target_h = min(h1, h2)
            if h1 != target_h:
                frame1 = cv2.resize(frame1, (w1, target_h), interpolation=cv2.INTER_LINEAR)
            if h2 != target_h:
                frame2 = cv2.resize(frame2, (w2, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 获取调整后的实际宽度和高度
        actual_h, actual_w1 = frame1.shape[:2]
        _, actual_w2 = frame2.shape[:2]
        
        # 在中间添加垂直黑边分隔
        border_width = 10  # 中间黑边宽度
        middle_border = np.zeros((actual_h, border_width, 3), dtype=np.uint8)
        
        # 左右拼接（包含中间黑边）
        concatenated_frame = np.concatenate([frame1, middle_border, frame2], axis=1)
        
        # 添加底部黑边用于标注
        h, w = concatenated_frame.shape[:2]
        border_height = 40  # 黑边高度
        # 创建带黑边的图像
        frame_with_border = np.zeros((h + border_height, w, 3), dtype=np.uint8)
        frame_with_border[:h, :] = concatenated_frame
        
        # 转换为PIL图像以便添加文字
        pil_img = Image.fromarray(frame_with_border)
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
        
        # 在左侧黑边区域添加 "base camera" 标注
        left_text = "base camera"
        left_text_bbox = draw.textbbox((0, 0), left_text, font=font)
        left_text_width = left_text_bbox[2] - left_text_bbox[0]
        left_text_height = left_text_bbox[3] - left_text_bbox[1]
        left_x = actual_w1 // 2 - left_text_width // 2  # 左侧图像中心位置
        left_y = h + (border_height - left_text_height) // 2
        draw.text((left_x, left_y), left_text, fill="white", font=font)
        
        # 在右侧黑边区域添加 "wrist camera" 标注
        right_text = "wrist camera"
        right_text_bbox = draw.textbbox((0, 0), right_text, font=font)
        right_text_width = right_text_bbox[2] - right_text_bbox[0]
        right_text_height = right_text_bbox[3] - right_text_bbox[1]
        right_x = actual_w1 + border_width + actual_w2 // 2 - right_text_width // 2  # 右侧图像中心位置（考虑中间黑边）
        right_y = h + (border_height - right_text_height) // 2
        draw.text((right_x, right_y), right_text, fill="white", font=font)
        
        # 转换回numpy数组
        concatenated_frame = np.array(pil_img)
        concatenated_frames.append(concatenated_frame)
    
    return concatenated_frames


def draw_marker(img, x, y):
    """Draws a red circle and cross at (x, y)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.copy()
    draw = ImageDraw.Draw(img)
    r = 5
    # Circle
    draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
    # Cross
    draw.line((x-r, y, x+r, y), fill="red", width=2)
    draw.line((x, y-r, x, y+r), fill="red", width=2)
    return img
