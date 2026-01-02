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


def concatenate_frames_horizontally(frames1, frames2, env_id=None):
    """
    将两个帧序列左右拼接成一个帧序列
    
    Args:
        frames1: 左侧视频帧列表（base frames）
        frames2: 右侧视频帧列表（wrist frames）
        env_id: 环境ID，用于决定是否显示坐标系（可选）
    
    Returns:
        拼接后的帧列表
    """
    # 需要显示坐标系的任务列表
    COORDINATE_AXES_ENVS = ["PatternLock", "RouteStick", "InsertPeg", "SwingXtimes"]
    show_coordinate_axes = env_id in COORDINATE_AXES_ENVS if env_id else False
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
        
        if show_coordinate_axes:
            # 添加左右黑色边框用于绘制坐标系
            side_border_width = 150  # 左右边框宽度
            left_border = np.zeros((actual_h, side_border_width, 3), dtype=np.uint8)
            right_border = np.zeros((actual_h, side_border_width, 3), dtype=np.uint8)
            
            # 左右拼接（包含左右边框、中间黑边）
            concatenated_frame = np.concatenate([left_border, frame1, middle_border, frame2, right_border], axis=1)
            
            # 转换为PIL图像以便在黑色边框区域绘制坐标系
            concatenated_pil = Image.fromarray(concatenated_frame)
            
            # 在左侧黑色边框绘制 base camera 坐标系（旋转180度）
            left_border_pil = Image.new('RGB', (side_border_width, actual_h), (0, 0, 0))
            left_border_pil = draw_coordinate_axes(left_border_pil, position="left", rotate_180=True)
            
            # 在右侧黑色边框绘制 wrist camera 坐标系（不旋转）
            right_border_pil = Image.new('RGB', (side_border_width, actual_h), (0, 0, 0))
            right_border_pil = draw_coordinate_axes(right_border_pil, position="right", rotate_180=False)
            
            # 将坐标系绘制到拼接后的图像上
            concatenated_pil.paste(left_border_pil, (0, 0))
            concatenated_pil.paste(right_border_pil, (side_border_width + actual_w1 + border_width + actual_w2, 0))
            
            # 转换回numpy数组
            concatenated_frame = np.array(concatenated_pil)
        else:
            # 不显示坐标系，直接拼接（不添加左右边框）
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
        if show_coordinate_axes:
            left_x = side_border_width + actual_w1 // 2 - left_text_width // 2  # 左侧图像中心位置（考虑左侧边框）
        else:
            left_x = actual_w1 // 2 - left_text_width // 2  # 左侧图像中心位置（无左侧边框）
        left_y = h + (border_height - left_text_height) // 2
        draw.text((left_x, left_y), left_text, fill="white", font=font)
        
        # 在右侧黑边区域添加 "wrist camera" 标注
        right_text = "wrist camera"
        right_text_bbox = draw.textbbox((0, 0), right_text, font=font)
        right_text_width = right_text_bbox[2] - right_text_bbox[0]
        right_text_height = right_text_bbox[3] - right_text_bbox[1]
        if show_coordinate_axes:
            right_x = side_border_width + actual_w1 + border_width + actual_w2 // 2 - right_text_width // 2  # 右侧图像中心位置（考虑左侧边框、中间黑边）
        else:
            right_x = actual_w1 + border_width + actual_w2 // 2 - right_text_width // 2  # 右侧图像中心位置（无左侧边框）
        right_y = h + (border_height - right_text_height) // 2
        draw.text((right_x, right_y), right_text, fill="white", font=font)
        
        # 转换回numpy数组
        concatenated_frame = np.array(pil_img)
        concatenated_frames.append(concatenated_frame)
    
    return concatenated_frames


def draw_coordinate_axes(img, position="right", rotate_180=False):
    """
    在图片外的黑色区域绘制坐标系，标注 forward/backward/left/right
    
    Args:
        img: PIL Image 或 numpy array
        position: "left" 或 "right"，指定在左侧还是右侧绘制
        rotate_180: 如果为 True，将坐标系顺时针旋转180度（用于 base camera）
    
    Returns:
        PIL Image with coordinate axes drawn
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    # 获取图片尺寸
    width, height = img.size
    
    # 坐标系位置（在黑色边框内）
    axis_size = 60  # 坐标系大小
    
    # 坐标轴中心位于边框宽度的中心
    origin_x = width // 2 - axis_size // 2
    origin_y = height // 2 - axis_size // 2
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
    
    # 绘制坐标轴（十字形）
    axis_length = axis_size - 20
    center_x = origin_x + axis_size // 2
    center_y = origin_y + axis_size // 2
    
    # 绘制坐标轴线条（白色，带半透明效果）
    line_color = (255, 255, 255)  # 白色
    line_width = 2
    
    # 根据是否旋转180度，调整方向
    if rotate_180:
        # 旋转180度：forward变成backward，left变成right
        # 水平轴（left-right，但方向相反）
        draw.line(
            [(center_x - axis_length // 2, center_y), 
             (center_x + axis_length // 2, center_y)],
            fill=line_color, width=line_width
        )
        
        # 垂直轴（forward-backward，但方向相反）
        draw.line(
            [(center_x, center_y - axis_length // 2), 
             (center_x, center_y + axis_length // 2)],
            fill=line_color, width=line_width
        )
        
        # 绘制箭头（旋转180度后的方向）
        arrow_size = 5
        # Forward 箭头（现在在下方，原来是上方）
        draw.polygon(
            [(center_x, center_y + axis_length // 2),
             (center_x - arrow_size, center_y + axis_length // 2 - arrow_size),
             (center_x + arrow_size, center_y + axis_length // 2 - arrow_size)],
            fill=line_color
        )
        # Backward 箭头（现在在上方，原来是下方）
        draw.polygon(
            [(center_x, center_y - axis_length // 2),
             (center_x - arrow_size, center_y - axis_length // 2 + arrow_size),
             (center_x + arrow_size, center_y - axis_length // 2 + arrow_size)],
            fill=line_color
        )
        # Right 箭头（现在在左侧，原来是右侧）
        draw.polygon(
            [(center_x - axis_length // 2, center_y),
             (center_x - axis_length // 2 + arrow_size, center_y - arrow_size),
             (center_x - axis_length // 2 + arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        # Left 箭头（现在在右侧，原来是左侧）
        draw.polygon(
            [(center_x + axis_length // 2, center_y),
             (center_x + axis_length // 2 - arrow_size, center_y - arrow_size),
             (center_x + axis_length // 2 - arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        
        # 添加文字标签（旋转180度后的位置）
        text_color = (255, 255, 255)  # 白色文字
        
        # Forward (现在在下方)
        forward_text = "forward"
        forward_bbox = draw.textbbox((0, 0), forward_text, font=font)
        forward_width = forward_bbox[2] - forward_bbox[0]
        forward_x = center_x - forward_width // 2
        forward_y = center_y + axis_length // 2 + 5
        draw.rectangle(
            [(forward_x - 2, forward_y - 2), 
             (forward_x + forward_width + 2, forward_y + (forward_bbox[3] - forward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((forward_x, forward_y), forward_text, fill=text_color, font=font)
        
        # Backward (现在在上方)
        backward_text = "backward"
        backward_bbox = draw.textbbox((0, 0), backward_text, font=font)
        backward_width = backward_bbox[2] - backward_bbox[0]
        backward_x = center_x - backward_width // 2
        backward_y = center_y - axis_length // 2 - 20
        draw.rectangle(
            [(backward_x - 2, backward_y - 2), 
             (backward_x + backward_width + 2, backward_y + (backward_bbox[3] - backward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((backward_x, backward_y), backward_text, fill=text_color, font=font)
        
        # Right (现在在左侧)
        right_text = "right"
        right_bbox = draw.textbbox((0, 0), right_text, font=font)
        right_width = right_bbox[2] - right_bbox[0]
        right_x = center_x - axis_length // 2 - right_width - 5
        right_y = center_y - (right_bbox[3] - right_bbox[1]) // 2
        draw.rectangle(
            [(right_x - 2, right_y - 2), 
             (right_x + right_width + 2, right_y + (right_bbox[3] - right_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((right_x, right_y), right_text, fill=text_color, font=font)
        
        # Left (现在在右侧)
        left_text = "left"
        left_bbox = draw.textbbox((0, 0), left_text, font=font)
        left_width = left_bbox[2] - left_bbox[0]
        left_x = center_x + axis_length // 2 + 5
        left_y = center_y - (left_bbox[3] - left_bbox[1]) // 2
        draw.rectangle(
            [(left_x - 2, left_y - 2), 
             (left_x + left_width + 2, left_y + (left_bbox[3] - left_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((left_x, left_y), left_text, fill=text_color, font=font)
    else:
        # 正常方向（不旋转）
        # 水平轴（left-right）
        draw.line(
            [(center_x - axis_length // 2, center_y), 
             (center_x + axis_length // 2, center_y)],
            fill=line_color, width=line_width
        )
        
        # 垂直轴（forward-backward）
        draw.line(
            [(center_x, center_y - axis_length // 2), 
             (center_x, center_y + axis_length // 2)],
            fill=line_color, width=line_width
        )
        
        # 绘制箭头（在轴的两端）
        arrow_size = 5
        # Forward (上) 箭头
        draw.polygon(
            [(center_x, center_y - axis_length // 2),
             (center_x - arrow_size, center_y - axis_length // 2 + arrow_size),
             (center_x + arrow_size, center_y - axis_length // 2 + arrow_size)],
            fill=line_color
        )
        # Backward (下) 箭头
        draw.polygon(
            [(center_x, center_y + axis_length // 2),
             (center_x - arrow_size, center_y + axis_length // 2 - arrow_size),
             (center_x + arrow_size, center_y + axis_length // 2 - arrow_size)],
            fill=line_color
        )
        # Right (右) 箭头
        draw.polygon(
            [(center_x + axis_length // 2, center_y),
             (center_x + axis_length // 2 - arrow_size, center_y - arrow_size),
             (center_x + axis_length // 2 - arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        # Left (左) 箭头
        draw.polygon(
            [(center_x - axis_length // 2, center_y),
             (center_x - axis_length // 2 + arrow_size, center_y - arrow_size),
             (center_x - axis_length // 2 + arrow_size, center_y + arrow_size)],
            fill=line_color
        )
        
        # 添加文字标签
        text_color = (255, 255, 255)  # 白色文字
        
        # Forward (上)
        forward_text = "forward"
        forward_bbox = draw.textbbox((0, 0), forward_text, font=font)
        forward_width = forward_bbox[2] - forward_bbox[0]
        forward_x = center_x - forward_width // 2
        forward_y = center_y - axis_length // 2 - 20
        draw.rectangle(
            [(forward_x - 2, forward_y - 2), 
             (forward_x + forward_width + 2, forward_y + (forward_bbox[3] - forward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((forward_x, forward_y), forward_text, fill=text_color, font=font)
        
        # Backward (下)
        backward_text = "backward"
        backward_bbox = draw.textbbox((0, 0), backward_text, font=font)
        backward_width = backward_bbox[2] - backward_bbox[0]
        backward_x = center_x - backward_width // 2
        backward_y = center_y + axis_length // 2 + 5
        draw.rectangle(
            [(backward_x - 2, backward_y - 2), 
             (backward_x + backward_width + 2, backward_y + (backward_bbox[3] - backward_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((backward_x, backward_y), backward_text, fill=text_color, font=font)
        
        # Right (右)
        right_text = "right"
        right_bbox = draw.textbbox((0, 0), right_text, font=font)
        right_width = right_bbox[2] - right_bbox[0]
        right_x = center_x + axis_length // 2 + 5
        right_y = center_y - (right_bbox[3] - right_bbox[1]) // 2
        draw.rectangle(
            [(right_x - 2, right_y - 2), 
             (right_x + right_width + 2, right_y + (right_bbox[3] - right_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((right_x, right_y), right_text, fill=text_color, font=font)
        
        # Left (左)
        left_text = "left"
        left_bbox = draw.textbbox((0, 0), left_text, font=font)
        left_width = left_bbox[2] - left_bbox[0]
        left_x = center_x - axis_length // 2 - left_width - 5
        left_y = center_y - (left_bbox[3] - left_bbox[1]) // 2
        draw.rectangle(
            [(left_x - 2, left_y - 2), 
             (left_x + left_width + 2, left_y + (left_bbox[3] - left_bbox[1]) + 2)],
            fill=(0, 0, 0)
        )
        draw.text((left_x, left_y), left_text, fill=text_color, font=font)
    
    return img


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
