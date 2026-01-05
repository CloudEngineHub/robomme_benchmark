"""
图像处理工具模块
无状态的图像处理函数
"""
import numpy as np
import tempfile
import os
import traceback
import math
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
        
        # 确定左右边框宽度（RouteStick 任务需要更宽的左侧边框以容纳旋转方向示意图）
        left_border_width = 0
        right_border_width = 0
        if show_coordinate_axes:
            if env_id == "RouteStick":
                left_border_width = 250  # RouteStick 任务的左侧边框宽度
                right_border_width = 150 # 右侧边框宽度保持不变
            else:
                left_border_width = 150  # 其他任务的左右边框宽度
                right_border_width = 150
        
        if show_coordinate_axes:
            # 添加左右黑色边框用于绘制坐标系
            left_border = np.zeros((actual_h, left_border_width, 3), dtype=np.uint8)
            right_border = np.zeros((actual_h, right_border_width, 3), dtype=np.uint8)
            
            # 左右拼接（包含左右边框、中间黑边）
            concatenated_frame = np.concatenate([left_border, frame1, middle_border, frame2, right_border], axis=1)
            
            # 转换为PIL图像以便在黑色边框区域绘制坐标系
            concatenated_pil = Image.fromarray(concatenated_frame)
            
            # 在左侧黑色边框绘制 base camera 坐标系（旋转180度）
            left_border_pil = Image.new('RGB', (left_border_width, actual_h), (0, 0, 0))
            left_border_pil = draw_coordinate_axes(left_border_pil, position="left", rotate_180=True, env_id=env_id)
            
            # 在右侧黑色边框绘制 wrist camera 坐标系（不旋转）
            right_border_pil = Image.new('RGB', (right_border_width, actual_h), (0, 0, 0))
            right_border_pil = draw_coordinate_axes(right_border_pil, position="right", rotate_180=False, env_id=env_id)
            
            # 将坐标系绘制到拼接后的图像上
            concatenated_pil.paste(left_border_pil, (0, 0))
            concatenated_pil.paste(right_border_pil, (left_border_width + actual_w1 + border_width + actual_w2, 0))
            
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
            left_x = left_border_width + actual_w1 // 2 - left_text_width // 2  # 左侧图像中心位置（考虑左侧边框）
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
            right_x = left_border_width + actual_w1 + border_width + actual_w2 // 2 - right_text_width // 2  # 右侧图像中心位置（考虑左侧边框、中间黑边）
        else:
            right_x = actual_w1 + border_width + actual_w2 // 2 - right_text_width // 2  # 右侧图像中心位置（无左侧边框）
        right_y = h + (border_height - right_text_height) // 2
        draw.text((right_x, right_y), right_text, fill="white", font=font)
        
        # 转换回numpy数组
        concatenated_frame = np.array(pil_img)
        concatenated_frames.append(concatenated_frame)
    
    return concatenated_frames


def draw_coordinate_axes(img, position="right", rotate_180=False, env_id=None):
    """
    在图片外的黑色区域绘制坐标系，标注 forward/backward/left/right
    
    Args:
        img: PIL Image 或 numpy array
        position: "left" 或 "right"，指定在左侧还是右侧绘制
        rotate_180: 如果为 True，将坐标系顺时针旋转180度（用于 base camera）
        env_id: 环境ID，用于决定是否绘制特殊示意图（如 RouteStick 的旋转方向）
    
    Returns:
        PIL Image with coordinate axes drawn
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    # 获取图片尺寸
    width, height = img.size
    
    # 如果是 RouteStick 任务且位置在左侧，在最左边绘制旋转方向示意图
    if env_id == "RouteStick" and position == "left":
        # 绘制 clockwise 和 counterclockwise 示意图
        # 示意图位置：在图像最左边，垂直排列
        illustration_width = 150  # 示意图区域宽度，增加宽度以容纳文字
        illustration_spacing = 20  # 两个示意图之间的间距
        
        # 尝试加载字体
        try:
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            try:
                small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                small_font = ImageFont.load_default()
        
        line_color = (255, 255, 255)  # 白色
        circle_radius = 25  # 圆形箭头半径
        
        # Clockwise 示意图（上方）
        cw_center_x = illustration_width // 2
        cw_center_y = height // 2 - circle_radius - illustration_spacing // 2
        
        # 绘制顺时针圆形箭头（从下方开始，顺时针转到右侧）
        # 范围：90度（下）-> 180度（左）-> 270度（上）-> 360度（右）
        # 这样箭头在右侧，方向向下
        arc_points = []
        for angle_deg in range(90, 361, 5):  # 顺时针增加
            angle_rad = math.radians(angle_deg)
            x = cw_center_x + circle_radius * math.cos(angle_rad)
            y = cw_center_y + circle_radius * math.sin(angle_rad)
            arc_points.append((x, y))
        # 绘制圆弧线段
        for i in range(len(arc_points) - 1):
            draw.line([arc_points[i], arc_points[i+1]], fill=line_color, width=2)
        
        # 绘制箭头头部（在0度/360度位置，即右侧）
        # 在右侧，顺时针切线方向是向下
        arrow_size = 6
        arrow_x = cw_center_x + circle_radius
        arrow_y = cw_center_y
        # 箭头指向下
        draw.polygon(
            [(arrow_x, arrow_y + arrow_size), # 尖端向下
             (arrow_x - arrow_size, arrow_y - arrow_size // 2),
             (arrow_x + arrow_size, arrow_y - arrow_size // 2)],
            fill=line_color
        )
        
        # 添加 "clockwise" 文字标签
        cw_text = "clockwise"
        cw_bbox = draw.textbbox((0, 0), cw_text, font=small_font)
        cw_text_width = cw_bbox[2] - cw_bbox[0]
        cw_text_height = cw_bbox[3] - cw_bbox[1]
        cw_text_x = cw_center_x - cw_text_width // 2
        cw_text_y = cw_center_y + circle_radius + 5
        draw.rectangle(
            [(cw_text_x - 2, cw_text_y - 2),
             (cw_text_x + cw_text_width + 2, cw_text_y + cw_text_height + 2)],
            fill=(0, 0, 0)
        )
        draw.text((cw_text_x, cw_text_y), cw_text, fill=line_color, font=small_font)
        
        # Counterclockwise 示意图（下方）
        ccw_center_x = illustration_width // 2
        ccw_center_y = height // 2 + circle_radius + illustration_spacing // 2
        
        # 绘制逆时针圆形箭头（从下方开始，逆时针转到左侧）
        # 范围：90度（下）-> 0度（右）-> -90度（上）-> -180度（左）
        # 这样箭头在左侧，方向向下
        arc_points_ccw = []
        for angle_deg in range(90, -181, -5):  # 逆时针减小
            angle_rad = math.radians(angle_deg)
            x = ccw_center_x + circle_radius * math.cos(angle_rad)
            y = ccw_center_y + circle_radius * math.sin(angle_rad)
            arc_points_ccw.append((x, y))
        # 绘制圆弧线段
        for i in range(len(arc_points_ccw) - 1):
            draw.line([arc_points_ccw[i], arc_points_ccw[i+1]], fill=line_color, width=2)
        
        # 绘制箭头头部（在180度/-180度位置，即左侧）
        # 在左侧，逆时针切线方向是向下
        arrow_x_ccw = ccw_center_x - circle_radius
        arrow_y_ccw = ccw_center_y
        # 箭头指向下
        draw.polygon(
            [(arrow_x_ccw, arrow_y_ccw + arrow_size), # 尖端向下
             (arrow_x_ccw - arrow_size, arrow_y_ccw - arrow_size // 2),
             (arrow_x_ccw + arrow_size, arrow_y_ccw - arrow_size // 2)],
            fill=line_color
        )
        
        # 添加 "counterclockwise" 文字标签
        ccw_text = "counterclockwise"
        ccw_bbox = draw.textbbox((0, 0), ccw_text, font=small_font)
        ccw_text_width = ccw_bbox[2] - ccw_bbox[0]
        ccw_text_height = ccw_bbox[3] - ccw_bbox[1]
        ccw_text_x = ccw_center_x - ccw_text_width // 2
        ccw_text_y = ccw_center_y + circle_radius + 5
        draw.rectangle(
            [(ccw_text_x - 2, ccw_text_y - 2),
             (ccw_text_x + ccw_text_width + 2, ccw_text_y + ccw_text_height + 2)],
            fill=(0, 0, 0)
        )
        draw.text((ccw_text_x, ccw_text_y), ccw_text, fill=line_color, font=small_font)
    
    # 坐标系位置（在黑色边框内）
    axis_size = 60  # 坐标系大小
    
    # 如果是 RouteStick 任务且位置在左侧，坐标系需要向右偏移以避开旋转方向示意图
    # 保持左右两个坐标系中心距离图片的距离一致
    # 右侧坐标系在宽为150的边框中心，即距离图片 150/2 = 75
    # 左侧坐标系在宽为250的边框中，应距离右边（图片侧）75
    # 即 x = width - 75 - axis_size/2
    if env_id == "RouteStick" and position == "left":
        # 保持与右侧对称，距离右边缘（靠近图片的一侧）75像素中心
        # width 是总宽度 (250)，中心点应为 width - 75
        center_x_pos = width - 75
        origin_x = center_x_pos - axis_size // 2
    else:
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
