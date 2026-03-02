"""
配置常量模块
"""
# --- Configuration ---
VIDEO_PLAYBACK_FPS = 30.0  # Frame rate for demonstration video playback
USE_SEGMENTED_VIEW = False  # Set to True to use segmented view, False to use original image

# 视图高度配置
REFERENCE_VIEW_HEIGHT = "30vh"      # 实时流和关键点图像高度
DEMO_VIDEO_HEIGHT = "30vh"          # 演示视频的固定高度（Watch video 部分的视频高度）

# 全局字体大小配置（绝对值）
FONT_SIZE = "20px"  # 统一字体大小，可在config.py中调整（如"14px", "16px", "18px", "20px"等）

# 主界面三列宽度比例 (System Log : Keypoint Selection : Control Panel)
SYSTEM_LOG_SCALE = 3
KEYPOINT_SELECTION_SCALE = 3
CONTROL_PANEL_SCALE = 3

# Session超时配置
SESSION_TIMEOUT = 300  # Session超时时间（秒），如果30秒内没有execute_step操作，将自动回收session

# 兜底执行次数配置
EXECUTE_LIMIT_OFFSET = 4  # 兜底执行次数 = non_demonstration_task_length + EXECUTE_LIMIT_OFFSET


# 【环境ID列表】
# 所有可用的测试环境ID，共16个环境
# 注意：PickHighlight 已添加到列表中，确保所有环境ID都可用
ENV_IDS = [
    "VideoPlaceOrder", "PickXtimes", "StopCube", "SwingXtimes", 
    "BinFill", "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", 
    "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "InsertPeg", 
    "MoveCube", "PatternLock", "RouteStick", "PickHighlight"  # 【新增】PickHighlight 环境ID
]

# 应该显示demonstration videos的环境ID列表
DEMO_VIDEO_ENV_IDS = [
    "VideoPlaceOrder",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick"
]

def should_show_demo_video(env_id):
    """
    判断指定的环境ID是否应该显示demonstration video
    只有DEMO_VIDEO_ENV_IDS列表中的环境才显示demonstration videos
    """
    return env_id in DEMO_VIDEO_ENV_IDS
