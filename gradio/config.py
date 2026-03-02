"""
配置常量模块
"""
# --- Configuration ---
RESTRICT_VIDEO_PLAYBACK = True  # Restrict controls; we will force autoplay via JS
VIDEO_PLAYBACK_FPS = 30.0  # Frame rate for demonstration video playback
USE_SEGMENTED_VIEW = False  # Set to True to use segmented view, False to use original image

# Zone 高度配置
# Reference Zone: 参考区域（包含任务信息、实时流、演示视频的整个顶部区域）
REFERENCE_ZONE_HEIGHT = "50vh"  # Reference Zone 容器固定高度（整个参考区域的固定高度）
REFERENCE_VIEW_HEIGHT = "30vh"      # 参考区域中实时流图像的高度（图像元素高度，应小于 REFERENCE_ZONE_HEIGHT）
DEMO_VIDEO_HEIGHT = "30vh"          # 演示视频的固定高度（Watch video 部分的视频高度）

# Operation Zone: 操作区域（用户交互和操作控制的底部区域）
OPERATION_ZONE_HEIGHT = "45vh"  # Operation Zone 容器固定高度（整个操作区域的固定高度）

# 全局字体大小配置（绝对值）
FONT_SIZE = "20px"  # 统一字体大小，可在config.py中调整（如"14px", "16px", "18px", "20px"等）

# Reference Zone 三列宽度比例 (Text Info : Combined View : Demo Video)
TEXT_INFO_SCALE = 1  # 文本信息列的宽度比例（左侧）
COMBINED_VIEW_SCALE = 1  # 执行实时流列的宽度比例（中间）
DEMO_VIDEO_SCALE = 1  # 演示视频列的宽度比例（右侧）

# Operation Zone 三列宽度比例 (Live Observation : Action : Control)
LIVE_OBSERVATION_SCALE = 3  # Live Observation 列的宽度比例
ACTION_SCALE = 3  # Action 列的宽度比例
CONTROL_SCALE = 3  # Control 列的宽度比例

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
