"""
配置常量模块
"""
# --- Configuration ---
VIDEO_PLAYBACK_FPS = 20.0  # Frame rate for demonstration video playback
USE_SEGMENTED_VIEW = False  # Set to True to use segmented view, False to use original image
LIVE_OBS_REFRESH_HZ = 30.0  # Live observation refresh frequency in Hz
KEYFRAME_DOWNSAMPLE_FACTOR = 1  # Keep 1 frame out of every N streamed frames

# 主界面两列宽度比例 (Keypoint Selection : Right Panel)
KEYPOINT_SELECTION_SCALE = 1
CONTROL_PANEL_SCALE = 2

# 右侧顶部并排比例 (Action Selection : System Log)
RIGHT_TOP_ACTION_SCALE = 2
RIGHT_TOP_LOG_SCALE = 1

# Session超时配置
SESSION_TIMEOUT = 300  # Session超时时间（秒），如果30秒内没有execute_step操作，将自动回收session

# 兜底执行次数配置
EXECUTE_LIMIT_OFFSET = 4  # 兜底执行次数 = non_demonstration_task_length + EXECUTE_LIMIT_OFFSET


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

UI_TEXT = {
    "log": {
        "action_selection_prompt": "please select the action below 👇🏻,\nsome actions also need to select keypoint",
        "demo_video_prompt": 'press "Watch Video Input🎬" to watch a video\nNote: you can only watch the video once',
        "session_error": "Session Error",
        "reference_action_error": "Ground Truth Action Error: {error}",
        "reference_action_message": "Ground Truth Action: {option_label}. {option_action}",
        "reference_action_message_with_coords": "Ground Truth Action: {option_label}. {option_action} | coords: {coords_text}",
        "reference_action_status": "Ground Truth Action: {message}",
        "execute_missing_action": "Error: No action selected",
        "episode_success_banner": "********************************\n****   episode success      ****\n********************************\n  ---please press change episode----   ",
        "episode_failed_banner": "********************************\n****   episode failed       ****\n********************************\n  ---please press change episode----   ",
    },
    "coords": {
        "not_needed": "No need for coordinates",
        "select_keypoint": "please click the keypoint selection image",
        "select_keypoint_before_execute": "please click the keypoint selection image before execute!",
    },
    "errors": {
        "load_missing_task": "Error loading task: missing current_task",
        "load_invalid_task": "Error loading task: invalid task payload",
        "load_episode_error": "Error: {load_msg}",
        "next_task_failed": "Failed to load next task",
        "restart_missing_task": "Failed to restart episode: missing current task",
        "restart_invalid_task": "Failed to restart episode: invalid task payload",
        "switch_env_failed": "Failed to switch environment to '{selected_env}'",
        "init_failed": "Initialization error: {error}",
        "reference_action_resolve_failed": "Failed to resolve ground truth action.",
    },
}

def should_show_demo_video(env_id):
    """
    判断指定的环境ID是否应该显示demonstration video
    只有DEMO_VIDEO_ENV_IDS列表中的环境才显示demonstration videos
    """
    return env_id in DEMO_VIDEO_ENV_IDS
