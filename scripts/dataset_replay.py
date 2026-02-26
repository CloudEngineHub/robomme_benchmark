"""
Replay episodes from HDF5 datasets and save rollout videos.
Loads recorded actions from record_dataset_<Task>.h5, steps the environment
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal, Union

import cv2
import h5py
import imageio
import numpy as np
import torch

from robomme.env_record_wrapper import BenchmarkEnvBuilder

GUI_RENDER = False
REPLAY_VIDEO_DIR = "replay_videos"
VIDEO_FPS = 30
VIDEO_BORDER_COLOR = (255, 0, 0)
VIDEO_BORDER_THICKNESS = 10

TaskID = Literal[
    # "BinFill",
    # "PickXtimes",
    # "SwingXtimes",
    # "StopCube",
    # "VideoUnmask",
    "VideoUnmaskSwap",
    # "ButtonUnmask",
    # "ButtonUnmaskSwap",
    # "PickHighlight",
    # "VideoRepick",
    # "VideoPlaceButton",
    # "VideoPlaceOrder",
    # "MoveCube",
    # "InsertPeg",
    # "PatternLock",
    # "RouteStick",
]

ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]

def _to_numpy(t) -> np.ndarray:
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _frame_from_obs(
    front: np.ndarray | torch.Tensor,
    wrist: np.ndarray | torch.Tensor,
    is_video_demo: bool = False,
) -> np.ndarray:
    frame = np.hstack([_to_numpy(front), _to_numpy(wrist)]).astype(np.uint8)
    if is_video_demo:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h),
                      VIDEO_BORDER_COLOR, VIDEO_BORDER_THICKNESS)
    return frame


def _extract_frames(obs: dict, is_video_demo_fn=None) -> list[np.ndarray]:
    n = len(obs["front_rgb_list"])
    return [
        _frame_from_obs(
            obs["front_rgb_list"][i],
            obs["wrist_rgb_list"][i],
            is_video_demo=(is_video_demo_fn(i) if is_video_demo_fn else False),
        )
        for i in range(n)
    ]


def _first_execution_step(episode_data: h5py.Group) -> int:
    """Return the index of the first non-video-demo timestep."""
    step_idx = 0
    while f"timestep_{step_idx}" in episode_data:
        ts = episode_data[f"timestep_{step_idx}"]
        if "info" in ts and "is_video_demo" in ts["info"]:
            if not ts["info"]["is_video_demo"][()]:
                break
        step_idx += 1
    return step_idx


def _load_action_from_timestep(
    episode_data: h5py.Group, step_idx: int, action_space_type: str
) -> Union[np.ndarray, Dict[str, Any]]:
    timestep_key = f"timestep_{step_idx}"
    action_group = episode_data[timestep_key]["action"]

    if action_space_type == "joint_angle":
        action_data = action_group["joint_action"][()]
        return np.asarray(action_data, dtype=np.float32)
    elif action_space_type == "ee_pose":
        action_data = action_group["eef_action"][()]
        return np.asarray(action_data, dtype=np.float32)
    elif action_space_type == "waypoint":
        action_data = action_group["waypoint_action"][()]
        return np.asarray(action_data, dtype=np.float32)
    elif action_space_type == "multi_choice":
        # Read JSON-encoded choice_action and parse it, following EpisodeDatasetResolver._extract_choice_action.
        raw = action_group["choice_action"][()]
        # Decode bytes -> str
        if isinstance(raw, np.ndarray):
            raw = np.reshape(raw, -1)[0]
        if isinstance(raw, (bytes, np.bytes_)):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"choice_action payload is not a dict at timestep_{step_idx}")
        label = payload.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"choice_action missing valid 'label' at timestep_{step_idx}")
        command: Dict[str, Any] = {"label": label}
        # point stored as [y, x]; normalize to execution format [x, y]
        point = payload.get("point")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                command["point"] = [int(point[1]), int(point[0])]
            except (TypeError, ValueError):
                pass
        serial = payload.get("serial_number")
        if serial is not None:
            try:
                command["serial_number"] = int(serial)
            except (TypeError, ValueError):
                pass
        return command
    else:
        raise ValueError(f"Unknown action space type: {action_space_type}")


def _save_video(
    frames: list[np.ndarray],
    task_id: str,
    episode_idx: int,
    task_goal: str,
    outcome: str,
    action_space_type: str,
) -> Path:
    video_dir = Path(REPLAY_VIDEO_DIR) / action_space_type
    video_dir.mkdir(parents=True, exist_ok=True)
    name = f"{outcome}_{task_id}_ep{episode_idx}_{task_goal}.mp4"
    path = video_dir / name
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS)
    return path


def _get_episode_indices(data: h5py.File) -> list[int]:
    return sorted(
        int(key.split("_")[1])
        for key in data.keys()
        if key.startswith("episode_")
    )


def process_episode(
    env_data: h5py.File,
    episode_idx: int,
    task_id: str,
    action_space_type: ActionSpaceType,
) -> None:
    """Replay one episode from HDF5 data, record frames, and save a video."""
    episode_data = env_data[f"episode_{episode_idx}"]
    task_goal = episode_data["setup"]["task_goal"][()][0].decode()
    total_steps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
    step_idx = _first_execution_step(episode_data)


    env = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset="train",
        action_space=action_space_type,
        gui_render=GUI_RENDER,
    ).make_env_for_episode(
        episode_idx,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )
    
    print(f"\nTask: {task_id}, Episode: {episode_idx}, ",
          f"Seed: {env.unwrapped.seed}, Difficulty: {env.unwrapped.difficulty}")
    print(f"Task goal: {task_goal}")
    print(f"Execution starts at step {step_idx}")


    obs, _ = env.reset()
    frames = _extract_frames(
        obs, is_video_demo_fn=lambda i, n=len(obs["front_rgb_list"]): i < n - 1
    )

    outcome = "unknown"
    while step_idx < total_steps:
        action = _load_action_from_timestep(episode_data, step_idx, action_space_type)
        try:
            obs, _, terminated, truncated, info = env.step(action)
            frames.extend(_extract_frames(obs))
        except Exception as e:
            print(f"Error at step {step_idx}: {e}")
            break

        if GUI_RENDER:
            env.render()
        if terminated or truncated:
            outcome = info.get("status", "unknown")
            print(f"Outcome: {outcome}")
            break
        step_idx += 1

    env.close()
    path = _save_video(frames, task_id, episode_idx, task_goal, outcome, action_space_type)
    print(f"Saved video to {path}\n")


def replay(
    h5_data_dir: str = "/data/hongzefu/data_0225",
    action_space_type: ActionSpaceType = "eef_action",
    replay_number: int = 1,
) -> None:
    """Replay episodes from HDF5 dataset files and save rollout videos."""
    #for task_id in BenchmarkEnvBuilder.get_task_list():
    for task_id in ["VideoUnmaskSwap"]:
        file_path = Path(h5_data_dir) / f"record_dataset_{task_id}.h5"

        if not file_path.exists():
            print(f"Skipping {task_id}: file not found: {file_path}")
            continue

        with h5py.File(file_path, "r") as data:
            episode_indices = _get_episode_indices(data)
            for episode_idx in episode_indices[:min(replay_number, len(episode_indices))]:
                process_episode(data, episode_idx, task_id, action_space_type)


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
