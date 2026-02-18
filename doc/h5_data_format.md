# HDF5 Training Dataset Format

Structure inside each `record_dataset_<EnvID>.h5` file:

```text
episode_<N>/
  timestep_<K>/
    obs/
    action/
    info/
  setup/
```
- `timestep_<K>/`: per-timestep data.
- `setup/`: episode-level configuration.

## `setup/` fields (episode configuration)

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `int` | Environment seed |
| `difficulty` | `str` | Difficulty level |
| `task_goal` | `str` | Natural-language task description |
| `front_camera_intrinsic` | `float32 (3, 3)` | Front camera intrinsic matrix |
| `wrist_camera_intrinsic` | `float32 (3, 3)` | Wrist camera intrinsic matrix |
| `available_multi_choices` | `str` | Current available options for multi-choice action |
| `fail_recover_mode` | `str` | Fail recovery mode |
| `fail_recover_seed_anchor` | `int64` | Seed anchor used for fail recovery |

## `obs/` fields (observations)

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `front_rgb` | `uint8 (H, W, 3)` | Front camera RGB |
| `wrist_rgb` | `uint8 (H, W, 3)` | Wrist camera RGB |
| `front_depth` | `int16 (H, W, 1)` | Front camera depth |
| `wrist_depth` | `int16 (H, W, 1)` | Wrist camera depth |
| `joint_state` | `float32 (7,)` | Joint state (7 joints) |

| `eef_state_raw/pose` | `float32 (3,)` | End-effector raw position `[x, y, z]` |
| `eef_state_raw/quat` | `float32 (4,)` | End-effector raw orientation (quaternion) |
| `eef_state_raw/rpy` | `float32 (3,)` | End-effector raw orientation `[roll, pitch, yaw]` |
| `gripper_state` | `float32 (2,)` | Gripper state |
| `is_gripper_close` | `bool` | Whether gripper is closed |
| `front_camera_extrinsic` | `float32 (3, 4)` | Front camera extrinsic matrix |
| `wrist_camera_extrinsic` | `float32 (3, 4)` | Wrist camera extrinsic matrix |

## `action/` fields (actions)

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `joint_action` | `float64 (8,)` or `str "None"` | Joint-space action |
| `eef_action` | `float64 (7,)` | End-effector action `[x, y, z, roll, pitch, yaw, gripper]` |
| `eef_action_raw/pose` | `float64 (3,)` | End-effector raw action position `[x, y, z]` |
| `eef_action_raw/quat` | `float64 (4,)` | End-effector raw action orientation (quaternion) |
| `eef_action_raw/rpy` | `float64 (3,)` | End-effector raw action orientation `[roll, pitch, yaw]` |
| `keypoint_action` | `float64 (7,)` | End-effector action at keyframes `[x, y, z, roll, pitch, yaw, gripper]`; only present at keyframe steps |
| `choice_action` | `str` | dict string for multi-choice selection and optional grounded pixel, e.g. `{"choice": "A", "point": [x, y]}`|

## `info/` fields (metadata)

| Field | Type | Description |
|-------|------|-------------|
| `simple_subgoal` | `bytes (UTF-8)` | Simple subgoal text (built-in planner view) |
| `simple_subgoal_online` | `bytes (UTF-8)` | Simple subgoal text (online view; may be earlier than planner view) |
| `grounded_subgoal` | `bytes (UTF-8)` | Grounded subgoal text |
| `grounded_subgoal_online` | `bytes (UTF-8)` | Online grounded subgoal text |
| `is_video_demo` | `bool` | Whether this frame is from the conditioning video input before execution |
| `is_keyframe` | `bool` | Whether this is a keyframe (i.e. a boundary between subtasks) |
