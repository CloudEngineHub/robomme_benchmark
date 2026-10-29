# RoboMME-Plus training data

The finalized split contains **50 successful episodes, 56,462 frames at 20 Hz**:
one easy, medium and hard episode for 15 tasks, and five MoveCube methods.
Download [H5](https://huggingface.co/datasets/Yinpei/robomme_ood_data_h5) or
[LeRobot v3](https://huggingface.co/datasets/Yinpei/robomme_ood_data_lerobot).
Both formats represent the same recordings.

## Replay

Replay saved native actions through the simulator and export front/wrist videos:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/plus_eval/dataset_replay.py \
  --h5-data-dir data/plus_train/h5 --tasks PatternLock --episodes 0
```

Omit `--tasks` and `--episodes` to replay all training episodes. Videos are saved
to `runs/plus_dataset_replay/<Task>/episode_<k>.mp4` at the recorded control rate;
demonstration frames have a red border. The script checks episode metadata and
stops each replay when the simulator terminates. It exits nonzero if a replay fails.

## Seeds and episode mapping

The source of training seeds is
[`src/robomme_plus/env_metadata/train/`](../src/robomme_plus/env_metadata/train/),
in `record_dataset_<Task>_metadata.json`. Record `k` corresponds to H5
`episode_<k>`; `BenchmarkEnvBuilder(task, dataset="train")` loads it.
Dataset copies of `train_plan.json`, `train_seeds.csv` and `metadata/train/`
include the task, episode, seed, difficulty/method and measured execution length.
LeRobot `meta/source_episodes.json` maps global episode indices back to these rows.

## Conventions

- Observations precede their paired actions: `obs[t+1]` follows `action[t]`.
- Conditioning-video frames come first (`is_video_demo`); execution follows.
  Return-to-home motion between phases is unrecorded. The native step index can jump.
- Evaluator demo observations follow their actions, so their image sequence is
  shifted one frame relative to H5. The first execution observation matches the
  final reset observation.
- Positions use metres, angles radians, quaternions `wxyz`, and extrinsic `xyz` Euler angles.
- VideoPlace recordings include release and at least 5 cm of upward gripper retreat.
- Some MoveCube recordings retain settling frames after first success. A public
  evaluator replay stops at first success.

## H5

Each `h5/record_dataset_<Task>.h5` contains `episode_<k>/setup` and
`timestep_<t>/{obs,action,info}`.

| Group/field | Content |
| --- | --- |
| `setup` | Goal, seed, difficulty, episode metadata, camera intrinsics and phase lengths |
| `obs/front_rgb`, `obs/wrist_rgb` | uint8 RGB, 256×256×3 |
| `obs/front_depth`, `obs/wrist_depth` | int16 depth in millimetres, 256×256×1 |
| `obs/*_camera_extrinsic` | World-to-camera `[R\|t]`, 3×4 |
| `obs/joint_state`, `obs/gripper_state` | 7 arm joints; two finger widths in metres |
| `obs/eef_state` | Native TCP `[x,y,z,roll,pitch,yaw]` |
| `action/native_joint_action` | Exact simulator command: 7 values for sticks, 8 normally, 15 for MoveCube |
| `action/joint_action` | Standard 8-value joint target + gripper command |
| `action/eef_action` | Forward kinematics of the target, plus gripper |
| `action/waypoint_action` | Current 7-value keyframe target when `waypoint_action_valid` is true |
| `info` | Subgoal text, grounding, demo/completion/boundary flags and native control step |

MoveCube native commands are `[q(7), dq(7), gripper]`; other tasks omit velocities.
Action gripper commands use −1 closed and +1 open. Observed finger widths use
0–0.04 m. Choice actions are unused. Use native actions for physical replay.

## LeRobot v3

The dataset root contains `meta/`, `data/` and `videos/`. Each episode has one
Parquet file and front/wrist H264 videos. Each row corresponds to one H5 timestep.

| Feature | Meaning |
| --- | --- |
| `observation.images.front`, `.wrist` | 256×256 RGB video at 20 FPS |
| `observation.state` | 7 joint positions + mean finger width in metres |
| `action` | 7 joint targets + gripper command |
| `action.native_sequence` | 75 values: reshape to 5×15, use the first `native_action_count` rows |
| `native_action_count`, `native_frame_index` | Always 1; corresponding H5 frame |
| `is_video_demo`, `control_freq` | Phase flag; 20 Hz |
| `task_index` | Goal text in `meta/tasks.parquet` |
| `subtask*_index` | Plain/grounded subgoal variants in `meta/subtasks.parquet` |
| `episode_index`, `frame_index`, `timestamp`, `index` | Episode, local frame, seconds and global row |
| `next.done` | True only on the last row |

`meta/native_action_format.json` describes native command packing. LeRobot omits
depth, camera extrinsics, TCP poses and waypoints; use H5 for these. H264 images
are lossy, so they are not pixel-identical to the H5 RGB arrays.

All 50 recordings passed public-evaluator execution replay with zero joint
trajectory difference. All 117 format checks and both dataset-loader checks passed.
Published manifests record file checksums and the matching source revision.
