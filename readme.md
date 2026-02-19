## RoboMME: A Robotic Benchmark for Memory-Augmented Manipulation

![Robomme bench](assets/robomme_bench.jpg)

- [Announcements](#announcements)
- [Installation](#installation)
- [Running Examples](#running-examples)
- [Tasks](#tasks)
- [Benchmark](#benchmark)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## 📢 Announcements

[02/2026] We release RoboMME! It's a cognitive-motivated large-scale robotic benchmark for memory-augmented manipulation, spanning 4 task suites with a total of 16 carefully designed tasks.

## 📦 Installation

After cloning the repo, install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
uv pip install -e .
```

## 🚀 Running Examples

Start an environment with a specified setup:

```bash
uv run scripts/run_example.py --action-space-type joint_angle --dataset test --env-id PickXtimes --episode-idx 0
```

This generates a rollout video in the `sample_run_videos` directory.

We provide four action types: `joint_action`, `ee_pose`, `keypoint`, and `multi_choice`. Use `joint_action` or `ee_pose` for continuous action prediction, `keypoint` for discrete waypoint actions, and `multi_choice` for VideoQA-style evaluation.

> **Note:** Currently, only `joint_action` is verified. Please use it rather than other types.

## 📁 Benchmark

### 📥 Training Data

Training data can be downloaded [here](https://huggingface.co/Yinpei/data_0214). There are 1,600 demonstrations in total (100 per task). The HDF5 format is described in [doc/h5_data_format.md](doc/h5_data_format.md).

> **Note:** Currently, the training data is not finalized and may differ from the documentation.

After downloading, replay the dataset for a sanity check:

```bash
uv run scripts/dataset_replay.py --h5-data-dir <your_downloaded_data_dir> --action-space-type joint_angle
```

### 📊 Evaluation

To evaluate on the val or test set, set the `dataset` argument of `BenchmarkEnvBuilder`:

```python
env_builder = BenchmarkEnvBuilder(
    env_id=task_id,
    dataset="test",  # or "val"
    ...
)
```

Each split has 50 episodes. 

### 🔧 Data Generation

You can also re-generate your own HDF5 data using scripts in `scripts/dev/`. Details on parallel generation TBD (@hongze).

```bash
uv run scripts/dev/xxxx
```

### 🎮 Play with Online Demo

Start the Gradio GUI to try the demo (@hongze). (Command TBD.)


## 🧠 Model Training

The [MME-VLA-Suite](https://github.com/RoboMME/MME-VLA-Suite) repo provides VLA model training and evaluation. Please check it out.

> **Note:** Currently, environment spawning is set up for imitation learning. We are working on extending it to support more general parallel environments for reinforcement learning.

## 🎯 Tasks

We have four task suites, each with 4 tasks:

| Suite      | Focus             | Task ID                                                                 |
| ---------- | ----------------- | --------------------------------------------------------------------- |
| Counting   | Temporal memory   | BinFill, PickXtimes, SwingXtimes, StopCube                            |
| Permanence | Spatial memory    | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap         |
| Reference  | Object memory     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder         |
| Imitation  | Procedural memory | MoveCube, InsertPeg, PatternLock, RouteStick                          |

All tasks are defined in `src/robomme/robomme_env`.

## 🔧 Troubleshooting

**Q1: RuntimeError: Create window failed: Renderer does not support display.**

A1: Use a physical display or set up a virtual display for GUI rendering (e.g. install a VNC server and set the `DISPLAY` variable correctly).

**Q2: Failure related to Vulkan installation.**

A2: We recommend reinstalling the NVIDIA driver and Vulkan packages. We use NVIDIA driver 570.211.01 and Vulkan 1.3.275. If it still does not work, you can switch to CPU rendering:

```python
os.environ['SAPIEN_RENDER_DEVICE'] = 'cpu'
os.environ['MUJOCO_GL'] = 'osmesa'
```



## 🙏 Acknowledgements

This work was supported in part by NSF SES-2128623, NSF CAREER #2337870, NSF NRI #2220876, NSF NAIRR250085. We would also like to thank the wonderful [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main) codebase from Physical-Intelligence.


## 📄 Citation

```
...
```