# v0.2.3 数据集格式与 `evaluate.py` 输出说明

本文档面向使用者，包含两部分内容：

1. 数据集文件格式（HDF5）说明。
2. 运行 `/data/hongzefu/robomme-v0.2b6/public_scripts/evaluate.py` 后可直接获得的信息与结果。

---

## 1. 数据集格式

### 1.1 文件与目录结构

```text
<dataset_root>/
  hdf5_files/
    <EnvID>_ep<Episode>_seed<Seed>.h5   # 每个 episode 一个文件
  videos/                               # （可选）录制视频
```

- 数据由录制 wrapper 写入 `hdf5_files/`。
- 通常按 `EnvID + episode + seed` 命名，便于回放和检索。

### 1.2 HDF5 层级总览

```text
env_<EnvID>/
  episode_<N>/
    record_timestep_<K>/
      obs/
      action/
      info/
    setup/
```

- `record_timestep_<K>/`：时间步级别数据。
- `setup/`：episode 级别一次性配置。

### 1.3 `obs/` 字段（观测）

| 字段 | 类型 / 形状 | 说明 |
|---|---|---|
| `front_camera_rgb` | `uint8 (H, W, 3)` | 正面相机 RGB |
| `wrist_camera_rgb` | `uint8 (H, W, 3)` | 腕部相机 RGB |
| `front_camera_depth` | `float (H, W, 1)` | 正面相机深度 |
| `wrist_camera_depth` | `float (H, W, 1)` | 腕部相机深度 |
| `joint_state` | `float64 (9,)` | 关节状态（7 关节 + 2 夹爪） |
| `eef_state` | `float64 (6,)` | 末端状态 `[x, y, z, roll, pitch, yaw]` |
| `gripper_state` | `float64 (2,)` | 夹爪状态 |
| `gripper_close` | `bool` | 夹爪是否闭合 |
| `eef_velocity` | `float64 (6,)` | 末端线速度 + 角速度 |
| `front_camera_segmentation` | `int (H, W, ...)` | 正面相机分割掩码 |
| `front_camera_segmentation_result` | 同上 | 过滤后的分割结果 |
| `front_camera_extrinsic` | `float (4, 4)` | 正面相机外参 |
| `wrist_camera_extrinsic` | `float (4, 4)` | 腕部相机外参 |

### 1.4 `action/` 字段（动作）

| 字段 | 类型 / 形状 | 说明 |
|---|---|---|
| `joint_action` | `float64 (8,)` 或 `str "None"` | 关节空间动作 |
| `eef_action` | `float64 (7,)` | 末端动作 `[x, y, z, roll, pitch, yaw, gripper]` |
| `keypoint_action` | `float64 (7,)` | 关键点动作 `[x, y, z, roll, pitch, yaw, gripper]` |

### 1.5 `info/` 字段（元信息）

| 字段 | 类型 | 说明 |
|---|---|---|
| `record_timestep` | `int` | 当前帧编号 |
| `simple_subgoal` | `bytes (UTF-8)` | 子目标文本（planner 视角） |
| `simple_subgoal_online` | `bytes (UTF-8)` | 子目标文本（online 视角） |
| `grounded_subgoal` | `bytes (UTF-8)` | grounding 后子目标文本 |
| `grounded_subgoal_online` | `bytes (UTF-8)` | online grounding 子目标文本 |
| `is_video_demo` | `bool` | 是否演示阶段帧 |
| `is_keyframe` | `bool` | 是否关键帧 |

### 1.6 `setup/` 字段（episode 配置）

| 字段 | 类型 | 说明 |
|---|---|---|
| `seed` | `int` | 环境种子 |
| `difficulty` | `str` | 难度等级 |
| `task_goal` | `str` | 自然语言任务 |
| `front_camera_intrinsic` | `float (3, 3)` | 正面相机内参 |
| `wrist_camera_intrinsic` | `float (3, 3)` | 腕部相机内参 |
| `subgoal_list/subgoal_<i>_name` | `bytes` | 第 `i` 个子目标名称 |
| `subgoal_list/subgoal_<i>_demonstration` | `bool` | 第 `i` 个子目标是否演示阶段 |

---

## 2. `evaluate.py` 运行后可获得什么

文件：`/data/hongzefu/robomme-v0.2b6/public_scripts/evaluate.py`

### 2.1 脚本用途

`evaluate.py` 是统一评测入口，支持 4 种 `ACTION_SPACE`：

- `joint_angle`
- `ee_pose`
- `keypoint`
- `oracle_planner`

脚本会遍历任务与 episode，执行交互并输出每个 episode 的结果。

### 2.2 终端可见输出

运行命令：

```bash
python /data/hongzefu/robomme-v0.2b6/public_scripts/evaluate.py
```

你会在终端看到：

- 任务列表：`Running envs: [...]`
- 当前动作空间：`Using action_space: ...`
- 每个任务 episode 数：`episode_count from metadata: ...`
- 每步动作：`dummy_action: ...`
- episode 结束结论：`success` / `failed` / `steps exceeded`

### 2.3 返回结构（`reset/step`）

先说明“batch”含义：

- 这里的 batch **不是**“多个 `env_id` 并行”。
- `evaluate.py` 是按 `env_id`、按 episode 串行执行：一次只跑一个环境实例。
- 这里的 batch 维度表示“**一次 `reset()` 或一次 `step()` 内部产生了多少帧（多少个底层 env.step）**”。

`reset()` 返回：

```python
obs_batch, info_batch = env.reset()
```

`step()` 返回：

```python
obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(action)
```

| 返回项 | 说明 | 常见形式 |
|---|---|---|
| `obs_batch` | 观测批量 | `dict[str, list]` |
| `info_batch` | 信息批量 | `dict[str, list]` |
| `reward_batch` | 奖励批量 | 一维张量 |
| `terminated_batch` | 终止标记批量 | 一维布尔张量 |
| `truncated_batch` | 截断标记批量 | 一维布尔张量 |

#### 2.3.1 哪些 `env_id` / `action_space` 会产生多帧 batch

| 场景 | `step()` 的 batch 长度 `N` | 说明 |
|---|---|---|
| `joint_angle` | 通常 `N=1` | 单次动作通常只对应一次底层 step |
| `ee_pose` | 通常 `N=1` | 外层先做 IK，再执行一次底层 step |
| `keypoint` | 常见 `N>=1`（经常大于 1） | 一次 keypoint 可能触发“移动 + 开/闭夹爪”等多次底层 step |
| `oracle_planner` | 常见 `N>=1`，也可能 `N=0` | 一次语义指令可能触发多步规划执行；无效指令可返回空 batch |

#### 2.3.2 `reset()` 的 batch 来自哪里

- `reset()` 不是单帧返回。
- 返回的是“演示轨迹 batch + 初始一步 batch”合并结果，因此通常 `N>1`。
- 该行为与 `env_id` 有关（任务演示轨迹长短不同），但接口格式一致。

#### 2.3.3 batch 结果具体是什么

设本次调用得到长度为 `N` 的 batch：

- `obs_batch[k]`：长度为 `N` 的列表，第 `i` 项是第 `i` 帧该观测键的值。
- `info_batch[k]`：长度为 `N` 的列表，第 `i` 项是第 `i` 帧该信息键的值。
- `reward_batch`：形状 `[N]` 的 `torch.float32`。
- `terminated_batch`：形状 `[N]` 的 `torch.bool`。
- `truncated_batch`：形状 `[N]` 的 `torch.bool`。

在 `evaluate.py` 中，通常用最后一帧作为“当前状态”：

```python
info = {k: v[-1] for k, v in info_batch.items()}
terminated = bool(terminated_batch[-1].item())
truncated = bool(truncated_batch[-1].item())
```

### 2.4 `obs_batch` 键

| 键 | 含义 | 常见内容 |
|---|---|---|
| `maniskill_obs` | 底层环境原始观测 | 原始观测字典（按底层接口组织） |
| `front_camera` | 正面相机 RGB 帧 | 图像帧（常见 `(H, W, 3)`） |
| `wrist_camera` | 腕部相机 RGB 帧 | 图像帧（常见 `(H, W, 3)`） |
| `front_camera_depth` | 正面相机深度帧 | 深度图（常见 `(H, W, 1)`） |
| `wrist_camera_depth` | 腕部相机深度帧 | 深度图（常见 `(H, W, 1)`） |
| `end_effector_pose` | 末端执行器位姿 | `[x, y, z, roll, pitch, yaw]` |
| `joint_states` | 机器人关节状态 | 关节向量（常见 9 维） |
| `velocity` | 末端速度 | 线速度 + 角速度（通常各 3 维） |

说明：`obs_batch` 的每个键都是“按时间帧打包”的列表；取当前帧通常用 `obs_batch[key][-1]`。

### 2.5 `info_batch` 键

| 键 | 含义 | 常见内容 |
|---|---|---|
| `language_goal` | 当前任务目标文本 | 自然语言描述字符串 |
| `subgoal` | 当前子目标文本 | 当前阶段要完成的动作描述 |
| `subgoal_grounded` | grounding 后子目标 | 带对象定位语义的子目标文本 |
| `available_options` | 当前可选动作集合 | 列表，元素如 `{"action": str, "need_parameter": bool}` |
| `front_camera_extrinsic_opencv` | 正面相机外参 | 相机外参矩阵 |
| `front_camera_intrinsic_opencv` | 正面相机内参 | 相机内参矩阵 |
| `wrist_camera_extrinsic_opencv` | 腕部相机外参 | 相机外参矩阵 |
| `wrist_camera_intrinsic_opencv` | 腕部相机内参 | 相机内参矩阵 |
| `success` | 成功标记 | 当前帧/当前批次是否成功 |
| `fail` | 失败标记 | 当前帧/当前批次是否失败 |

说明：`info_batch` 同样是按时间帧打包；脚本里常用 `info = {k: v[-1] for k, v in info_batch.items()}` 取当前状态。

### 2.6 不同 `ACTION_SPACE` 的输入动作

| `ACTION_SPACE` | 输入动作形式 |
|---|---|
| `joint_angle` | 关节动作向量 |
| `ee_pose` | `[x, y, z, roll, pitch, yaw, gripper]` |
| `keypoint` | 前 7 维同 `ee_pose` 定义 |
| `oracle_planner` | 指令字典（如 `{"action": "...", "point": [x, y]}`） |

### 2.7 最终评测结论

每个 episode 会有明确结论：

- `success`：任务完成
- `failed`：任务失败
- `steps exceeded`：超过 `MAX_STEPS` 被截断（默认 `MAX_STEPS=3000`）

---

## 3. 用户侧快速理解

1. 用第 1 节理解数据集如何组织与读取。
2. 用第 2 节理解 `evaluate.py` 运行时可获得的数据和结果。
3. 实际使用时，可把第 1 节（离线数据）与第 2 节（在线评测）联合使用。
