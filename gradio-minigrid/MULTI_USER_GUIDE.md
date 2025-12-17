# 多用户支持与数据保存说明

## 🔐 多用户实现原理

### 1. Gradio 的 Session 机制

Gradio 使用 **Session ID** 来区分不同的用户连接。每个打开网页的用户都会获得一个唯一的 Session ID。

### 2. `gr.State()` 组件的作用

```python
session_state = gr.State()
```

- **`gr.State()`** 是一个隐藏组件，不在 UI 上显示
- 它为**每个 Session ID** 维护一份独立的状态数据
- 当用户 A 点击按钮时，只会访问和更新用户 A 的 `session_state`
- 当用户 B 点击按钮时，只会访问和更新用户 B 的 `session_state`
- **完全隔离，互不干扰**

### 3. 工作流程

```
用户 A 打开网页
  ↓
Gradio 分配 Session ID: "abc123"
  ↓
demo.load() 触发 → init_session()
  ↓
创建 SessionManager 实例 A（包含独立的 Gym 环境）
  ↓
存入 session_state["abc123"] = SessionManager_A
  ↓
用户 A 点击按钮 → 从 session_state["abc123"] 取出 SessionManager_A
  ↓
执行动作 → 更新 SessionManager_A 的状态
  ↓
返回结果 → 更新用户 A 的界面

用户 B 打开网页（同时）
  ↓
Gradio 分配 Session ID: "xyz789"
  ↓
创建 SessionManager 实例 B（独立的 Gym 环境）
  ↓
存入 session_state["xyz789"] = SessionManager_B
  ↓
用户 B 的操作只影响 SessionManager_B
```

### 4. 关键代码位置

**初始化（每个用户首次加载页面时）：**
```python
demo.load(fn=init_session, outputs=[session_state, ...])
```

**动作处理（每次点击按钮）：**
```python
btn_fwd.click(lambda s: on_action(s, "forward"), 
              inputs=[session_state],  # 输入：当前用户的 session
              outputs=[session_state, ...])  # 输出：更新后的 session
```

## 💾 数据保存机制

### 1. 保存时机

数据在以下情况会被保存：
- ✅ **游戏结束时**（`is_done = True`）
- ✅ 自动保存，无需手动操作

### 2. 保存位置

```
data/
  └── experiment_logs.jsonl
```

### 3. 数据格式（JSON Lines）

每行是一个完整的 JSON 对象，代表一次游戏会话：

```json
{
  "uid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "env_id": "MiniGrid-Empty-8x8-v0",
  "seed": null,
  "total_steps": 45,
  "total_reward": 0.9,
  "duration": 120.5,
  "history": [
    {"step": 1, "action": "forward", "reward": 0.0, "timestamp": 1234567890.1},
    {"step": 2, "action": "left", "reward": 0.0, "timestamp": 1234567890.2},
    ...
  ],
  "finished": true,
  "logged_at": "2025-12-17T10:30:45.123456"
}
```

### 4. 数据字段说明

| 字段 | 说明 |
|------|------|
| `uid` | 用户唯一标识符（UUID） |
| `env_id` | 环境ID |
| `seed` | 随机种子（如果有） |
| `total_steps` | 总步数 |
| `total_reward` | 累计奖励 |
| `duration` | 游戏时长（秒） |
| `history` | 每一步的详细记录 |
| `finished` | 是否完成游戏 |
| `logged_at` | 保存时间戳 |

### 5. 线程安全

使用 `threading.Lock()` 确保多用户同时写入时不会损坏文件：

```python
with lock:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(session_data) + "\n")
```

## 📊 数据查看功能

### 在网页界面查看

1. 点击 **"📊 数据统计"** 标签页
2. 点击 **"🔄 刷新数据"** 按钮
3. 查看：
   - 总体统计（总会话数、总步数、平均奖励等）
   - 最近10条记录

### 命令行查看

```bash
# 查看所有记录
cat data/experiment_logs.jsonl

# 查看最后10条
tail -n 10 data/experiment_logs.jsonl

# 使用 jq 格式化查看（如果安装了 jq）
cat data/experiment_logs.jsonl | jq .
```

### Python 脚本分析

```python
import json

# 读取所有数据
sessions = []
with open('data/experiment_logs.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            sessions.append(json.loads(line))

# 分析数据
print(f"总会话数: {len(sessions)}")
print(f"平均步数: {sum(s['total_steps'] for s in sessions) / len(sessions)}")
```

## 🔍 用户识别

### 在界面上

- 每个用户会看到自己的 **用户ID**（UUID 的前8位）
- 显示在右上角的 "用户ID" 文本框中
- 例如：`用户: a1b2c3d4`

### 在日志中

- 完整的 UUID 保存在 `uid` 字段中
- 可以通过 UUID 追踪特定用户的所有游戏记录

## ⚠️ 注意事项

### 1. Session 生命周期

- 用户关闭浏览器标签页 → Session 会在 1 小时后自动清理
- 用户刷新页面 → Session 保持不变（如果未过期）
- 服务器重启 → 所有 Session 丢失，用户需要刷新页面重新初始化

### 2. 数据持久化

- ✅ 游戏数据保存在文件中，**服务器重启不会丢失**
- ❌ Session 状态（当前游戏进度）在服务器内存中，**重启会丢失**

### 3. 性能考虑

- Gradio 默认最多保存 10,000 个 Session
- 超过限制时，最旧的 Session 会被自动清理
- 可以通过 `demo.launch(state_session_capacity=...)` 调整

## 🚀 扩展建议

### 1. 实时保存（可选）

如果想在每一步都保存数据，可以修改 `on_action`：

```python
# 在每次 step 后都保存
logger.log_session(session.export_data())
```

### 2. 数据库存储（可选）

对于大量用户，可以考虑使用数据库：

```python
# 使用 SQLite
import sqlite3
conn = sqlite3.connect('data/sessions.db')
# ... 插入数据
```

### 3. 用户认证（可选）

如果需要用户登录：

```python
# 添加用户名输入
username = gr.Textbox(label="用户名")
# 在 SessionManager 中保存用户名
```

## 📝 总结

- ✅ **多用户支持**：通过 `gr.State()` 自动实现，每个用户独立环境
- ✅ **数据保存**：游戏结束时自动保存到 JSONL 文件
- ✅ **数据查看**：网页界面和命令行都可以查看
- ✅ **线程安全**：使用锁机制防止文件损坏
- ✅ **用户识别**：通过 UUID 区分不同用户
