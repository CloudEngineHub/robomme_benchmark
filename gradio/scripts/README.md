# 服务器管理脚本

本目录包含用于管理 HistoryBench 服务器的脚本。

## 脚本说明

- **start_server.sh** - 启动服务器（后台运行，使用 Micromamba 环境）
- **stop_server.sh** - 停止服务器
- **status_server.sh** - 查看服务器状态
- **view_logs.sh** - 查看日志（交互式）

## 快速开始

### 启动服务器

```bash
cd /data/hongzefu/historybench-v5.6/gradio
./scripts/start_server.sh
```

### 查看日志

```bash
# 使用交互式脚本
./scripts/view_logs.sh

# 或直接使用 tail
tail -f logs/server.log
```

### 查看状态

```bash
./scripts/status_server.sh
```

### 停止服务器

```bash
./scripts/stop_server.sh
```

## 配置

脚本会自动使用以下配置：

- **Micromamba 环境**: `/data/hongzefu/maniskillenv1114`
- **工作目录**: `/data/hongzefu/historybench-v5.6/gradio`
- **日志目录**: `logs/`
- **PID 文件**: `server.pid`

## 注意事项

1. 所有脚本必须从 `gradio` 目录运行（使用相对路径 `./scripts/xxx.sh`）
2. 日志文件保存在 `gradio/logs/` 目录中
3. PID 文件保存在 `gradio/` 目录中
4. 即使关闭 SSH 连接，服务器也会继续运行

