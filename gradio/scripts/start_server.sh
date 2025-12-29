#!/bin/bash
# 启动脚本 - 在后台运行服务器并保存所有日志

# 获取脚本所在目录，然后定位到 gradio 目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRADIO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$GRADIO_DIR"

# Micromamba 环境路径
MICROMAMBA_ENV="/data/hongzefu/maniskillenv1114"

# 日志目录（放在 gradio 目录中）
LOG_DIR="$GRADIO_DIR/logs"
PID_FILE="$GRADIO_DIR/server.pid"
# 合并日志文件（包含所有输出：标准输出 + 错误输出）
# 所有日志都会完整显示在这个文件中
LOG_FILE="$LOG_DIR/server.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 检查是否已经在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  服务器已经在运行中 (PID: $OLD_PID)"
        echo "   如需重启，请先运行: ./scripts/stop_server.sh"
        exit 1
    else
        echo "清理旧的 PID 文件..."
        rm -f "$PID_FILE"
    fi
fi

# 检查 micromamba 环境是否存在
if [ ! -d "$MICROMAMBA_ENV" ]; then
    echo "❌ 错误: Micromamba 环境不存在: $MICROMAMBA_ENV"
    exit 1
fi

# 检查 Python 可执行文件
PYTHON_EXE="$MICROMAMBA_ENV/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    echo "❌ 错误: Python 可执行文件不存在: $PYTHON_EXE"
    exit 1
fi

# 启动服务器
echo "🚀 正在启动服务器..."
echo "   Micromamba 环境: $MICROMAMBA_ENV"
echo "   Python 可执行文件: $PYTHON_EXE"
echo "   工作目录: $GRADIO_DIR"
echo "   完整日志文件: $LOG_FILE (包含所有输出)"
echo ""

# 使用环境中的 Python 直接运行服务器
# 使用 nohup 在后台运行，并将所有输出重定向到日志文件
# 设置环境变量以确保使用环境中的包
# 使用 unbuffered 模式 (-u) 和 PYTHONUNBUFFERED=1 确保输出立即写入，不缓冲
# 将标准输出和错误输出合并到一个文件 (2>&1)，这样所有日志都会完整显示
nohup env PATH="$MICROMAMBA_ENV/bin:$PATH" PYTHONUNBUFFERED=1 "$PYTHON_EXE" -u "$GRADIO_DIR/main.py" >> "$LOG_FILE" 2>&1 &

# 保存进程ID
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

# 等待一下，检查进程是否成功启动
sleep 3

if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "✅ 服务器已成功启动！"
    echo "   PID: $SERVER_PID"
    echo "   Micromamba 环境: $MICROMAMBA_ENV"
    echo "   完整日志文件: $LOG_FILE"
    echo ""
    echo "📋 常用命令:"
    echo "   查看实时日志（完整输出）: tail -f $LOG_FILE"
    echo "   或使用交互式脚本: ./scripts/view_logs.sh"
    echo "   停止服务器: ./scripts/stop_server.sh"
    echo "   查看进程状态: ./scripts/status_server.sh"
    echo ""
    echo "💡 提示:"
    echo "   - 所有日志（标准输出和错误输出）都保存在 $LOG_FILE"
    echo "   - 即使关闭SSH连接，服务器也会继续运行"
    echo "   - 使用 PYTHONUNBUFFERED=1 确保日志实时写入"
else
    echo "❌ 服务器启动失败！"
    echo "   请查看完整日志: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

