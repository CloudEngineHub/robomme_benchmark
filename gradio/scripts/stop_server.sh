#!/bin/bash
# 停止脚本 - 优雅地停止服务器

# 获取脚本所在目录，然后定位到 gradio 目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRADIO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$GRADIO_DIR/server.pid"

# 检查PID文件是否存在
if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  未找到 PID 文件，服务器可能未运行"
    exit 1
fi

# 读取PID
SERVER_PID=$(cat "$PID_FILE")

# 检查进程是否存在
if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "⚠️  进程 $SERVER_PID 不存在，可能已经停止"
    rm -f "$PID_FILE"
    exit 1
fi

# 停止进程
echo "🛑 正在停止服务器 (PID: $SERVER_PID)..."
kill "$SERVER_PID"

# 等待进程结束（最多等待10秒）
for i in {1..10}; do
    if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
        echo "✅ 服务器已成功停止"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# 如果还在运行，强制杀死
if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "⚠️  进程未响应，强制终止..."
    kill -9 "$SERVER_PID"
    sleep 1
    if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
        echo "✅ 服务器已强制停止"
        rm -f "$PID_FILE"
    else
        echo "❌ 无法停止服务器，请手动检查"
        exit 1
    fi
fi

