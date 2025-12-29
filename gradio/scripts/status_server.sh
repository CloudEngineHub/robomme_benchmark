#!/bin/bash
# 查看服务器状态脚本

# 获取脚本所在目录，然后定位到 gradio 目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRADIO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$GRADIO_DIR/server.pid"
LOG_DIR="$GRADIO_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"

echo "📊 服务器状态信息"
echo "=========================================="

# 检查PID文件
if [ ! -f "$PID_FILE" ]; then
    echo "❌ 服务器未运行 (未找到 PID 文件)"
    exit 1
fi

SERVER_PID=$(cat "$PID_FILE")

# 检查进程是否存在
if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "✅ 服务器正在运行"
    echo "   PID: $SERVER_PID"
    echo ""
    
    # 显示进程信息
    echo "📋 进程信息:"
    ps -p "$SERVER_PID" -o pid,ppid,user,%cpu,%mem,etime,cmd
    echo ""
    
    # 显示日志文件信息
    if [ -f "$LOG_FILE" ]; then
        LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
        LOG_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
        echo "📄 日志文件信息:"
        echo "   文件: $LOG_FILE"
        echo "   大小: $LOG_SIZE"
        echo "   行数: $LOG_LINES"
        echo "   最后修改: $(stat -c %y "$LOG_FILE" 2>/dev/null || stat -f %Sm "$LOG_FILE" 2>/dev/null || echo "未知")"
    fi
    
    # 显示最后几行日志
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "📝 最近的日志输出 (最后10行):"
        echo "----------------------------------------"
        tail -n 10 "$LOG_FILE"
    fi
else
    echo "❌ 服务器未运行 (进程 $SERVER_PID 不存在)"
    echo "   清理 PID 文件..."
    rm -f "$PID_FILE"
    exit 1
fi

