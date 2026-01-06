#!/bin/bash
# 后台运行脚本 - 统一管理 HistoryBench Gradio 服务器
# 使用方法: bash run_background.sh [start|stop|status|restart|logs]

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
LOG_FILE="$LOG_DIR/server.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 函数：启动服务器
start_server() {
    # 检查是否已经在运行
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "⚠️  服务器已经在运行中 (PID: $OLD_PID)"
            echo "   如需重启，请使用: bash $0 restart"
            return 1
        else
            echo "清理旧的 PID 文件..."
            rm -f "$PID_FILE"
        fi
    fi

    # 检查 micromamba 环境是否存在
    if [ ! -d "$MICROMAMBA_ENV" ]; then
        echo "❌ 错误: Micromamba 环境不存在: $MICROMAMBA_ENV"
        return 1
    fi

    # 检查 Python 可执行文件
    PYTHON_EXE="$MICROMAMBA_ENV/bin/python"
    if [ ! -f "$PYTHON_EXE" ]; then
        echo "❌ 错误: Python 可执行文件不存在: $PYTHON_EXE"
        return 1
    fi

    # 启动服务器
    echo "🚀 正在后台启动服务器..."
    echo "   Micromamba 环境: $MICROMAMBA_ENV"
    echo "   Python 可执行文件: $PYTHON_EXE"
    echo "   工作目录: $GRADIO_DIR"
    echo "   日志文件: $LOG_FILE"
    echo ""

    # 使用环境中的 Python 直接运行服务器
    # 使用 nohup 在后台运行，并将所有输出重定向到日志文件
    # 设置环境变量以确保使用环境中的包和正确的输出行为
    # 使用 unbuffered 模式 (-u) 和 PYTHONUNBUFFERED=1 确保输出立即写入，不缓冲
    # 使用 stdbuf -oL -eL 确保行缓冲输出（如果可用）
    # 将标准输出和错误输出合并到一个文件 (2>&1)，这样所有日志都会完整显示
    # 使用 >> 追加模式，确保日志不会覆盖
    
    # 检查是否可以使用 stdbuf（Linux 系统通常有）
    if command -v stdbuf >/dev/null 2>&1; then
        # 使用 stdbuf 确保行缓冲输出，所有 print 和日志都会实时写入
        nohup env PATH="$MICROMAMBA_ENV/bin:$PATH" \
            PYTHONUNBUFFERED=1 \
            PYTHONIOENCODING=utf-8 \
            stdbuf -oL -eL "$PYTHON_EXE" -u "$GRADIO_DIR/main.py" >> "$LOG_FILE" 2>&1 &
    else
        # 如果没有 stdbuf，使用 Python 的 unbuffered 模式
        # 仍然设置所有必要的环境变量确保输出实时写入
        nohup env PATH="$MICROMAMBA_ENV/bin:$PATH" \
            PYTHONUNBUFFERED=1 \
            PYTHONIOENCODING=utf-8 \
            "$PYTHON_EXE" -u "$GRADIO_DIR/main.py" >> "$LOG_FILE" 2>&1 &
    fi

    # 保存进程ID
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"

    # 等待一下，检查进程是否成功启动
    sleep 3

    if ps -p "$SERVER_PID" > /dev/null 2>&1; then
        echo "✅ 服务器已成功在后台启动！"
        echo "   PID: $SERVER_PID"
        echo "   Micromamba 环境: $MICROMAMBA_ENV"
        echo "   日志文件: $LOG_FILE"
        echo ""
        echo "📋 常用命令:"
        echo "   查看实时日志: bash $0 logs"
        echo "   查看状态: bash $0 status"
        echo "   停止服务器: bash $0 stop"
        echo ""
        echo "💡 提示:"
        echo "   - 所有输出都保存在 $LOG_FILE（包括所有 print、uvicorn 日志等）"
        echo "   - 日志实时写入，与前台运行完全一致"
        echo "   - 即使关闭SSH连接，服务器也会继续运行"
        echo "   - 使用 PYTHONUNBUFFERED=1 和 stdbuf 确保日志实时写入"
        echo ""
        echo "🌐 服务器启动后，请查看日志文件获取访问地址："
        echo "   bash $0 logs"
        return 0
    else
        echo "❌ 服务器启动失败！"
        echo "   请查看完整日志: $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# 函数：停止服务器
stop_server() {
    # 检查PID文件是否存在
    if [ ! -f "$PID_FILE" ]; then
        echo "⚠️  未找到 PID 文件，服务器可能未运行"
        return 1
    fi

    # 读取PID
    SERVER_PID=$(cat "$PID_FILE")

    # 检查进程是否存在
    if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
        echo "⚠️  进程 $SERVER_PID 不存在，可能已经停止"
        rm -f "$PID_FILE"
        return 1
    fi

    # 停止进程
    echo "🛑 正在停止服务器 (PID: $SERVER_PID)..."
    kill "$SERVER_PID"

    # 等待进程结束（最多等待10秒）
    for i in {1..10}; do
        if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
            echo "✅ 服务器已成功停止"
            rm -f "$PID_FILE"
            return 0
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
            return 0
        else
            echo "❌ 无法停止服务器，请手动检查"
            return 1
        fi
    fi
}

# 函数：查看服务器状态
status_server() {
    echo "📊 服务器状态信息"
    echo "=========================================="

    # 检查PID文件
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ 服务器未运行 (未找到 PID 文件)"
        return 1
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
        return 0
    else
        echo "❌ 服务器未运行 (进程 $SERVER_PID 不存在)"
        echo "   清理 PID 文件..."
        rm -f "$PID_FILE"
        return 1
    fi
}

# 函数：重启服务器
restart_server() {
    echo "🔄 正在重启服务器..."
    stop_server
    sleep 2
    start_server
}

# 函数：查看日志
view_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "⚠️  日志文件不存在: $LOG_FILE"
        return 1
    fi
    
    echo "📝 查看服务器日志 (按 Ctrl+C 退出)"
    echo "=========================================="
    tail -f "$LOG_FILE"
}

# 函数：显示帮助信息
show_help() {
    echo "HistoryBench 服务器管理脚本"
    echo ""
    echo "使用方法:"
    echo "  bash $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  start    - 启动服务器（后台运行）"
    echo "  stop     - 停止服务器"
    echo "  status   - 查看服务器状态"
    echo "  restart  - 重启服务器"
    echo "  logs     - 查看实时日志（按 Ctrl+C 退出）"
    echo "  help     - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash $0 start      # 启动服务器"
    echo "  bash $0 status    # 查看状态"
    echo "  bash $0 logs      # 查看日志"
    echo "  bash $0 stop      # 停止服务器"
    echo ""
}

# 主逻辑：根据命令行参数执行相应操作
case "${1:-help}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    status)
        status_server
        ;;
    restart)
        restart_server
        ;;
    logs)
        view_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ 未知命令: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

exit $?
