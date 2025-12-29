#!/bin/bash
# 查看日志脚本 - 实时查看服务器日志

# 获取脚本所在目录，然后定位到 gradio 目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRADIO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$GRADIO_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"

# 检查日志文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  未找到日志文件，服务器可能还未启动"
    echo "   日志文件路径: $LOG_FILE"
    exit 1
fi

# 显示选项菜单
echo "📋 日志查看选项:"
echo "   1) 查看完整日志 (实时) - 推荐"
echo "   2) 查看完整日志 (最后100行)"
echo "   3) 查看完整日志 (最后500行)"
echo "   4) 查看完整日志 (最后1000行)"
echo "   5) 搜索日志中的关键词"
echo ""
read -p "请选择 [1-5]: " choice

case $choice in
    1)
        echo "📄 查看完整日志 (实时，按 Ctrl+C 退出)..."
        echo "   文件: $LOG_FILE"
        echo "=========================================="
        tail -f "$LOG_FILE" 2>/dev/null || echo "日志文件不存在或为空"
        ;;
    2)
        echo "📄 完整日志 (最后100行):"
        echo "=========================================="
        tail -n 100 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在或为空"
        ;;
    3)
        echo "📄 完整日志 (最后500行):"
        echo "=========================================="
        tail -n 500 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在或为空"
        ;;
    4)
        echo "📄 完整日志 (最后1000行):"
        echo "=========================================="
        tail -n 1000 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在或为空"
        ;;
    5)
        read -p "请输入要搜索的关键词: " keyword
        if [ -z "$keyword" ]; then
            echo "❌ 关键词不能为空"
            exit 1
        fi
        echo "📄 搜索结果 (包含关键词 '$keyword' 的行):"
        echo "=========================================="
        grep -i "$keyword" "$LOG_FILE" 2>/dev/null || echo "未找到包含 '$keyword' 的日志"
        ;;
    *)
        echo "❌ 无效的选择"
        exit 1
        ;;
esac

