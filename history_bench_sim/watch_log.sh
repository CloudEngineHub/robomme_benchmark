#!/bin/bash

# ============================================================================
# 运行说明:
#   在项目根目录下运行此脚本:
#   cd /home/hongzefu/historybench-v5.6.1b2-refractor
#   bash history_bench_sim/watch_log.sh
#
#   或者指定日志文件:
#   bash history_bench_sim/watch_log.sh logs/oraclev4_20240101_120000.log
#
#   或者直接运行:
#   bash /home/hongzefu/historybench-v5.6.1b2-refractor/history_bench_sim/watch_log.sh
# ============================================================================

# 实时查看日志脚本
LOG_DIR="/home/hongzefu/historybench-v5.6.1b2-refractor/logs"

# 如果提供了日志文件名，使用指定的文件
if [ -n "$1" ]; then
    LOG_FILE="$1"
else
    # 否则自动查找最新的日志文件
    LOG_FILE=$(ls -t "${LOG_DIR}"/oraclev4_*.log 2>/dev/null | head -1)
fi

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    echo "错误: 找不到日志文件"
    echo "用法: $0 [日志文件路径]"
    echo "或者: $0  (自动使用最新的日志文件)"
    exit 1
fi

echo "正在实时查看日志文件: ${LOG_FILE}"
echo "按 Ctrl+C 退出"
echo "=========================================="

# 使用 tail -f 实时跟踪日志文件
tail -f "${LOG_FILE}"



