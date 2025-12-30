#!/bin/bash

# ============================================================================
# 运行说明:
#   在项目根目录下运行此脚本:
#   cd /home/hongzefu/historybench-v5.6.1b2-refractor
#   bash history_bench_sim/run_oraclev4.sh
#
#   或者直接运行:
#   bash /home/hongzefu/historybench-v5.6.1b2-refractor/history_bench_sim/run_oraclev4.sh
# ============================================================================

# 脚本配置
SCRIPT_DIR="/home/hongzefu/historybench-v5.6.1b2-refractor"
PYTHON_SCRIPT="${SCRIPT_DIR}/history_bench_sim/oraclev4.py"
ENV_PATH="/home/hongzefu/micromamba/envs/maniskillenv1228"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/oraclev4_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 记录开始时间
echo "==========================================" | tee -a "${LOG_FILE}"
echo "开始运行时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "Python脚本: ${PYTHON_SCRIPT}" | tee -a "${LOG_FILE}"
echo "环境路径: ${ENV_PATH}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# 切换到脚本目录
cd "${SCRIPT_DIR}" || {
    echo "错误: 无法切换到目录 ${SCRIPT_DIR}" | tee -a "${LOG_FILE}"
    exit 1
}

# 使用micromamba环境中的Python运行脚本
# 使用 -u 参数确保无缓冲输出，实现实时日志
# 将所有输出（stdout和stderr）重定向到日志文件，同时也在终端显示
"${ENV_PATH}/bin/python" -u "${PYTHON_SCRIPT}" 2>&1 | stdbuf -oL -eL tee -a "${LOG_FILE}"

# 记录结束时间和退出状态
EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "结束运行时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "退出状态码: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}

