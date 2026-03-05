#!/bin/sh
set -eu

pick_vulkan_icd() {
    for candidate in \
        /etc/vulkan/icd.d/nvidia_icd.json \
        /etc/vulkan/icd.d/nvidia_icd.x86_64.json \
        /usr/share/vulkan/icd.d/nvidia_icd.json \
        /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
    do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

run_diagnostic() {
    label="$1"
    shift
    echo "[entrypoint] $label"
    if "$@"; then
        return 0
    fi
    status=$?
    echo "[entrypoint] $label failed with exit code $status"
    return 0
}

if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=1
fi

if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    if detected_icd="$(pick_vulkan_icd)"; then
        export VK_ICD_FILENAMES="$detected_icd"
        echo "[entrypoint] Using Vulkan ICD: $VK_ICD_FILENAMES"
    else
        echo "[entrypoint] Vulkan ICD file not found under /etc or /usr/share"
    fi
else
    echo "[entrypoint] Respecting preset VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
fi

echo "[entrypoint] OMP_NUM_THREADS=$OMP_NUM_THREADS"
run_diagnostic "nvidia-smi" nvidia-smi
run_diagnostic "vulkaninfo --summary" vulkaninfo --summary
exec "$@"
