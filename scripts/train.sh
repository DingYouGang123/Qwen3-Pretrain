#!/bin/bash
# 训练启动脚本
# 支持单卡和多卡分布式训练

set -e

# 默认配置
CONFIG_FILE="${1:-configs/qwen3_0.6b.yaml}"
NUM_GPUS="${2:-1}"

echo "========================================"
echo "Qwen3-0.6B Pretraining Script"
echo "========================================"
echo "Config file: ${CONFIG_FILE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "========================================"

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# 根据 GPU 数量选择训练方式
if [ "${NUM_GPUS}" -eq 1 ]; then
    echo "Starting single-GPU training..."
    python train.py --config "${CONFIG_FILE}"
else
    echo "Starting multi-GPU training with ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node="${NUM_GPUS}" train.py --config "${CONFIG_FILE}"
fi

echo "Training completed!"
