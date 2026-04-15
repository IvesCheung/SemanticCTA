#!/bin/bash
# ============================================================
#  LLM Hidden-State CTA Pipeline — 一键运行脚本
#
#  Step 1: index_cta_llm_hidden.py  生成 embedding
#  Step 2: train_cta.py             训练分类器 + 5-fold 评估
#
#  用法：
#    1. 修改下方 =================== 之间的配置项
#    2. chmod +x run_llm_hidden_cta.sh
#    3. ./run_llm_hidden_cta.sh
# ============================================================

set -euo pipefail

# =================== 配置项（按需修改）====================

# --- 模型 ---
MODEL_PATH="/home/project/schema_profiling/results/aitelco/model/qwen7B"           # 本地模型路径 (HuggingFace 格式)
GPU_IDS="0,1"                         # 可用 GPU，逗号分隔；device_map=auto 自动分配

export CUDA_LAUNCH_BLOCKING=1
LAYERS="-1"                               # 提取哪些层的隐藏状态 (例: "-1" 或 "-1,-4,-8,-12")
MAX_PREFIX_LENGTH=1024                     # prefix 最大 token 数

# --- 数据 ---
DATASET_DIR="datasets/gittables-semtab22-db-all_wrangled"  # fold_*.csv 和表文件所在目录
SAMPLE_ROWS=5                             # 每张表序列化行数

# --- 输出 ---
RESULT_DIR="results/cta_llm_hidden/gittables-semtab22"

# --- 训练超参 ---
HIDDEN_DIM=1024                            # MLP 隐藏层维度
TRAIN_BATCH_SIZE=64
LR=3e-4
EPOCHS=300
NUM_RESIDUAL_BLOCKS=2
DROPOUT=0.3
WEIGHT_DECAY=1e-2
LABEL_SMOOTHING=0.1
WARMUP_EPOCHS=3
FOCAL_GAMMA=2.0
MIXUP_ALPHA=0.2
PATIENCE=0
SAVE_METRIC="f1"

# =================== 配置项结束 ==========================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMBEDDING_PATH="${RESULT_DIR}/embeddings_llm_hidden.pkl"
TRAIN_RESULT_DIR="${RESULT_DIR}/train_$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "============================================================"
echo " LLM Hidden-State CTA Pipeline"
echo "============================================================"
echo " Model:           ${MODEL_PATH}"
echo " GPUs:            ${GPU_IDS}"
echo " Layers:          ${LAYERS}"
echo " Dataset:         ${DATASET_DIR}"
echo " Result dir:      ${RESULT_DIR}"
echo " Embedding path:  ${EMBEDDING_PATH}"
echo " Train result:    ${TRAIN_RESULT_DIR}"
echo "============================================================"

mkdir -p "${RESULT_DIR}"

# ----------------------------------------------------------
#  Step 1: 生成 Embedding
# ----------------------------------------------------------
echo ""
echo ">>>>>>>>>> Step 1: Generating Embeddings >>>>>>>>>>"
echo ""

python3 "${SCRIPT_DIR}/column_type_annotation/index_cta_llm_hidden.py" \
    --model_path "${MODEL_PATH}" \
    --fold_dir "${DATASET_DIR}" \
    --table_dir "${DATASET_DIR}" \
    --output_path "${EMBEDDING_PATH}" \
    --sample_rows "${SAMPLE_ROWS}" \
    --max_prefix_length "${MAX_PREFIX_LENGTH}" \
    --layers="${LAYERS}" \
    --no_require_profile

echo ""
echo "<<<<<<<<<< Step 1 Done <<<<<<<<<<"
echo ""

# ----------------------------------------------------------
#  Step 2: 训练分类器
# ----------------------------------------------------------
echo ""
echo ">>>>>>>>>> Step 2: Training Classifier >>>>>>>>>>"
echo ""

mkdir -p "${TRAIN_RESULT_DIR}"

python3 "${SCRIPT_DIR}/column_type_annotation/train_cta.py" \
    --fold_dir "${DATASET_DIR}" \
    --embedding_path "${EMBEDDING_PATH}" \
    --result_dir "${TRAIN_RESULT_DIR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --num_epochs "${EPOCHS}" \
    --num_residual_blocks "${NUM_RESIDUAL_BLOCKS}" \
    --dropout_rate "${DROPOUT}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --label_smoothing "${LABEL_SMOOTHING}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --focal_gamma "${FOCAL_GAMMA}" \
    --mixup_alpha "${MIXUP_ALPHA}" \
    --patience "${PATIENCE}" \
    --save_metric "${SAVE_METRIC}" \
    --embedding_model "llm_hidden"

echo ""
echo "<<<<<<<<<< Step 2 Done <<<<<<<<<<"
echo ""

echo "============================================================"
echo " All Done!"
echo " Embeddings:  ${EMBEDDING_PATH}"
echo " Train logs:  ${TRAIN_RESULT_DIR}"
echo " Results:     ${TRAIN_RESULT_DIR}/results.json"
echo "============================================================"
