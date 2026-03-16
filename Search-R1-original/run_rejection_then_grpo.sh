#!/usr/bin/env bash
# Rejection Sampling Pipeline 优先；失败则回退到 train_grpo.sh
# 用法: nohup bash run_rejection_then_grpo.sh > run_rejection_then_grpo.log 2>&1 &

set -e
cd /ssd1/zz/AI_efficency/RAG/Search-R1

LOG_DIR=./logs
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline_rejection_sampling_v2.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始执行 Rejection Sampling Pipeline，日志: $PIPELINE_LOG"

if python -m search_r1.pipeline_rejection_sampling \
  --train_data data/nq_search/train.parquet \
  --test_data data/nq_search/test.parquet \
  --base_model /ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./rejection_sampling_output \
  --num_samples 5 \
  --sft_epochs 3 \
  --rl_epochs 15 \
  --num_gpus 8 >> "$PIPELINE_LOG" 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline 正常结束"
  exit 0
fi

# 执行失败，启动 train_grpo.sh 作为后备
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline 执行失败，启动后备: train_grpo.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ssd1/zz/envs/searchr1
export CUDA_HOME=/ssd1/zz/envs/searchr1
export HF_HOME=/ssd1/zz/.cache/huggingface

nohup bash train_grpo.sh > train_grpo_v3.log 2>&1 &
GRPO_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] train_grpo.sh 已在后台启动, PID=$GRPO_PID, 日志: train_grpo_v3.log"
exit 1
