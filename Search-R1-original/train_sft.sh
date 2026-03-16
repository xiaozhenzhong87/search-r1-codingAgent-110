#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# FSDP: 模型分片到 8 卡，每卡只存 1/8 参数+优化器
# 8 GPU × batch_size 2 × grad_accum 2 = effective batch 32
# 1148 / 32 = ~36 steps/epoch × 3 epochs = ~108 steps

/ssd1/zz/envs/searchr1/bin/torchrun \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    -m search_r1.train_sft \
    --base_model /ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct \
    --data_file /ssd1/zz/AI_efficency/RAG/Search-R1/rejection-sampling/output/vllm_rs_10k/sft_dataset.parquet \
    --output_dir /ssd1/zz/AI_efficency/RAG/Search-R1/sft_output/sft_1148_rs \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --gradient_accumulation_steps 2
