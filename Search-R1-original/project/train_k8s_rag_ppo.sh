#!/bin/bash
#
# K8s RAG PPO Training Script
# Based on Search-R1 framework with LLM-as-Judge reward
#

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Data paths
export DATA_DIR='/ssd1/zz/AI_efficency/RAG/data/k8s_qa_data_en'
export TRAIN_DATA="$DATA_DIR/qa_2000_searchr1_format.parquet"
export VAL_DATA="$DATA_DIR/qa_100_searchr1_format.parquet"

# Model configuration
export BASE_MODEL='/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='k8s-rag-ppo-qwen2.5-7b-llm-judge'
export WAND_PROJECT='K8s-RAG-Search-R1'

# Resume from checkpoint (set to null for fresh start)
export RESUME_FROM_CHECKPOINT=null

# VLLM backend configuration
export VLLM_ATTENTION_BACKEND=XFORMERS

# Retrieval service configuration
export RETRIEVER_URL="http://127.0.0.1:8001/retrieve"
export RETRIEVER_TOPK=3

# Logging
mkdir -p logs
export LOG_FILE="./project/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "K8s RAG PPO Training"
echo "=========================================="
echo "Model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "Train Data: $TRAIN_DATA"
echo "Val Data: $VAL_DATA"
echo "Retriever: $RETRIEVER_URL (topk=$RETRIEVER_TOPK)"
echo "Log: $LOG_FILE"
echo "=========================================="

# Start training
cd /ssd1/zz/AI_efficency/RAG/Search-R1

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_training_steps=300 \
    trainer.save_freq=50 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=$RETRIEVER_TOPK \
    2>&1 | tee $LOG_FILE
