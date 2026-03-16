#!/bin/bash
#
# K8s RAG Search-R1 训练脚本
# 基于 Search-R1 框架，使用 QA-EM reward 机制
# 
# 数据格式: search-r1 格式 (JSON)
# 模型: Qwen2.5-7B-Instruct
# 训练方法: PPO with retrieval
#

set -euo pipefail

# ========== GPU 配置 ==========
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ========== 数据路径配置 ==========
export DATA_DIR='/ssd1/zz/AI_efficency/RAG/data/k8s_qa_data_en'
export TRAIN_DATA="$DATA_DIR/qa_1950_searchr1_format.parquet"
export VAL_DATA="$DATA_DIR/qa_50_test_searchr1_format.parquet"
export TEST_DATA="$DATA_DIR/qa_50_test_searchr1_format.parquet"

# 验证数据文件存在
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 错误: 找不到训练数据文件: $TRAIN_DATA"
    echo "提示: 需要先将 JSON 转换为 Parquet 格式"
    echo "运行: python project/convert_json_to_parquet.py $DATA_DIR/qa_2000_searchr1_format.json $TRAIN_DATA"
    exit 1
fi

# ========== 模型配置 ==========
export BASE_MODEL='/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='k8s-rag-searchr1-ppo-qwen2.5-7b-LLMAsJudge'
export WAND_PROJECT='K8s-RAG-Search-R1'

# 验证模型路径存在
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 错误: 找不到基础模型: $BASE_MODEL"
    exit 1
fi

# ========== 训练恢复配置 ==========
# 设置为 null 表示从头开始训练
# 设置为检查点路径表示恢复训练: verl_checkpoints/$EXPERIMENT_NAME
export RESUME_FROM_CHECKPOINT=null

# ========== VLLM 后端配置 ==========
# 使用 XFORMERS 以避免 Qwen2.5 + flash_attn 的已知问题
export VLLM_ATTENTION_BACKEND=XFORMERS

# ========== Retrieval 服务配置 ==========
export RETRIEVER_URL="http://127.0.0.1:8001/retrieve"
export RETRIEVER_TOPK=3

# 验证检索服务是否可用
echo "正在检查检索服务..."
if curl -s --connect-timeout 5 "$RETRIEVER_URL" > /dev/null 2>&1; then
    echo "✅ 检索服务可用: $RETRIEVER_URL"
else
    echo "⚠️  警告: 无法连接到检索服务: $RETRIEVER_URL"
    echo "请确保检索服务已启动，否则训练将失败"
    read -p "按回车继续或 Ctrl+C 取消..." 
fi

# ========== 日志配置 ==========
LOG_DIR="./project/logs"
mkdir -p "$LOG_DIR"
export LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

# ========== 打印配置信息 ==========
echo "=========================================="
echo "  K8s RAG Search-R1 PPO 训练"
echo "=========================================="
echo "📦 基础模型: $BASE_MODEL"
echo "🔬 实验名称: $EXPERIMENT_NAME"
echo "📊 项目名称: $WAND_PROJECT"
echo "📁 训练数据: $TRAIN_DATA"
echo "📁 验证数据: $VAL_DATA"
echo "📁 测试数据: $TEST_DATA"
echo "🔍 检索服务: $RETRIEVER_URL (topk=$RETRIEVER_TOPK)"
echo "💾 检查点目录: verl_checkpoints/$EXPERIMENT_NAME"
echo "📝 日志文件: $LOG_FILE"
echo "🔄 恢复训练: $RESUME_FROM_CHECKPOINT"
echo "=========================================="
echo ""

# 统计数据条数
TRAIN_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_DATA')))" 2>/dev/null || echo "未知")
TEST_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$TEST_DATA')))" 2>/dev/null || echo "未知")
echo "📊 训练样本数: $TRAIN_COUNT 条"
echo "📊 测试样本数: $TEST_COUNT 条"
echo ""

# ========== 切换到项目目录 ==========
cd /ssd1/zz/AI_efficency/RAG/Search-R1

echo "🚀 开始训练..."
echo ""

# ========== 启动训练 ==========
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10 \
    trainer.total_training_steps=300 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
    max_turns=4 \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=$RETRIEVER_TOPK \
    2>&1 | tee $LOG_FILE

# ========== 训练完成 ==========
echo ""
echo "=========================================="
echo "  训练完成!"
echo "=========================================="
echo "📊 查看日志: cat $LOG_FILE"
echo "📈 查看 WandB: https://wandb.ai (项目: $WAND_PROJECT)"
echo "💾 检查点位置: verl_checkpoints/$EXPERIMENT_NAME"
echo "=========================================="
