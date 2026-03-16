#!/usr/bin/env bash
#
# 使用改进的reward function从Step 200继续训练
# 改进: 惩罚 answer 后继续输出
#
# 使用方法:
#   bash train_ppo_resume_step200_improved.sh
#

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据配置
# export DATA_DIR='data/nq_search'
export DATA_DIR='data/nq_hotpot_search'
export NQ_TEST='data/nq_search/test.parquet'
export HOTPOT_TEST='data/hotpot_search/test.parquet'

WAND_PROJECT='Search-R1'

# 模型配置
export BASE_MODEL='/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct'

# ═══════════════════════════════════════════════════════════════
# 实验配置 - 从现有checkpoint继续训练,使用改进的 reward
# ═══════════════════════════════════════════════════════════════
# 
# 修复说明: 
# - resume_from_checkpoint 应该指向实验根目录,而不是具体的 global_step_XXX
# - 代码会自动检测最新的checkpoint (如 global_step_500)
# 
# 如果想从特定step恢复,有两个选择:
# 选择1: 从最新checkpoint (step 500) 恢复 - 直接用根目录
export EXPERIMENT_NAME=nq-hp-ppo-improved-reward-resumefrom200steps
export RESUME_FROM_CHECKPOINT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl

# 选择2: 从step 200恢复 - 需要创建临时目录(见下面注释)
# export EXPERIMENT_NAME=nq-hp-ppo-step200-improved-reward-v2
# TEMP_DIR=verl_checkpoints/temp_resume_step_200
# mkdir -p $TEMP_DIR/actor $TEMP_DIR/critic
# ln -sf $(pwd)/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200 $TEMP_DIR/actor/global_step_200
# ln -sf $(pwd)/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/critic/global_step_200 $TEMP_DIR/critic/global_step_200
# export RESUME_FROM_CHECKPOINT=$TEMP_DIR

echo "========================================================================"
echo "  从 Step 200 继续训练 - 使用改进的 Reward Function"
echo "========================================================================"
echo ""
echo "基础配置:"
echo "  模型: $BASE_MODEL"
echo "  实验名: $EXPERIMENT_NAME"
echo "  恢复检查点: $RESUME_FROM_CHECKPOINT"
echo "  数据目录: $DATA_DIR"
echo ""
echo "Reward 改进:"
echo "  - 答案正确 + 无多余输出: 1.0 ✅"
echo "  - 答案正确 + 有多余输出: 0.8 ⚠️ (惩罚 -0.2)"
echo "  - 答案错误: 0.0 ❌"
echo ""
echo "训练参数:"
echo "  总步数: 500"
echo "  保存频率: 每 100 步"
echo "  测试频率: 每 50 步"
echo "  最大轮数: 4"
echo "  KL系数: 0.001"
echo ""
echo "========================================================================"
echo ""

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# 创建日志目录
mkdir -p ./logs/improved_reward

# 启动训练
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$NQ_TEST \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ./logs/improved_reward/${EXPERIMENT_NAME}.log

echo ""
echo "========================================================================"
echo "  训练完成"
echo "========================================================================"
echo ""
echo "查看日志:"
echo "  tail -f ./logs/improved_reward/${EXPERIMENT_NAME}.log"
echo ""
echo "检查点目录:"
echo "  ls -lh verl_checkpoints/${EXPERIMENT_NAME}/"
echo ""
echo "监控训练:"
echo "  - WandB: 查看项目 $WAND_PROJECT / $EXPERIMENT_NAME"
echo "  - 关注指标: 'Has continuation after answer' 的比例应该逐渐下降"
echo ""
