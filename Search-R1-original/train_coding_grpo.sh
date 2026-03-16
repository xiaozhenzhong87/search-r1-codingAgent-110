#!/bin/bash
# Coding Agent GRPO Training Script
# Uses Docker sandbox for Verilog code execution and test pass rate reward
#
# Usage: source activate /ssd1/zz/envs/searchr1 && bash train_coding_grpo.sh

export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='data/cvdp_coding'
export VAL_DATA="${DATA_DIR}/test.parquet"

WAND_PROJECT='Coding-Agent-RL'

# Use Qwen2.5-3B if available, otherwise fall back to 7B-Instruct
if [ -d "/ssd1/zz/models/Qwen/Qwen2.5-3B" ]; then
    export BASE_MODEL='/ssd1/zz/models/Qwen/Qwen2.5-3B'
    export EXPERIMENT_NAME=cvdp-coding-grpo-qwen2.5-3b
else
    export BASE_MODEL='/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct'
    export EXPERIMENT_NAME=cvdp-coding-grpo-qwen2.5-7b-it
fi

echo "Using model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"

export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_TMPDIR=/ssd1/zz/ray_tmp

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_coding \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$VAL_DATA \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=32 \
    data.val_batch_size=7 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=1024 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=30 \
    trainer.total_training_steps=200 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/ssd1/zz/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=10 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    do_search=true \
    2>&1 | tee $EXPERIMENT_NAME.log
