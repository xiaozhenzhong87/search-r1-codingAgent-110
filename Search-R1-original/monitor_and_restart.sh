#!/bin/bash
# 监控 GRPO 训练日志，如果 critic/score/mean 连续 5 个 step 为 0，
# 自动停止训练并以新参数重启。

#############################
# 可配置参数
#############################
LOG_V3="/ssd1/zz/AI_efficency/RAG/Search-R1/train_grpo_v3.log"
LOG_V4="train_grpo_v4.log"
WORK_DIR="/ssd1/zz/AI_efficency/RAG/Search-R1"
CONSECUTIVE_ZERO_THRESHOLD=5    # 连续多少个 step score=0 触发重启
POLL_INTERVAL=30                # 轮询间隔（秒）
NEW_KL_LOSS_COEF=0.01          # 重启时使用的 kl_loss_coef

#############################
# 函数：从日志提取最近 N 个 step 的 score
#############################
get_recent_scores() {
    local log_file="$1"
    local n="$2"
    # 匹配训练 step 行（含 critic/score/mean），提取 step 编号和 score
    grep -oP 'step:\d+ .*critic/score/mean:\K[0-9.]+' "$log_file" | tail -n "$n"
}

#############################
# 函数：获取最新 step 编号
#############################
get_latest_step() {
    local log_file="$1"
    grep -oP 'step:\K\d+(?= -)' "$log_file" | tail -1
}

#############################
# 函数：杀掉训练进程
#############################
kill_training() {
    echo "[$(date)] 正在停止训练进程..."

    # 方式1：通过 python verl.trainer.main_ppo 找到主进程
    local pids=$(ps aux | grep '[v]erl.trainer.main_ppo' | awk '{print $2}')
    if [ -n "$pids" ]; then
        echo "[$(date)] 找到 verl 训练进程: $pids"
        for pid in $pids; do
            # 杀掉整个进程组
            kill -TERM -$(ps -o pgid= -p "$pid" | tr -d ' ') 2>/dev/null
            kill -TERM "$pid" 2>/dev/null
        done
    fi

    # 方式2：杀掉所有 ray 相关进程（确保清理干净）
    sleep 5
    local ray_pids=$(ps aux | grep '[r]ay::' | awk '{print $2}')
    if [ -n "$ray_pids" ]; then
        echo "[$(date)] 清理残留 ray 进程..."
        echo "$ray_pids" | xargs kill -9 2>/dev/null
    fi

    # 等待进程完全退出
    sleep 10
    echo "[$(date)] 训练进程已停止"
}

#############################
# 函数：启动 v4 训练
#############################
start_v4_training() {
    echo "[$(date)] 启动 v4 训练（kl_loss_coef=${NEW_KL_LOSS_COEF}）..."

    cd "$WORK_DIR"

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export DATA_DIR='data/nq_search'
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export BASE_MODEL='/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct'
    export EXPERIMENT_NAME="nq-search-r1-grpo-qwen2.5-7b-it-em-paper-klcoef${NEW_KL_LOSS_COEF}"

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=512 \
        data.val_batch_size=256 \
        data.max_prompt_length=4096 \
        data.max_response_length=500 \
        data.max_start_length=2048 \
        data.max_obs_length=500 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.use_kl_loss=true \
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
        actor_rollout_ref.actor.kl_loss_coef=${NEW_KL_LOSS_COEF} \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.no_think_rl=false \
        actor_rollout_ref.rollout.n_agent=5 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.actor.state_masking=true \
        trainer.logger=['console','wandb'] \
        +trainer.val_only=false \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=50 \
        trainer.project_name='Search-R1' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=15 \
        trainer.total_training_steps=500 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
        max_turns=4 \
        retriever.url="http://127.0.0.1:8000/retrieve" \
        retriever.topk=3 \
        2>&1 | tee ${LOG_V4} &

    echo "[$(date)] v4 训练已启动，日志: ${LOG_V4}，PID: $!"
}

#############################
# 主监控循环
#############################
echo "========================================"
echo " GRPO 训练监控脚本"
echo " 监控日志: ${LOG_V3}"
echo " 触发条件: critic/score/mean 连续 ${CONSECUTIVE_ZERO_THRESHOLD} 个 step 为 0"
echo " 重启参数: kl_loss_coef=${NEW_KL_LOSS_COEF}"
echo " 轮询间隔: ${POLL_INTERVAL}s"
echo "========================================"

consecutive_zeros=0
last_checked_step=""

while true; do
    if [ ! -f "$LOG_V3" ]; then
        echo "[$(date)] 日志文件不存在，等待中..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # 获取最新 step
    current_step=$(get_latest_step "$LOG_V3")
    if [ -z "$current_step" ] || [ "$current_step" = "$last_checked_step" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # 获取最近的 score
    latest_score=$(grep -oP 'step:'"$current_step"' .*critic/score/mean:\K[0-9.]+' "$LOG_V3" | tail -1)

    if [ -z "$latest_score" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    last_checked_step="$current_step"

    # 判断 score 是否为 0（包括 0.000）
    is_zero=$(echo "$latest_score" | awk '{if ($1 + 0 == 0) print "yes"; else print "no"}')

    if [ "$is_zero" = "yes" ]; then
        consecutive_zeros=$((consecutive_zeros + 1))
        echo "[$(date)] step:${current_step} score=${latest_score} | 连续归零: ${consecutive_zeros}/${CONSECUTIVE_ZERO_THRESHOLD}"
    else
        if [ "$consecutive_zeros" -gt 0 ]; then
            echo "[$(date)] step:${current_step} score=${latest_score} | 连续归零计数重置"
        fi
        consecutive_zeros=0
        echo "[$(date)] step:${current_step} score=${latest_score} | 正常"
    fi

    # 触发重启
    if [ "$consecutive_zeros" -ge "$CONSECUTIVE_ZERO_THRESHOLD" ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[$(date)] 检测到连续 ${CONSECUTIVE_ZERO_THRESHOLD} 个 step score=0，触发重启！"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""

        kill_training
        start_v4_training

        echo "[$(date)] 监控脚本退出。如需监控 v4，请另起监控。"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
