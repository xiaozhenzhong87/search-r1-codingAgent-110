#!/usr/bin/env bash
# 完整流程: 停止训练 → 修复WandB → 重启训练(独立新曲线)

set -e

cd /ssd1/zz/AI_efficency/RAG/Search-R1

echo "========================================================================"
echo "  停止当前训练并创建独立WandB曲线"
echo "========================================================================"
echo ""

# 1. 停止当前训练
echo "[1/6] 停止当前训练进程..."
pkill -f "train_ppo_from_step200_improved.sh" 2>/dev/null || true
sleep 3
pkill -9 -f "ray::main_task" 2>/dev/null || true
sleep 2

echo "✓ 训练已停止"
echo ""

# 2. 备份并修改trainer_state.json (删除wandb_run_id)
echo "[2/6] 修复trainer_state.json..."

ACTOR_STATE="verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200/trainer_state.json"
CRITIC_STATE="verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/critic/global_step_200/trainer_state.json"

# 备份actor
if [ -f "$ACTOR_STATE" ]; then
    cp "$ACTOR_STATE" "${ACTOR_STATE}.backup_$(date +%Y%m%d_%H%M%S)"
    echo "✓ 已备份 actor/trainer_state.json"
    
    # 删除wandb_run_id
    cat > "$ACTOR_STATE" << 'EOF'
{
  "global_steps": 200,
  "kl_coef": 0.001
}
EOF
    echo "✓ 已删除 actor 中的 wandb_run_id"
fi

# 备份critic
if [ -f "$CRITIC_STATE" ]; then
    cp "$CRITIC_STATE" "${CRITIC_STATE}.backup_$(date +%Y%m%d_%H%M%S)"
    echo "✓ 已备份 critic/trainer_state.json"
    
    cat > "$CRITIC_STATE" << 'EOF'
{
  "global_steps": 200,
  "kl_coef": 0.001
}
EOF
    echo "✓ 已删除 critic 中的 wandb_run_id"
fi

echo ""

# 3. 清理WandB缓存
echo "[3/6] 清理WandB本地缓存..."
rm -rf wandb/run-* 2>/dev/null || true
echo "✓ 已清理旧的WandB缓存"
echo ""

# 4. 清理临时resume目录(可选)
echo "[4/6] 清理临时resume目录..."
if [ -L "verl_checkpoints/temp_resume_from_step_200/actor/global_step_200" ]; then
    rm -rf verl_checkpoints/temp_resume_from_step_200
    echo "✓ 已清理临时目录"
fi
echo ""

# 5. 重新创建临时resume目录
echo "[5/6] 重新创建resume目录..."
TEMP_RESUME_DIR=verl_checkpoints/temp_resume_from_step_200
SOURCE_CKPT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl

mkdir -p "$TEMP_RESUME_DIR/actor"
mkdir -p "$TEMP_RESUME_DIR/critic"

ln -s "$(pwd)/$SOURCE_CKPT/actor/global_step_200" "$TEMP_RESUME_DIR/actor/global_step_200"
ln -s "$(pwd)/$SOURCE_CKPT/critic/global_step_200" "$TEMP_RESUME_DIR/critic/global_step_200"

echo "✓ 已创建临时resume目录"
echo ""

# 6. 重新启动训练
echo "[6/6] 重新启动训练(创建新WandB run)..."
nohup bash train_ppo_from_step200_improved.sh > nohup_step200_newrun_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NEW_PID=$!
echo $NEW_PID > train_step200_newrun.pid

echo "✓ 训练已启动"
echo "  PID: $NEW_PID"
echo ""

# 等待WandB初始化
echo "等待WandB初始化(30秒)..."
sleep 30

echo ""
echo "========================================================================"
echo "  完成"
echo "========================================================================"
echo ""

# 检查WandB初始化
echo "检查WandB状态:"
LATEST_LOG=$(ls -t nohup_step200_newrun_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo ""
    grep -E "wandb:.*run|Resuming run" "$LATEST_LOG" | grep -v "repeated" | tail -10
    echo ""
    
    if grep -q "Resuming run nq-ppo-sft-noformat-v2-lowvarkl" "$LATEST_LOG"; then
        echo "⚠️  警告: 仍在resume旧run,可能需要手动检查"
    else
        echo "✓ 成功创建新的WandB run!"
    fi
else
    echo "未找到日志文件"
fi

echo ""
echo "监控日志:"
echo "  tail -f $LATEST_LOG"
echo ""
echo "查看GPU使用:"
echo "  watch -n 1 nvidia-smi"
echo ""
