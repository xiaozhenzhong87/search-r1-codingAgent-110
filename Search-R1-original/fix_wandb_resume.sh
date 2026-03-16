#!/usr/bin/env bash
# 修复WandB Resume问题 - 让训练创建新的WandB run而不是resume原来的

set -e

echo "========================================================================"
echo "  修复WandB Resume问题"
echo "========================================================================"
echo ""

# 备份原始trainer_state.json
ORIGINAL_STATE="/ssd1/zz/AI_efficency/RAG/Search-R1/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200/trainer_state.json"
BACKUP_STATE="${ORIGINAL_STATE}.backup_$(date +%Y%m%d_%H%M%S)"

if [ -f "$ORIGINAL_STATE" ]; then
    echo "备份原始trainer_state.json:"
    echo "  $BACKUP_STATE"
    cp "$ORIGINAL_STATE" "$BACKUP_STATE"
    
    echo ""
    echo "修改trainer_state.json (删除wandb_run_id)..."
    
    # 删除wandb_run_id字段,让verl创建新的run
    cat > "$ORIGINAL_STATE" << 'EOF'
{
  "global_steps": 200,
  "kl_coef": 0.001
}
EOF
    
    echo "✓ 已删除 wandb_run_id 字段"
    echo ""
    echo "新的trainer_state.json内容:"
    cat "$ORIGINAL_STATE"
else
    echo "✗ 未找到trainer_state.json: $ORIGINAL_STATE"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  修复完成"
echo "========================================================================"
echo ""
echo "现在可以重新启动训练:"
echo "  1. 停止当前训练 (如果正在运行)"
echo "  2. 删除wandb缓存:"
echo "     rm -rf /ssd1/zz/AI_efficency/RAG/Search-R1/wandb/run-*"
echo "  3. 重新启动:"
echo "     cd /ssd1/zz/AI_efficency/RAG/Search-R1"
echo "     nohup bash train_ppo_from_step200_improved.sh > nohup_step200_newrun.log 2>&1 &"
echo ""
echo "恢复原始文件 (如果需要):"
echo "  cp $BACKUP_STATE $ORIGINAL_STATE"
echo ""
