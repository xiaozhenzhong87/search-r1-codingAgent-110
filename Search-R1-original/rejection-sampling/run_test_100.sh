#!/usr/bin/env bash
# ==========================================================================
#  小规模验证脚本：100 题 × 5 次采样 + 带搜索 + 构建 SFT 数据集
# ==========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 输出目录 ──
TEST_DIR="./output/test_100"
mkdir -p "$TEST_DIR"

SAMPLING_FILE="$TEST_DIR/rejection_sampling.jsonl"
SFT_FILE="$TEST_DIR/sft_dataset.parquet"
LOG_FILE="$TEST_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# ── 可调参数（按需修改） ──
DATA_FILE="../data/nq_hotpot_search/train.parquet"
MAX_QUESTIONS=100
NUM_SAMPLES=5
MAX_TURNS=5
TEMPERATURE=1.0
WORKERS=4
SEARCH_URL="http://127.0.0.1:8000/retrieve"
SEARCH_TOPK=3
MODEL="deploy_gpt5_chat"
MAX_TOKENS=2048

# ── 打印配置 ──
echo "========================================================" | tee "$LOG_FILE"
echo "  Rejection Sampling 小规模测试 (100 题)"                   | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "数据文件:     $DATA_FILE"                                    | tee -a "$LOG_FILE"
echo "问题数:       $MAX_QUESTIONS"                                | tee -a "$LOG_FILE"
echo "每题采样:     $NUM_SAMPLES 次"                               | tee -a "$LOG_FILE"
echo "最大轮数:     $MAX_TURNS"                                    | tee -a "$LOG_FILE"
echo "并发数:       $WORKERS"                                      | tee -a "$LOG_FILE"
echo "检索服务:     $SEARCH_URL (topk=$SEARCH_TOPK)"              | tee -a "$LOG_FILE"
echo "模型:         $MODEL"                                        | tee -a "$LOG_FILE"
echo "采样输出:     $SAMPLING_FILE"                                | tee -a "$LOG_FILE"
echo "SFT 输出:     $SFT_FILE"                                    | tee -a "$LOG_FILE"
echo "日志:         $LOG_FILE"                                     | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ── 检查检索服务是否可用 ──
echo "[检查] 测试检索服务连通性..." | tee -a "$LOG_FILE"
if curl -s "$SEARCH_URL" -H "Content-Type: application/json" \
   -d '{"queries":["test"],"topk":1,"return_scores":true}' \
   --connect-timeout 5 | grep -q '"result"'; then
    echo "[检查] 检索服务正常 ✓" | tee -a "$LOG_FILE"
else
    echo "[警告] 检索服务不可用，将使用 --no_search 模式" | tee -a "$LOG_FILE"
    SEARCH_URL="none"
fi
echo "" | tee -a "$LOG_FILE"

# ── 构建搜索参数 ──
SEARCH_ARGS=()
if [ "$SEARCH_URL" = "none" ]; then
    SEARCH_ARGS+=(--no_search)
else
    SEARCH_ARGS+=(--search_url "$SEARCH_URL" --search_topk "$SEARCH_TOPK")
fi

# ══════════════════════════════════════════════════════════════
#  步骤 1: Rejection Sampling
# ══════════════════════════════════════════════════════════════
echo "========================================================" | tee -a "$LOG_FILE"
echo "  步骤 1/2: Rejection Sampling"                            | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

STEP1_START=$(date +%s)

python step1_rejection_sampling.py \
    --data_file      "$DATA_FILE" \
    --output_file    "$SAMPLING_FILE" \
    --num_samples    "$NUM_SAMPLES" \
    --max_questions  "$MAX_QUESTIONS" \
    --max_turns      "$MAX_TURNS" \
    --temperature    "$TEMPERATURE" \
    --workers        "$WORKERS" \
    --model          "$MODEL" \
    --max_tokens     "$MAX_TOKENS" \
    "${SEARCH_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

STEP1_END=$(date +%s)
STEP1_COST=$((STEP1_END - STEP1_START))
echo "" | tee -a "$LOG_FILE"
echo "[计时] 步骤 1 耗时: ${STEP1_COST}s ($((STEP1_COST/60))min $((STEP1_COST%60))s)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ══════════════════════════════════════════════════════════════
#  步骤 2: 构建 SFT 数据集
# ══════════════════════════════════════════════════════════════
echo "========================================================" | tee -a "$LOG_FILE"
echo "  步骤 2/2: 构建 SFT 数据集"                                | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

python step2_build_sft_dataset.py \
    --input_file  "$SAMPLING_FILE" \
    --output_file "$SFT_FILE" \
    --strategy    best \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"

# ══════════════════════════════════════════════════════════════
#  汇总报告
# ══════════════════════════════════════════════════════════════
TOTAL_END=$(date +%s)
TOTAL_COST=$((TOTAL_END - STEP1_START))

echo "========================================================" | tee -a "$LOG_FILE"
echo "  测试完成 - 汇总"                                          | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# 统计采样结果
python3 -c "
import json, sys
samples = []
with open('$SAMPLING_FILE') as f:
    for line in f:
        samples.append(json.loads(line))
n = len(samples)
qids = set(s['question_id'] for s in samples)
correct = sum(1 for s in samples if s['reward'] >= 1.0)
fmt_ok  = sum(1 for s in samples if s['reward'] > 0)
has_search = sum(1 for s in samples if any(t.get('action')=='search' for t in s.get('turn_log',[])))
avg_turns = sum(s['num_turns'] for s in samples) / max(n, 1)

# 每题是否至少有一个正确
q_correct = set()
for s in samples:
    if s['reward'] >= 1.0:
        q_correct.add(s['question_id'])

print(f'  采样文件:       $SAMPLING_FILE')
print(f'  问题数:         {len(qids)}')
print(f'  总样本数:       {n}')
print(f'  正确样本:       {correct} ({100*correct/max(n,1):.1f}%)')
print(f'  格式正确:       {fmt_ok} ({100*fmt_ok/max(n,1):.1f}%)')
print(f'  涉及搜索:       {has_search} ({100*has_search/max(n,1):.1f}%)')
print(f'  平均轮数:       {avg_turns:.2f}')
print(f'  至少1次答对:    {len(q_correct)}/{len(qids)} ({100*len(q_correct)/max(len(qids),1):.1f}%)')
print()
print(f'  SFT 数据集:     $SFT_FILE')
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "  总耗时: ${TOTAL_COST}s ($((TOTAL_COST/60))min $((TOTAL_COST%60))s)" | tee -a "$LOG_FILE"
echo "  日志:   $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
