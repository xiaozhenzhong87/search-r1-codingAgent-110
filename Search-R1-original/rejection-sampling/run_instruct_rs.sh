#!/usr/bin/env bash
# ==========================================================================
#  vLLM Rejection Sampling — Qwen2.5-7B-Instruct 基座模型
#  使用 2 个 vLLM 实例 (TP=4, 各占4卡) 做负载均衡
# ==========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 输出目录 ──
TEST_DIR="./output/instruct_rs_5k"
mkdir -p "$TEST_DIR"

SAMPLING_FILE="$TEST_DIR/rejection_sampling.jsonl"
SFT_FILE="$TEST_DIR/sft_dataset.parquet"
LOG_FILE="$TEST_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# ── vLLM 配置 (两个实例，逗号分隔) ──
VLLM_URLS="http://localhost:8002,http://localhost:8003"
VLLM_MODEL="qwen-7b-instruct"
VLLM_API_KEY="EMPTY"

# ── 采样参数 ──
DATA_FILE="../data/nq_hotpot_search/train.parquet"
MAX_QUESTIONS=5000
NUM_SAMPLES=5
MAX_TURNS=5
TEMPERATURE=1.0
WORKERS=16
SEARCH_URL="http://127.0.0.1:8000/retrieve"
SEARCH_TOPK=3
MAX_TOKENS=2048

# ── 打印配置 ──
echo "========================================================" | tee "$LOG_FILE"
echo "  Instruct RS (Qwen2.5-7B-Instruct)"                      | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "vLLM URLs:      $VLLM_URLS"                                | tee -a "$LOG_FILE"
echo "模型:           $VLLM_MODEL"                               | tee -a "$LOG_FILE"
echo "数据文件:       $DATA_FILE"                                 | tee -a "$LOG_FILE"
echo "问题数:         $MAX_QUESTIONS"                             | tee -a "$LOG_FILE"
echo "每题采样:       $NUM_SAMPLES 次"                            | tee -a "$LOG_FILE"
echo "最大轮数:       $MAX_TURNS"                                 | tee -a "$LOG_FILE"
echo "并发数:         $WORKERS"                                   | tee -a "$LOG_FILE"
echo "检索服务:       $SEARCH_URL (topk=$SEARCH_TOPK)"           | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ── 检查 vLLM 服务 ──
echo "[检查] 测试 vLLM 服务连通性..." | tee -a "$LOG_FILE"
OK=0
for PORT in 8002 8003; do
    HEALTH=$(curl -s "http://localhost:${PORT}/v1/models" --connect-timeout 5 2>/dev/null || echo "FAIL")
    if echo "$HEALTH" | grep -q "$VLLM_MODEL"; then
        echo "[检查] vLLM :${PORT} 正常 ($VLLM_MODEL)" | tee -a "$LOG_FILE"
        OK=$((OK+1))
    else
        echo "[警告] vLLM :${PORT} 不可用" | tee -a "$LOG_FILE"
    fi
done
if [ "$OK" -eq 0 ]; then
    echo "[错误] 没有可用的 vLLM 服务" | tee -a "$LOG_FILE"
    exit 1
fi

# ── 检查检索服务 ──
echo "[检查] 测试检索服务连通性..." | tee -a "$LOG_FILE"
SEARCH_ARGS=()
if curl -s "$SEARCH_URL" -H "Content-Type: application/json" \
   -d '{"queries":["test"],"topk":1,"return_scores":true}' \
   --connect-timeout 5 | grep -q '"result"'; then
    echo "[检查] 检索服务正常" | tee -a "$LOG_FILE"
    SEARCH_ARGS=(--search_url "$SEARCH_URL" --search_topk "$SEARCH_TOPK")
else
    echo "[警告] 检索服务不可用，将使用 --no_search 模式" | tee -a "$LOG_FILE"
    SEARCH_ARGS=(--no_search)
fi
echo "" | tee -a "$LOG_FILE"

# ══════════════════════════════════════════════════════════════
#  步骤 1: Rejection Sampling (vLLM chat backend)
# ══════════════════════════════════════════════════════════════
echo "========================================================" | tee -a "$LOG_FILE"
echo "  步骤 1/2: Rejection Sampling (vLLM chat)"                | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

STEP1_START=$(date +%s)

python step1_rejection_sampling.py \
    --backend        vllm_chat \
    --data_file      "$DATA_FILE" \
    --output_file    "$SAMPLING_FILE" \
    --num_samples    "$NUM_SAMPLES" \
    --max_questions  "$MAX_QUESTIONS" \
    --max_turns      "$MAX_TURNS" \
    --temperature    "$TEMPERATURE" \
    --workers        "$WORKERS" \
    --api_url        "$VLLM_URLS" \
    --api_key        "$VLLM_API_KEY" \
    --model          "$VLLM_MODEL" \
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
echo "  完成 - 汇总"                                              | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

python3 -c "
import json
samples = []
with open('$SAMPLING_FILE') as f:
    for line in f:
        samples.append(json.loads(line))
n = len(samples)
qids = set(s['question_id'] for s in samples)
correct = sum(1 for s in samples if s['reward'] >= 1.0)
has_search = sum(1 for s in samples if any(t.get('action')=='search' for t in s.get('turn_log',[])))
avg_turns = sum(s['num_turns'] for s in samples) / max(n, 1)
q_correct = set(s['question_id'] for s in samples if s['reward'] >= 1.0)
has_think = sum(1 for s in samples if '<think>' in s.get('response',''))
has_answer = sum(1 for s in samples if '<answer>' in s.get('response',''))

print(f'  问题数:         {len(qids)}')
print(f'  总样本数:       {n}')
print(f'  正确样本:       {correct} ({100*correct/max(n,1):.1f}%)')
print(f'  含<think>:      {has_think} ({100*has_think/max(n,1):.1f}%)')
print(f'  含<answer>:     {has_answer} ({100*has_answer/max(n,1):.1f}%)')
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
