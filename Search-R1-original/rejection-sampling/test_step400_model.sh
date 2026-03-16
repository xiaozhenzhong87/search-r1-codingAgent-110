#!/usr/bin/env bash
# ==========================================================================
#  测试 global_step_400 模型：vLLM 四卡部署 + 100 测试样本
# ==========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python 环境 ──
PYTHON_ENV="/ssd1/zz/envs/searchr1/bin/python"

# ── 模型路径 ──
MODEL_PATH="/ssd1/zz/AI_efficency/RAG/Search-R1/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl-from-step200/actor/global_step_400"

# ── 输出目录 ──
TEST_DIR="./output/test_step400_100samples"
mkdir -p "$TEST_DIR"

SAMPLING_FILE="$TEST_DIR/rejection_sampling.jsonl"
SFT_FILE="$TEST_DIR/sft_dataset.parquet"
LOG_FILE="$TEST_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# ── vLLM 配置 ──
VLLM_PORT=8003
VLLM_URL="http://localhost:${VLLM_PORT}"
VLLM_MODEL="step400-test"
VLLM_API_KEY="EMPTY"
GPUS="4,5,6,7"  # 使用4-7卡，避免与0-3卡冲突

# ── 采样参数 ──
DATA_FILE="../data/nq_hotpot_search/train.parquet"
MAX_QUESTIONS=100
NUM_SAMPLES=5
MAX_TURNS=5
TEMPERATURE=1.0
WORKERS=8
SEARCH_URL="http://127.0.0.1:8000/retrieve"
SEARCH_TOPK=3
MAX_TOKENS=2048

# ── 打印配置 ──
echo "========================================================" | tee "$LOG_FILE"
echo "  测试 global_step_400 模型 (100 样本)"                   | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "模型路径:     $MODEL_PATH"                                 | tee -a "$LOG_FILE"
echo "使用GPU:      $GPUS"                                      | tee -a "$LOG_FILE"
echo "vLLM端口:     $VLLM_PORT"                                 | tee -a "$LOG_FILE"
echo "数据文件:     $DATA_FILE"                                  | tee -a "$LOG_FILE"
echo "问题数:       $MAX_QUESTIONS"                              | tee -a "$LOG_FILE"
echo "每题采样:     $NUM_SAMPLES 次"                             | tee -a "$LOG_FILE"
echo "最大轮数:     $MAX_TURNS"                                  | tee -a "$LOG_FILE"
echo "并发数:       $WORKERS"                                    | tee -a "$LOG_FILE"
echo "检索服务:     $SEARCH_URL (topk=$SEARCH_TOPK)"            | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ══════════════════════════════════════════════════════════════
#  步骤 0: 启动 vLLM 服务
# ══════════════════════════════════════════════════════════════
echo "========================================================" | tee -a "$LOG_FILE"
echo "  步骤 0: 启动 vLLM 服务 (4卡)"                            | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# 检查端口是否被占用
if lsof -Pi :${VLLM_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[警告] 端口 $VLLM_PORT 已被占用" | tee -a "$LOG_FILE"
    PID=$(lsof -Pi :${VLLM_PORT} -sTCP:LISTEN -t)
    echo "[警告] 占用进程 PID: $PID" | tee -a "$LOG_FILE"
    echo "[警告] 如需停止，请手动执行: kill $PID" | tee -a "$LOG_FILE"
    echo "[信息] 将尝试使用已有服务..." | tee -a "$LOG_FILE"
else
    echo "[启动] 启动 vLLM 服务..." | tee -a "$LOG_FILE"
    VLLM_LOG="$TEST_DIR/vllm_server.log"
    
    CUDA_VISIBLE_DEVICES=$GPUS nohup $PYTHON_ENV -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        > "$VLLM_LOG" 2>&1 &
    
    VLLM_PID=$!
    echo "[启动] vLLM 服务 PID: $VLLM_PID" | tee -a "$LOG_FILE"
    echo "[启动] 日志文件: $VLLM_LOG" | tee -a "$LOG_FILE"
    echo "[等待] 等待服务启动 (60秒)..." | tee -a "$LOG_FILE"
    sleep 60
fi

# ── 检查 vLLM 服务 ──
echo "[检查] 测试 vLLM 服务连通性..." | tee -a "$LOG_FILE"
MAX_RETRY=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRY ]; do
    HEALTH=$(curl -s "${VLLM_URL}/v1/models" --connect-timeout 5 2>/dev/null || echo "FAIL")
    if echo "$HEALTH" | grep -q "$VLLM_MODEL"; then
        echo "[检查] vLLM 服务正常 ($VLLM_MODEL) ✓" | tee -a "$LOG_FILE"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "[检查] 第 $RETRY_COUNT/$MAX_RETRY 次尝试失败，等待 10 秒..." | tee -a "$LOG_FILE"
        sleep 10
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRY ]; then
    echo "[错误] vLLM 服务不可用" | tee -a "$LOG_FILE"
    exit 1
fi

# ── 检查检索服务 ──
echo "[检查] 测试检索服务连通性..." | tee -a "$LOG_FILE"
SEARCH_ARGS=()
if curl -s "$SEARCH_URL" -H "Content-Type: application/json" \
   -d '{"queries":["test"],"topk":1,"return_scores":true}' \
   --connect-timeout 5 | grep -q '"result"'; then
    echo "[检查] 检索服务正常 ✓" | tee -a "$LOG_FILE"
    SEARCH_ARGS=(--search_url "$SEARCH_URL" --search_topk "$SEARCH_TOPK")
else
    echo "[警告] 检索服务不可用，将使用 --no_search 模式" | tee -a "$LOG_FILE"
    SEARCH_ARGS=(--no_search)
fi
echo "" | tee -a "$LOG_FILE"

# ══════════════════════════════════════════════════════════════
#  步骤 1: Rejection Sampling (vLLM backend)
# ══════════════════════════════════════════════════════════════
echo "========================================================" | tee -a "$LOG_FILE"
echo "  步骤 1/2: Rejection Sampling (vLLM)"                    | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

STEP1_START=$(date +%s)

$PYTHON_ENV step1_rejection_sampling.py \
    --backend        vllm \
    --data_file      "$DATA_FILE" \
    --output_file    "$SAMPLING_FILE" \
    --num_samples    "$NUM_SAMPLES" \
    --max_questions  "$MAX_QUESTIONS" \
    --max_turns      "$MAX_TURNS" \
    --temperature    "$TEMPERATURE" \
    --workers        "$WORKERS" \
    --api_url        "$VLLM_URL" \
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

$PYTHON_ENV step2_build_sft_dataset.py \
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

print(f'  采样文件:       $SAMPLING_FILE')
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

echo ""
echo "测试完成！查看结果："
echo "  - 采样文件: $SAMPLING_FILE"
echo "  - SFT文件:  $SFT_FILE"
echo "  - 日志:     $LOG_FILE"
echo ""
echo "如需停止 vLLM 服务，请运行："
echo "  lsof -ti:${VLLM_PORT} | xargs kill -9"
