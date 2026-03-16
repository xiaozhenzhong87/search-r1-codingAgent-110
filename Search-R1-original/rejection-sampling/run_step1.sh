#!/usr/bin/env bash
# Step 1: Rejection Sampling
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_FILE="${DATA_FILE:-../data/nq_hotpot_search/train.parquet}"
OUTPUT_FILE="${OUTPUT_FILE:-./output/rejection_sampling_results.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
MAX_TURNS="${MAX_TURNS:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
WORKERS="${WORKERS:-4}"
API_URL="${API_URL:-http://osagw.simeji.me/gbu/rest/v1/ai_chat/openai_service}"
API_KEY="${API_KEY:-mediago_platform.oijio3f4893u2898}"
MODEL="${MODEL:-deploy_gpt5_chat}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
SEARCH_URL="${SEARCH_URL:-}"
SEARCH_TOPK="${SEARCH_TOPK:-3}"

CMD=(python step1_rejection_sampling.py
    --data_file    "$DATA_FILE"
    --output_file  "$OUTPUT_FILE"
    --num_samples  "$NUM_SAMPLES"
    --max_turns    "$MAX_TURNS"
    --temperature  "$TEMPERATURE"
    --workers      "$WORKERS"
    --api_url      "$API_URL"
    --api_key      "$API_KEY"
    --model        "$MODEL"
    --max_tokens   "$MAX_TOKENS"
    --search_topk  "$SEARCH_TOPK"
)

if [ -n "$MAX_QUESTIONS" ]; then
    CMD+=(--max_questions "$MAX_QUESTIONS")
fi

if [ -z "$SEARCH_URL" ] || [ "$SEARCH_URL" = "none" ]; then
    CMD+=(--no_search)
else
    CMD+=(--search_url "$SEARCH_URL")
fi

echo "========================================"
echo "  Rejection Sampling - Step 1"
echo "========================================"
echo "Data:        $DATA_FILE"
echo "Output:      $OUTPUT_FILE"
echo "Samples/Q:   $NUM_SAMPLES"
echo "Max turns:   $MAX_TURNS"
echo "Model:       $MODEL"
echo "Workers:     $WORKERS"
echo "Search URL:  ${SEARCH_URL:-disabled}"
echo "========================================"
echo ""

"${CMD[@]}" 2>&1 | tee "step1_$(date +%Y%m%d_%H%M%S).log"
