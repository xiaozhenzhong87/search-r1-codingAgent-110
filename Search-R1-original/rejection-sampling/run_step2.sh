#!/usr/bin/env bash
# Step 2: Build SFT Dataset
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_FILE="${INPUT_FILE:-./output/rejection_sampling_results.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-./output/sft_dataset.parquet}"
STRATEGY="${STRATEGY:-best}"
MIN_REWARD="${MIN_REWARD:-0.0}"

echo "========================================"
echo "  Build SFT Dataset - Step 2"
echo "========================================"
echo "Input:       $INPUT_FILE"
echo "Output:      $OUTPUT_FILE"
echo "Strategy:    $STRATEGY"
echo "Min reward:  $MIN_REWARD"
echo "========================================"
echo ""

python step2_build_sft_dataset.py \
    --input_file  "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --strategy    "$STRATEGY" \
    --min_reward  "$MIN_REWARD" \
    2>&1 | tee "step2_$(date +%Y%m%d_%H%M%S).log"
