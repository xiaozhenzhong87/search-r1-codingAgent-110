#!/bin/bash
# Test PPO step-400 checkpoint output format stability
#
# Prerequisites:
#   - Retrieval server running at http://127.0.0.1:8000/retrieve
#   - Single GPU available
#
# Usage:
#   bash test_ppo_format.sh           # default 30 samples
#   bash test_ppo_format.sh 50        # 50 samples with verbose

set -e
cd /ssd1/zz/AI_efficency/RAG/Search-R1

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS

NUM_SAMPLES=${1:-30}
MODEL_PATH="/ssd1/zz/AI_efficency/RAG/Search-R1/verl_checkpoints/nq-hp-ppo-sft-fixed-7b/actor/global_step_300"
PYTHON="/ssd1/zz/envs/searchr1/bin/python"
SAVE_FILE="results_ppo_step400_format_$(date +%Y%m%d_%H%M%S).jsonl"

echo "========================================="
echo "  PPO Step-400 Format Stability Test"
echo "  Model: $MODEL_PATH"
echo "  Samples: $NUM_SAMPLES"
echo "  Save to: $SAVE_FILE"
echo "========================================="

# Check retrieval server
echo "Checking retrieval server..."
if curl -s --max-time 5 http://127.0.0.1:8000/retrieve -X POST \
    -H "Content-Type: application/json" \
    -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
    echo "  Retrieval server is running."
else
    echo "  WARNING: Retrieval server not reachable at http://127.0.0.1:8000/retrieve"
    echo "  Please start it first. Exiting."
    exit 1
fi

$PYTHON test_ppo_format.py \
    --model_path "$MODEL_PATH" \
    -n "$NUM_SAMPLES" \
    --save "$SAVE_FILE" \
    -v
