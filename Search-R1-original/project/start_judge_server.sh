#!/bin/bash
#
# 启动 LLM Judge 服务
# 使用 vLLM 提供 OpenAI 兼容的 API
#

set -euo pipefail

# ========== 配置 ==========
export JUDGE_MODEL="/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct"
export JUDGE_PORT=8002
export CUDA_VISIBLE_DEVICES=7  # 使用最后一张 GPU

# ========== 检查模型路径 ==========
if [ ! -d "$JUDGE_MODEL" ]; then
    echo "❌ 错误: 找不到模型 $JUDGE_MODEL"
    exit 1
fi

echo "=========================================="
echo "  启动 LLM Judge 服务"
echo "=========================================="
echo "📦 模型: $JUDGE_MODEL"
echo "🔌 端口: $JUDGE_PORT"
echo "🎮 GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# ========== 检查端口是否被占用 ==========
if lsof -Pi :$JUDGE_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  警告: 端口 $JUDGE_PORT 已被占用"
    echo "尝试停止现有服务..."
    PID=$(lsof -Pi :$JUDGE_PORT -sTCP:LISTEN -t)
    kill $PID 2>/dev/null || true
    sleep 2
fi

# ========== 启动 vLLM 服务 ==========
echo "🚀 启动 vLLM Judge 服务..."
echo ""

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $JUDGE_MODEL \
    --port $JUDGE_PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --disable-log-requests \
    > /tmp/judge_server_$JUDGE_PORT.log 2>&1 &

JUDGE_PID=$!
echo "Judge 服务已启动 (PID: $JUDGE_PID)"
echo "日志文件: /tmp/judge_server_$JUDGE_PORT.log"

# ========== 等待服务启动 ==========
echo ""
echo "等待服务启动..."
sleep 10

# ========== 测试服务 ==========
echo "测试 Judge 服务..."
TEST_RESPONSE=$(curl -s -X POST http://127.0.0.1:$JUDGE_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$JUDGE_MODEL"'",
        "messages": [{"role": "user", "content": "Rate this: answer=kubernetes, ground_truth=kubernetes. Score:"}],
        "temperature": 0,
        "max_tokens": 10
    }' 2>&1)

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    echo "✅ Judge 服务正常运行!"
    echo ""
    echo "=========================================="
    echo "  Judge 服务信息"
    echo "=========================================="
    echo "📍 API 地址: http://127.0.0.1:$JUDGE_PORT/v1"
    echo "📊 查看日志: tail -f /tmp/judge_server_$JUDGE_PORT.log"
    echo "🛑 停止服务: kill $JUDGE_PID"
    echo "=========================================="
else
    echo "❌ 服务启动失败,查看日志:"
    tail -20 /tmp/judge_server_$JUDGE_PORT.log
    exit 1
fi
