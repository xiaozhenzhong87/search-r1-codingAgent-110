#!/bin/bash
#
# 重启 K8s 检索服务（支持批量查询）
#

set -euo pipefail

echo "=========================================="
echo "  重启 K8s 检索服务"
echo "=========================================="
echo ""

# 停止旧服务
echo "1. 停止旧的检索服务..."
# 查找所有在 8001 端口的 Python 进程
OLD_PIDS=$(ps aux | grep python | grep -E "(8001|retrieval|retrieve)" | grep -v grep | awk '{print $2}' || echo "")
if [ -n "$OLD_PIDS" ]; then
    for PID in $OLD_PIDS; do
        echo "   找到旧服务 PID: $PID"
        kill $PID 2>/dev/null || echo "   (PID $PID 可能已停止)"
    done
    sleep 3
    echo "   ✅ 旧服务已停止"
else
    echo "   ℹ️  未找到运行中的检索服务"
fi
echo ""

# 设置 Python 路径
echo "2. 设置 Python 环境..."
PYTHON_BIN="/ssd1/zz/envs/retriever/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="/ssd1/zz/envs/searchr1/bin/python"
fi
if [ ! -f "$PYTHON_BIN" ]; then
    echo "   ❌ 找不到 Python: $PYTHON_BIN"
    exit 1
fi
echo "   ✅ Python: $PYTHON_BIN"
echo ""

# 启动新服务
echo "3. 启动新的检索服务..."
cd /ssd1/zz/AI_efficency/RAG/Search-R1
nohup $PYTHON_BIN project/k8s_retrieval_server.py > /tmp/k8s_retrieval_batch.log 2>&1 &
NEW_PID=$!
echo "   ✅ 新服务 PID: $NEW_PID"
echo ""

# 等待服务启动
echo "4. 等待服务启动..."
sleep 5

# 检查服务状态
echo "5. 检查服务状态..."
if curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "   ✅ 服务启动成功!"
    echo ""
    
    # 测试单查询
    echo "6. 测试单查询..."
    curl -s -X POST http://127.0.0.1:8001/retrieve \
        -H "Content-Type: application/json" \
        -d '{"query": "What is a Pod?", "topk": 1}' | jq -r '.result[0].id' 2>/dev/null || echo "   ⚠️  单查询测试失败"
    
    # 测试批量查询
    echo "7. 测试批量查询..."
    BATCH_TEST=$(curl -s -X POST http://127.0.0.1:8001/retrieve \
        -H "Content-Type: application/json" \
        -d '{"queries": ["What is a Pod?", "What is a Service?"], "topk": 1, "return_scores": false}')
    
    if echo "$BATCH_TEST" | jq -e '.result' > /dev/null 2>&1; then
        RESULT_COUNT=$(echo "$BATCH_TEST" | jq '.result | length')
        echo "   ✅ 批量查询成功! 返回 $RESULT_COUNT 个结果"
    else
        echo "   ❌ 批量查询失败:"
        echo "$BATCH_TEST" | jq . 2>/dev/null || echo "$BATCH_TEST"
    fi
    
    echo ""
    echo "=========================================="
    echo "  检索服务已就绪!"
    echo "=========================================="
    echo "PID: $NEW_PID"
    echo "Port: 8001"
    echo "Log: /tmp/k8s_retrieval_batch.log"
    echo ""
    echo "查看日志: tail -f /tmp/k8s_retrieval_batch.log"
    echo "停止服务: kill $NEW_PID"
    echo "=========================================="
else
    echo "   ❌ 服务启动失败!"
    echo ""
    echo "查看错误日志:"
    tail -50 /tmp/k8s_retrieval_batch.log
    exit 1
fi
