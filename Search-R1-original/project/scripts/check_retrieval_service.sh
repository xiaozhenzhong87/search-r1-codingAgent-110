#!/bin/bash
# Check K8s Retrieval Service Status

RETRIEVER_URL="http://127.0.0.1:8001/retrieve"

echo "Checking K8s retrieval service at $RETRIEVER_URL..."

if curl -s -X POST "$RETRIEVER_URL" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a Pod?","topk":1}' | grep -q "result"; then
    echo "✓ Retrieval service is RUNNING"
    exit 0
else
    echo "✗ Retrieval service is NOT accessible"
    echo ""
    echo "To start the service:"
    echo "  cd /ssd1/zz/AI_efficency/RAG/Search-R1/scripts"
    echo "  bash k8s_retrieval_launch.sh"
    exit 1
fi
