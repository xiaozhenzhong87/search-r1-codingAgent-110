#!/bin/bash
# 启动K8s检索服务 - 端口8001

INDEX_PATH=/ssd1/zz/AI_efficency/RAG/data/k8s_index/e5_Flat.index
CORPUS_PATH=/ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus.jsonl
RETRIEVER_NAME=e5
RETRIEVER_MODEL=intfloat/e5-base-v2
PORT=8001

echo "===== Starting K8s Retrieval Server ====="
echo "Index: $INDEX_PATH"
echo "Corpus: $CORPUS_PATH"
echo "Port: $PORT"
echo ""

# 使用retriever环境
export HF_HOME=/ssd1/zz/hf_cache
export TRANSFORMERS_CACHE=/ssd1/zz/hf_cache

/ssd1/zz/envs/retriever/bin/python scripts/simple_retrieval_server.py \
    --index_path $INDEX_PATH \
    --corpus_path $CORPUS_PATH \
    --topk 3 \
    --retriever_name $RETRIEVER_NAME \
    --retriever_model $RETRIEVER_MODEL \
    --faiss_gpu \
    --port $PORT
