#!/usr/bin/env python3
"""简单的K8s检索服务器 - 基于FAISS索引提供HTTP API（兼容批量请求）"""
import os
import argparse
import json
import traceback
from typing import List, Optional
from dataclasses import dataclass
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 设置环境变量
os.environ['HF_HOME'] = '/ssd1/zz/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/ssd1/zz/hf_cache'

# 导入retrieval模块
import sys
sys.path.append('/ssd1/zz/AI_efficency/RAG/Search-R1')
from search_r1.search.retrieval import DenseRetriever

@dataclass
class RetrieverConfig:
    index_path: str
    corpus_path: str
    retrieval_method: str
    retrieval_model_path: str
    retrieval_topk: int
    faiss_gpu: bool
    retrieval_pooling_method: str = "mean"
    retrieval_query_max_length: int = 256
    retrieval_use_fp16: bool = True
    retrieval_batch_size: int = 512

app = FastAPI()
retriever = None

# 兼容训练代码的请求格式：支持单query和批量queries
class SearchRequest(BaseModel):
    queries: Optional[List[str]] = None  # 批量查询（训练代码用）
    query: Optional[str] = None          # 单查询（兼容测试）
    topk: int = 3                        # 兼容topk参数

@app.post("/retrieve")
def retrieve(request: SearchRequest):
    try:
        # 1. 解析请求：优先批量queries，其次单query
        if request.queries is not None and len(request.queries) > 0:
            query_list = request.queries
        elif request.query is not None:
            query_list = [request.query]
        else:
            return {"result": []}  # 无查询时返回空result
        
        # 2. 执行批量检索
        results = retriever.batch_search(
            query_list=query_list,
            num=request.topk,
            return_score=False
        )
        
        # 3. 严格对齐训练代码的响应格式：{"result": [查询1结果, 查询2结果]}
        # 确保results是二维列表（即使只有1个查询）
        if isinstance(results, list) and len(results) == len(query_list):
            resp = {"result": results}
        else:
            # 兼容retriever返回格式异常的情况
            resp = {"result": [results] if isinstance(results, list) else []}
        
        return resp
    
    except Exception as e:
        # 异常兜底：确保始终返回result字段
        print(f"Retrieval error: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "result": []  # 空result避免训练代码KeyError
        }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="e5")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    print(f"Loading K8s retriever...")
    print(f"  Index: {args.index_path}")
    print(f"  Corpus: {args.corpus_path}")
    print(f"  Model: {args.retriever_model}")
    print(f"  Port: {args.port}")
    print(f"  GPU: {args.faiss_gpu}")
    
    config = RetrieverConfig(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_method=args.retriever_name,
        retrieval_model_path=args.retriever_model,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512
    )
    
    print("Initializing DenseRetriever...")
    retriever = DenseRetriever(config)
    
    print(f"\n✅ K8s Retrieval Server started on port {args.port}")
    print(f"Test single query: curl -X POST http://localhost:{args.port}/retrieve -H 'Content-Type: application/json' -d '{{\"query\":\"What is a Pod?\"}}'")
    print(f"Test batch queries: curl -X POST http://localhost:{args.port}/retrieve -H 'Content-Type: application/json' -d '{{\"queries\":[\"What is a Pod?\", \"What is K8s?\"], \"topk\":3}}'")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)