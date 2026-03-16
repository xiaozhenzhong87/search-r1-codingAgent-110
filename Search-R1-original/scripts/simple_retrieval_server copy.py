#!/usr/bin/env python3
"""简单的K8s检索服务器 - 基于FAISS索引提供HTTP API"""
import os
import argparse
import json
from typing import List
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

class SearchRequest(BaseModel):
    query: str
    topk: int = 3

@app.post("/retrieve")
def retrieve(request: SearchRequest):
    try:
        results = retriever.batch_search(
            query_list=[request.query],
            num=request.topk,
            return_score=False
        )
        if results and len(results) > 0:
            return {"result": results[0]}
        else:
            return {"result": []}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "result": []}

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
    print(f"Test: curl -X POST http://localhost:{args.port}/retrieve -H 'Content-Type: application/json' -d '{{\"query\":\"What is a Pod?\"}}'")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
