#!/usr/bin/env python3
"""
K8s 检索服务 - 支持批量查询
用于 Search-R1 训练框架
"""

import sys
sys.path.insert(0, '/ssd1/zz/AI_efficency/RAG/Search-R1')

from search_r1.search.retrieval import DenseRetriever
from dataclasses import dataclass
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

@dataclass
class Config:
    index_path: str = '/ssd1/zz/AI_efficency/RAG/data/k8s_index/e5_Flat.index'
    corpus_path: str = '/ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus.jsonl'
    retrieval_method: str = 'e5'
    retrieval_model_path: str = 'intfloat/e5-base-v2'
    retrieval_topk: int = 3
    faiss_gpu: bool = True
    retrieval_pooling_method: str = 'mean'
    retrieval_query_max_length: int = 256
    retrieval_use_fp16: bool = True
    retrieval_batch_size: int = 512

app = FastAPI()

print('=' * 60)
print('Loading K8s Dense Retriever...')
print('=' * 60)
retriever = DenseRetriever(Config())
print('✅ Retriever loaded successfully!')
print('=' * 60)

class SingleQueryRequest(BaseModel):
    """单查询请求（兼容旧接口）"""
    query: str
    topk: int = 3
    top_k: Optional[int] = None  # 兼容不同命名

class BatchQueryRequest(BaseModel):
    """批量查询请求（Search-R1 训练需要）"""
    queries: List[str]
    topk: int = 3
    return_scores: bool = False

@app.post('/retrieve')
def retrieve(request: dict):
    """
    统一检索接口，支持单查询和批量查询
    
    单查询格式:
        {"query": "xxx", "topk": 3}
    
    批量查询格式:
        {"queries": ["xxx", "yyy"], "topk": 3, "return_scores": false}
    """
    try:
        # 判断是单查询还是批量查询
        if 'queries' in request:
            # 批量查询模式
            queries = request['queries']
            topk = request.get('topk', 3)
            return_scores = request.get('return_scores', False)
            
            # 批量检索
            results = retriever.batch_search(
                query_list=queries, 
                num=topk, 
                return_score=return_scores
            )
            
            # 处理返回格式
            if return_scores:
                # return_score=True 时返回 (documents, scores) 元组
                documents, scores = results
                return {'result': [documents, scores]}
            else:
                # return_score=False 时只返回 documents 列表
                return {'result': results}
        
        elif 'query' in request:
            # 单查询模式（兼容）
            query = request['query']
            topk = request.get('topk') or request.get('top_k', 3)
            
            # 单个检索
            results = retriever.batch_search(
                query_list=[query], 
                num=topk, 
                return_score=False
            )
            
            return {'result': results[0] if results else []}
        
        else:
            return {'error': 'Missing required field: query or queries', 'result': []}
    
    except Exception as e:
        print(f"❌ Error in retrieve: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'result': []}

@app.get('/health')
def health():
    """健康检查"""
    return {
        'status': 'ok',
        'service': 'k8s-retrieval',
        'index': Config().index_path,
        'corpus': Config().corpus_path
    }

@app.get('/')
def root():
    """根路径信息"""
    return {
        'service': 'K8s Dense Retrieval Server',
        'endpoints': {
            '/retrieve': 'POST - 检索接口（支持单查询和批量查询）',
            '/health': 'GET - 健康检查'
        },
        'examples': {
            'single_query': {
                'url': '/retrieve',
                'method': 'POST',
                'body': {'query': 'What is a Pod?', 'topk': 3}
            },
            'batch_query': {
                'url': '/retrieve',
                'method': 'POST', 
                'body': {'queries': ['What is a Pod?', 'What is a Service?'], 'topk': 3, 'return_scores': False}
            }
        }
    }

if __name__ == '__main__':
    print('🚀 Starting K8s Retrieval Server on port 8001...')
    print('   - Single query:  POST /retrieve {"query": "xxx", "topk": 3}')
    print('   - Batch query:   POST /retrieve {"queries": [...], "topk": 3}')
    print('   - Health check:  GET  /health')
    print('=' * 60)
    
    uvicorn.run(app, host='0.0.0.0', port=8001)
