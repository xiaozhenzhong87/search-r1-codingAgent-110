#!/usr/bin/env python3
"""直接调用index_builder构建K8s索引"""
import subprocess
import sys
import os

# 设置huggingface缓存目录到/ssd1避免根分区满
os.environ['HF_HOME'] = '/ssd1/zz/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/ssd1/zz/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/ssd1/zz/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cmd = [
    sys.executable,  # 使用当前Python
    "search_r1/search/index_builder.py",
    "--retrieval_method", "e5",
    "--model_path", "intfloat/e5-base-v2",
    "--corpus_path", "/ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus.jsonl",
    "--save_dir", "/ssd1/zz/AI_efficency/RAG/data/k8s_index",
    "--use_fp16",
    "--max_length", "256",
    "--batch_size", "256",
    "--pooling_method", "mean",
    "--faiss_type", "Flat",
    "--save_embedding"
]

print("Running:", " ".join(cmd))
print("HF_HOME:", os.environ['HF_HOME'])
subprocess.run(cmd, cwd="/ssd1/zz/AI_efficency/RAG/Search-R1")
