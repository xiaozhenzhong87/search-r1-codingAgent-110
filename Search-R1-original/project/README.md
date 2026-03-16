# K8s RAG Agent - PPO Training Project

Based on Search-R1 framework with LLM-as-Judge reward function

## Quick Start

### 1. Check Prerequisites
bash project/scripts/check_retrieval_service.sh

### 2. Start Training
cd /ssd1/zz/AI_efficency/RAG/Search-R1
bash project/train_k8s_rag_ppo.sh

### 3. Monitor
tail -f logs/k8s-rag-ppo-qwen2.5-7b-llm-judge_*.log

## Key Components

- **Reward**: LLM-as-Judge (GPT-5) evaluates answer quality
- **Model**: Qwen2.5-7B-Instruct
- **Data**: 2000 K8s QA pairs
- **Retrieval**: E5 + FAISS on port 8001

## Output

Checkpoints: verl_checkpoints/k8s-rag-ppo-qwen2.5-7b-llm-judge/
Logs: logs/k8s-rag-ppo-qwen2.5-7b-llm-judge_*.log
