#!/bin/bash
# 一键启动训练 - 最简单版本
# 直接运行，不做任何环境检查（假设已在 searchr1 环境中）

cd /ssd1/zz/AI_efficency/RAG/Search-R1
nohup bash project/train_k8s_rag_search_r1.sh > project/training.log 2>&1 &
echo "训练已启动! PID: $!"
echo "查看日志: tail -f project/training.log"
