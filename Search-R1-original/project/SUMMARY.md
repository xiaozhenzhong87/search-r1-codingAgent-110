# K8s RAG Agent - Phase 1&2 完成总结

## ✅ 已完成任务

### 1. 数据基础设施 (Phase 1)

**K8s文档处理**:
- 清洗: 1732个K8s文档chunks
- 方法: Hugo Front Matter清除 + 智能分块(500-800 tokens)
- 输出: `/ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus.jsonl`

**检索索引**:
- 模型: E5-base-v2 (dense retrieval)
- 索引: FAISS Flat (GPU加速)
- 位置: `/ssd1/zz/AI_efficency/RAG/data/k8s_index/`

**检索服务**:
- 端口: 8001
- 状态: ✓ 运行中
- 测试: `curl -X POST http://localhost:8001/retrieve -d '{"query":"What is Pod?","topk":3}'`

### 2. 训练数据生成 (Phase 2)

**QA生成**:
- 总数: 2000条英文QA对 (1000 chunks × 2)
- 成功率: 100% (1000/1000)
- API: GPT-5 (Few-shot prompting)

**数据质量**:
- 难度分布:
  * Easy: 209 (10.4%)
  * Medium: 1729 (86.5%)
  * Hard: 62 (3.1%)
- 问题类型:
  * Concept: 934 (46.7%)
  * Troubleshoot: 601 (30.0%)
  * Best-practice: 399 (20.0%)
  * Comparison: 66 (3.3%)

**数据格式**:
```json
{
  "data_source": "k8s_concepts",
  "prompt": [{
    "role": "user",
    "content": "Answer the given question... Question: ..."
  }],
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "target": ["key point 1", "key point 2", ...]
    }
  },
  "extra_info": {
    "chunk_id": "...",
    "difficulty": "medium",
    "question_type": "concept",
    "doc_path": "workloads/pods/...",
    "target_answer": "Complete reference answer for LLM judge"
  }
}
```

### 3. PPO训练框架 (Phase 3准备)

**LLM-as-Judge Reward**:
- 位置: `project/reward/llm_judge_reward.py`
- 评分标准:
  * Correctness (0-4分): 准确性
  * Completeness (0-3分): 完整性
  * Clarity (0-2分): 清晰度
  * Conciseness (0-1分): 简洁性
- 归一化: 0-10分 → [0, 1]
- 惩罚: 答案后多余输出 -20%

**训练脚本**:
- 位置: `project/train_k8s_rag_ppo.sh`
- 基础模型: Qwen2.5-7B-Instruct
- 训练步数: 300 steps
- 批次大小: 256 (train), 128 (val)
- 检索: E5 + FAISS (topk=3)

## 📁 输出文件位置

### 数据文件
```
/ssd1/zz/AI_efficency/RAG/data/
├── k8s-concepts-corpus.jsonl                    # 文档corpus
├── k8s-concepts-corpus_with_metadata.jsonl      # 带元数据的corpus
├── k8s_index/
│   └── e5_Flat.index                            # FAISS索引
└── k8s_qa_data_en/
    ├── raw_qa_2000.jsonl                        # 原始QA (1000行)
    ├── qa_2000_searchr1_format.json             # Search-R1格式 (2000条)
    ├── raw_qa_100.jsonl                         # 验证集原始QA (50行)
    └── qa_100_searchr1_format.json              # 验证集 (100条)
```

### 项目文件
```
/ssd1/zz/AI_efficency/RAG/Search-R1/project/
├── reward/
│   └── llm_judge_reward.py                      # LLM-as-Judge reward
├── scripts/
│   ├── check_retrieval_service.sh               # 检查检索服务
│   └── convert_data_format.py                   # 数据格式转换
├── config/                                       # 配置文件
├── train_k8s_rag_ppo.sh                         # PPO训练脚本
├── README.md                                     # 使用文档
└── SUMMARY.md                                    # 本文件
```

### 脚本文件
```
/ssd1/zz/AI_efficency/RAG/Search-R1/scripts/
└── data_process/
    ├── k8s_corpus_builder.py                    # Corpus构建
    ├── k8s_qa_synthesis_en.py                   # QA生成(英文)
    └── simple_retrieval_server.py               # 检索服务
```

## 🚀 启动训练

### 前置条件

1. **检索服务运行**:
```bash
# 检查服务
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a Pod?","topk":3}'

# 如未运行,启动服务
cd /ssd1/zz/AI_efficency/RAG/Search-R1/scripts
python simple_retrieval_server.py --port 8001 \
  --index_path /ssd1/zz/AI_efficency/RAG/data/k8s_index/e5_Flat.index \
  --corpus_path /ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus.jsonl
```

2. **训练数据就绪**: ✓ 已完成
3. **GPU可用**: 8×GPU (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)

### 启动训练

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1
bash project/train_k8s_rag_ppo.sh
```

### 监控训练

```bash
# 查看实时日志
tail -f logs/k8s-rag-ppo-qwen2.5-7b-llm-judge_*.log

# 查看checkpoint
ls -lh verl_checkpoints/k8s-rag-ppo-qwen2.5-7b-llm-judge/actor/
```

## 📊 预期性能

### 训练时间
- 每step: ~30-60秒 (含LLM judge)
- 总计 (300 steps): ~2.5-5小时
- Checkpoint频率: 每50 steps

### 指标预期
- 初始reward: ~0.3-0.4
- 目标reward: ~0.7-0.8
- KL散度: <0.05

## 🔧 配置调整

### 修改batch size (显存不足时)
编辑 `project/train_k8s_rag_ppo.sh`:
```bash
data.train_batch_size=128  # 降低至128
data.val_batch_size=64     # 降低至64
```

### 修改学习率
```bash
actor_rollout_ref.actor.optim.lr=1e-6  # 调整actor学习率
critic.optim.lr=1e-5                    # 调整critic学习率
```

### 修改检索参数
```bash
retriever.topk=5           # 增加检索文档数
```

## 📝 关键技术点

1. **target vs target_answer**:
   - `target` (key_points): 保留用于参考
   - `target_answer`: LLM-as-Judge用于评分
   - Reward function从`extra_info.target_answer`获取参考答案

2. **LLM Judge调用**:
   - 异步批量调用GPT-5 API
   - 温度0.3 (一致性评分)
   - 重试机制: 3次,指数退避
   - Fallback: API失败时返回0.5中性分

3. **Reward计算**:
   - Base score: LLM judge评分 (0-1)
   - Penalty: 答案后有多余输出 -20%
   - Format score: 无答案格式返回0

## 🎯 下一步工作

1. **启动训练**: 运行 `bash project/train_k8s_rag_ppo.sh`
2. **监控指标**: 观察reward曲线和KL散度
3. **中期评估**: 在step 150左右评估模型性能
4. **超参调优**: 根据训练曲线调整学习率和batch size
5. **扩展数据**: 如效果好,扩展到5K-10K QA对

## 📚 参考资料

- Search-R1框架: `/ssd1/zz/AI_efficency/RAG/Search-R1/`
- K8s文档: `/ssd1/zz/AI_efficency/RAG/data/concepts/`
- 原始训练脚本: `train_ppo.sh`

## ✅ 完成时间

- Phase 1: 2026-03-06
- Phase 2: 2026-03-06
- 总耗时: ~1天

---

**项目就绪,可以开始PPO训练!** 🚀
