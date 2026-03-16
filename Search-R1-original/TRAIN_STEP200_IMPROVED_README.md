# 从 Step 200 开始使用改进 Reward 训练

钟老师您好!所有准备工作已完成,可以开始训练了!

---

## ✅ 已完成的操作

### 1. Reward Function 替换

```bash
✓ 备份: verl/utils/reward_score/qa_em.py.original (4597 bytes)
✓ 新版: verl/utils/reward_score/qa_em.py (6871 bytes) - 来自 qa_em_improved.py
```

**改进内容**:
- 添加了 `has_continuation_after_answer()` 检测函数
- 修改了 `compute_score_em()` 和 `compute_score_subem()`
- 答案正确但有多余输出: 从 1.0 → 0.8 (惩罚 -0.2)

### 2. 创建训练脚本

新脚本: `train_ppo_resume_step200_improved.sh`

**关键配置**:
- 恢复检查点: `verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200`
- 实验名: `nq-hp-ppo-step200-improved-reward`
- 输出目录: `verl_checkpoints/nq-hp-ppo-step200-improved-reward/`
- 日志文件: `logs/improved_reward/nq-hp-ppo-step200-improved-reward.log`

---

## 🚀 开始训练

### 方式1: 直接运行 (前台)

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1
bash train_ppo_resume_step200_improved.sh
```

### 方式2: 后台运行 (推荐)

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1
nohup bash train_ppo_resume_step200_improved.sh > nohup_improved_reward.log 2>&1 &
echo $! > train_improved_reward.pid
echo "训练已启动, PID: $(cat train_improved_reward.pid)"
```

### 方式3: 使用 screen/tmux

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1
screen -S ppo_improved
bash train_ppo_resume_step200_improved.sh
# 按 Ctrl+A, D 退出screen
# 恢复: screen -r ppo_improved
```

---

## 📊 监控训练

### 1. 查看实时日志

```bash
# 查看训练日志
tail -f logs/improved_reward/nq-hp-ppo-step200-improved-reward.log

# 查看后台日志 (如果用方式2)
tail -f nohup_improved_reward.log
```

### 2. 关键监控指标

#### 在日志中查找 (每64次采样打印一次):

```
--------------------------------
Golden answers: ['xxx']
Extracted answer: xxx
Has continuation after answer: True/False  ← 关键指标!
```

#### 预期趋势:

| 训练步数 | Has continuation: True 比例 | 说明 |
|---------|--------------------------|------|
| 200-250 | ~70-80% | 初期,还在学习 |
| 250-350 | ~40-50% | 中期,开始改善 |
| 350-500 | <20% ✅ | 后期,接近目标 |

### 3. WandB 监控

```
项目: Search-R1
实验: nq-hp-ppo-step200-improved-reward
```

关注的指标:
- `reward/mean`: 平均reward (应该逐渐提升)
- `val/test_score/nq`: NQ测试准确率
- 自定义统计 (如果添加): `continuation_ratio`

---

## 📁 训练输出

### 检查点保存

```bash
# 查看保存的检查点
ls -lh verl_checkpoints/nq-hp-ppo-step200-improved-reward/

# 应该看到:
# actor/global_step_300/
# actor/global_step_400/
# actor/global_step_500/
# critic/global_step_300/
# ...
```

### 日志文件

```bash
# 主日志
logs/improved_reward/nq-hp-ppo-step200-improved-reward.log

# 如果用nohup
nohup_improved_reward.log
```

---

## 🔍 训练中检查

### 1. 检查GPU使用

```bash
watch -n 1 nvidia-smi
```

### 2. 检查进程

```bash
# 查看训练进程
ps aux | grep main_ppo

# 如果用方式2
cat train_improved_reward.pid
ps -p $(cat train_improved_reward.pid)
```

### 3. 快速验证reward改进

训练100步后,可以用测试脚本验证:

```bash
# 停止训练,运行测试
cd rejection-sampling
./test_step_comparison.sh \
    --step1 300 \
    --step2 200
```

---

## ⚙️ 训练参数说明

### 从原脚本继承的参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `total_training_steps` | 500 | 总训练步数 |
| `save_freq` | 100 | 每100步保存 |
| `test_freq` | 50 | 每50步测试 |
| `train_batch_size` | 512 | 训练批大小 |
| `actor.lr` | 1e-6 | Actor学习率 |
| `critic.lr` | 1e-5 | Critic学习率 |
| `kl_coef` | 0.001 | KL散度系数 |
| `max_turns` | 4 | 最大搜索轮数 |

### Reward 相关参数

在 `qa_em.py` 中,可以调整:

```python
# penalty_ratio: 惩罚比例
# 默认 0.2 (答案对但有多余输出 → 0.8分)

# 如需调整,修改函数调用:
compute_score_em(..., penalty_ratio=0.2)  # 当前
compute_score_em(..., penalty_ratio=0.3)  # 更严格
compute_score_em(..., penalty_ratio=0.1)  # 更温和
```

---

## 🛠️ 问题排查

### 问题1: 训练启动失败

```bash
# 检查检查点是否存在
ls -lh verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200/

# 检查数据文件
ls -lh data/nq_hotpot_search/train.parquet
ls -lh data/nq_search/test.parquet
```

### 问题2: GPU内存不足

编辑脚本,降低:
- `gpu_memory_utilization=0.6` → `0.5`
- `train_batch_size=512` → `256`

### 问题3: 检索服务未启动

```bash
# 检查检索服务
curl http://127.0.0.1:8000/retrieve

# 如果失败,启动检索服务
# (根据您的检索服务启动脚本)
```

### 问题4: reward没有改进效果

查看日志,如果 `Has continuation: True` 比例没下降:
- 检查 `qa_em.py` 是否正确替换
- 尝试增加 `penalty_ratio` (如改为0.3)

---

## 📋 训练后评估

### 1. 对比 Step 300 (旧) vs Step 300 (新)

```bash
cd rejection-sampling

# 测试新训练的 step 300
./test_step_comparison.sh \
    --checkpoint1 verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_300 \
    --checkpoint2 verl_checkpoints/nq-hp-ppo-step200-improved-reward/actor/global_step_300
```

### 2. 查看 continuation 统计

```bash
# 分析新模型的输出
python << 'EOF'
import json
import re

file = "output/step_XXX/nq_results.jsonl"
cont_count = 0
total = 0

with open(file) as f:
    for line in f:
        data = json.loads(line)
        resp = data['response']
        matches = list(re.finditer(r'</answer>', resp))
        if matches:
            after = resp[matches[-1].end():].strip()
            if re.search(r'<(search|think|information)>', after):
                cont_count += 1
        total += 1

print(f"Continuation ratio: {cont_count}/{total} ({cont_count/total*100:.1f}%)")
EOF
```

预期: 新训练的模型应该 <20%,而旧的 step 300 是 90%

---

## 🎯 成功标准

训练成功的标志:

1. ✅ **Continuation比例下降**: 从初期70-80% → 后期<20%
2. ✅ **准确率保持或提升**: val/test_score/nq ≥ 48% (不低于原step 200)
3. ✅ **Reward合理分布**: 看到0.8和1.0两个峰值
4. ✅ **训练稳定**: 没有reward突然崩溃

---

## 🔄 如需回退

如果需要恢复原reward function:

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1
cp verl/utils/reward_score/qa_em.py.original verl/utils/reward_score/qa_em.py
echo "✓ 已恢复原版 reward function"
```

---

## 📞 联系信息

- 训练脚本: `train_ppo_resume_step200_improved.sh`
- 原reward备份: `verl/utils/reward_score/qa_em.py.original`
- 改进reward: `verl/utils/reward_score/qa_em_improved.py`
- 说明文档: `TRAIN_STEP200_IMPROVED_README.md` (本文件)

---

**一切准备就绪,可以开始训练了!** 🚀

建议使用方式2 (nohup后台运行),这样可以断开SSH连接后继续训练。

祝训练顺利! 🎉
