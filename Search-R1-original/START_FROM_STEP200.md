# 从 Step 200 Resume 训练 - 后台执行指令

钟老师您好!所有准备就绪,以下是后台执行指令:

---

## 🚀 后台执行指令

### 方式1: 使用 nohup (推荐)

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 后台启动训练
nohup bash train_ppo_from_step200_improved.sh > nohup_step200_improved.log 2>&1 &

# 保存进程ID
echo $! > train_step200_improved.pid

# 查看PID
cat train_step200_improved.pid
```

### 方式2: 使用 screen

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 创建新screen会话
screen -S ppo_step200

# 启动训练
bash train_ppo_from_step200_improved.sh

# 按 Ctrl+A, 然后按 D 退出screen (训练继续运行)

# 恢复screen会话
screen -r ppo_step200
```

### 方式3: 使用 tmux

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 创建新tmux会话
tmux new -s ppo_step200

# 启动训练
bash train_ppo_from_step200_improved.sh

# 按 Ctrl+B, 然后按 D 退出tmux (训练继续运行)

# 恢复tmux会话
tmux attach -t ppo_step200
```

---

## 📊 训练监控

### 查看实时日志

```bash
# 查看主日志
tail -f logs/improved_reward/nq-hp-ppo-step200-improved-reward-v2.log

# 如果用nohup
tail -f nohup_step200_improved.log
```

### 查看训练进程

```bash
# 查看进程状态
ps aux | grep main_ppo

# 或者用PID
ps -p $(cat train_step200_improved.pid)
```

### 查看GPU使用

```bash
watch -n 1 nvidia-smi
```

---

## ✅ 启动验证

启动后,应该看到:

```
======================================================================
  准备从 Step 200 恢复训练
======================================================================

清理旧的临时目录...
创建resume目录结构...
创建软链接到 Step 200 checkpoints...
✓ Resume目录结构创建成功
  Actor: verl_checkpoints/temp_resume_from_step_200/actor/global_step_200
  Critic: verl_checkpoints/temp_resume_from_step_200/critic/global_step_200

======================================================================
  从 Step 200 继续训练 - 使用改进的 Reward Function
======================================================================

基础配置:
  模型: /ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct
  实验名: nq-hp-ppo-step200-improved-reward-v2
  恢复检查点: verl_checkpoints/temp_resume_from_step_200
  源检查点: verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200
  数据目录: data/nq_hotpot_search

Reward 改进:
  - 答案正确 + 无多余输出: 1.0 ✅
  - 答案正确 + 有多余输出: 0.8 ⚠️ (惩罚 -0.2)
  - 答案错误: 0.0 ❌

训练参数:
  起始步数: 200
  目标步数: 700 (额外训练500步)
  保存频率: 每 100 步
  测试频率: 每 50 步
```

然后在日志中应该看到:

```
[Resume] Found checkpoint at step 200
[Resume] Actor weights: verl_checkpoints/temp_resume_from_step_200/actor/global_step_200
[Resume] Critic weights: verl_checkpoints/temp_resume_from_step_200/critic/global_step_200
```

---

## 📁 输出位置

```bash
# 检查点
verl_checkpoints/nq-hp-ppo-step200-improved-reward-v2/
  ├── actor/
  │   ├── global_step_250/
  │   ├── global_step_300/
  │   ├── global_step_350/
  │   └── ...
  └── critic/
      └── ...

# 日志
logs/improved_reward/nq-hp-ppo-step200-improved-reward-v2.log

# 临时resume目录 (训练完成后可删除)
verl_checkpoints/temp_resume_from_step_200/
```

---

## 🛑 停止训练

### 优雅停止 (推荐)

```bash
# 方式1: 如果用nohup
kill $(cat train_step200_improved.pid)

# 方式2: 找到进程并停止
ps aux | grep main_ppo
kill <PID>
```

### 强制停止 (不推荐)

```bash
kill -9 $(cat train_step200_improved.pid)
```

---

## 📋 关键监控指标

### 在日志中查找

```bash
# 查看每64次采样打印的信息
grep "Has continuation after answer" logs/improved_reward/nq-hp-ppo-step200-improved-reward-v2.log

# 查看测试分数
grep "val/test_score/nq" logs/improved_reward/nq-hp-ppo-step200-improved-reward-v2.log | tail -10

# 查看step信息
grep "step:" logs/improved_reward/nq-hp-ppo-step200-improved-reward-v2.log | tail -5
```

### 预期趋势

| Step | val/test_score/nq | Has continuation: True | 说明 |
|------|------------------|----------------------|------|
| 200 (初始) | ~0.31 | ~90% | 原训练的状态 |
| 250 | ~0.35-0.40 | ~70% | 开始改善 |
| 300 | ~0.40-0.42 | ~50% | 持续改善 |
| 400 | ~0.42-0.45 | ~30% | 接近最优 |
| 500+ | ~0.45+ | <20% ✅ | 目标达成 |

---

## 🔧 常见问题

### 问题1: Resume时报错 "no actor checkpoint"

**解决**: 脚本会自动创建正确的目录结构,如果还是失败:

```bash
# 手动检查
ls -la verl_checkpoints/temp_resume_from_step_200/actor/global_step_200
ls -la verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200
```

### 问题2: GPU内存不足

**解决**: 检查是否有其他训练在运行

```bash
nvidia-smi
# 如果有其他进程,停止它们或调整gpu_memory_utilization
```

### 问题3: 检索服务未启动

**解决**: 确保检索服务在运行

```bash
curl http://127.0.0.1:8000/retrieve
```

---

## 🎯 训练完成后

### 清理临时目录

```bash
# 训练完成后可以删除临时resume目录
rm -rf verl_checkpoints/temp_resume_from_step_200
```

### 对比测试

```bash
cd rejection-sampling

# 对比 Step 200 (原) vs Step 300 (新)
./test_step_comparison.sh \
    --checkpoint1 verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_300 \
    --checkpoint2 verl_checkpoints/nq-hp-ppo-step200-improved-reward-v2/actor/global_step_300
```

---

## 📞 文件清单

- **训练脚本**: `train_ppo_from_step200_improved.sh`
- **启动说明**: `START_FROM_STEP200.md` (本文件)
- **详细分析**: `RESUME_PATH_FIX.md`
- **Reward改进**: `REWARD_IMPROVEMENT_FOR_QA_EM.md`

---

**现在可以执行了!** 🚀

推荐使用方式1 (nohup),这样可以关闭SSH也能继续训练。
