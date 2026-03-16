# Rollout Batch配置详解

钟老师您好!查看了代码的默认配置,现在给您完整说明:

---

## 📋 默认配置文件

**位置**: `verl/trainer/config/ppo_trainer.yaml`

### 关键配置项

```yaml
# 第90行 - Rollout配置
rollout:
  n: 1              # 每个prompt生成的response数量
  n_agent: 1        # agent数量

# 第12行 - 训练batch配置  
data:
  train_batch_size: 1024    # 默认值
```

---

## ✅ 您当前的实际配置

从 `train_ppo_resume_step200_improved.sh` 看到:

```bash
# 第81行
data.train_batch_size=512

# 第105行
actor_rollout_ref.rollout.n_agent=1
```

**注意**: 脚本中**没有显式设置 `rollout.n`**,所以使用默认值 `n=1`

---

## 🎯 正确理解

### 1. **Rollout Batch = train_batch_size = 512**

这是**每个训练step rollout阶段生成的prompt数量**:
- 从训练数据采样 **512个问题(prompts)**
- 每个问题生成 **1个response** (因为 `n=1`)
- 总共生成 **512条完整轨迹**

### 2. **每条轨迹是多轮对话**

因为设置了 `max_turns=4`:
```
一条轨迹 = 1个问题 + 最多4轮(think→search→information)交互 + 1个answer
```

从日志看到:
```
ACTIVE_TRAJ_NUM: [512, 512, 502, 113, 28, 9]
                  ^    ^    ^    ^    ^   ^
                  |    |    |    |    |   第6轮还剩9条在继续
                  |    |    |    |    第5轮剩28条
                  |    |    |    第4轮剩113条  
                  |    |    第3轮剩502条
                  |    第2轮512条都还在
                  第1轮开始512条轨迹
```

### 3. **训练样本数 = 512**

因为 `n=1`,所以:
- **生成**: 512个prompts × 1 response/prompt = **512个样本**
- **训练**: 用这512个样本训练

---

## 🆚 如果是GRPO (n > 1)

假设改成 `n=5`:

```yaml
rollout:
  n: 5              # 每个prompt生成5个responses
  
actor:
  use_kl_loss: True  # GRPO需要开启
```

那么:
- **Rollout**: 512个prompts × 5 responses = **2560个样本**
- **训练**: 用这2560个样本,通过group-wise对比学习

---

## 📊 您的完整配置总结

```
┌────────────────────────────────────────┐
│ Rollout阶段                             │
├────────────────────────────────────────┤
│ train_batch_size = 512                 │
│ n (每prompt的response数) = 1 (默认)    │
│ n_agent = 1                             │
│ max_turns = 4                           │
├────────────────────────────────────────┤
│ → 每step生成512条轨迹                   │
│ → 每条轨迹最多4轮交互                   │
└────────────────────────────────────────┘
           ↓
┌────────────────────────────────────────┐
│ 训练阶段 (PPO)                          │
├────────────────────────────────────────┤
│ ppo_mini_batch_size = 256              │
│ ppo_micro_batch_size = 64 (Actor)      │
│ ppo_micro_batch_size = 8 (Critic)      │
├────────────────────────────────────────┤
│ → 512条轨迹分成2个mini-batch            │
│ → 每个mini-batch分成4个micro-batch      │
│   送入GPU训练                           │
└────────────────────────────────────────┘
```

---

## 💡 术语澄清

| 术语 | 含义 | 您的配置 |
|------|------|----------|
| **Rollout Batch** | 每step rollout生成的**prompt数量** | **512** |
| **n** | 每个prompt生成的**response数量** | **1** (PPO模式) |
| **训练样本数** | rollout_batch × n | **512 × 1 = 512** |
| **轨迹数** | 完整对话的数量 | **512条** |
| **max_turns** | 每条轨迹的最大交互轮数 | **4轮** |

---

## ✅ 最终答案

**您的Rollout Batch Size = 512**

含义:
- ✅ 每个训练step从数据中采样**512个问题**
- ✅ 每个问题生成**1个response**(**标准PPO**,不是GRPO)
- ✅ 总共生成**512条多轮交互轨迹**用于训练
- ✅ 每条轨迹包含最多**4轮**think→search→information的交互过程

---

**感谢您的纠正!** 现在解释应该准确了 🎯
