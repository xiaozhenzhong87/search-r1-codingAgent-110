# Resume 失败原因和解决方案

钟老师您好,我找到问题了!

---

## 🔍 问题根源

### Resume代码的期望路径

查看 `verl/trainer/ppo/ray_trainer.py` 第414-438行:

```python
def _setup_resume_from_checkpoint(self):
    resume_dir = self.config.trainer.resume_from_checkpoint
    
    # 1. 期望 resume_dir 是实验的根目录
    # 2. 然后在下面找 actor/ 子目录
    actor_base_dir = os.path.join(resume_dir, 'actor')
    
    # 3. 再在 actor/ 下找 global_step_XXX/
    step_dirs = [d for d in os.listdir(actor_base_dir) 
                 if d.startswith('global_step_')]
    
    # 4. 选择最新的step
    latest_step = max(int(d.split('_')[-1]) for d in step_dirs)
    latest_actor_path = os.path.join(actor_base_dir, f'global_step_{latest_step}')
```

### 您当前的设置

```bash
# train_ppo_resume_step200_improved.sh 第27行
export RESUME_FROM_CHECKPOINT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200
```

### 实际的目录结构

```
verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/
├── actor/
│   ├── global_step_50/
│   ├── global_step_100/
│   ├── global_step_150/
│   ├── global_step_200/    ← 这里有模型文件
│   ├── global_step_250/
│   └── ...
└── critic/
    ├── global_step_50/
    └── ...
```

### 为什么失败?

您设置的路径是:
```
verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200
```

代码期望的操作:
1. `resume_dir` = `verl_checkpoints/.../actor/global_step_200`
2. 然后找 `resume_dir/actor/` ← 这里失败了!
3. 实际路径变成: `verl_checkpoints/.../actor/global_step_200/actor/` (不存在!)

**所以出现警告**:
```
[Resume] WARNING: no actor checkpoint dir found at 
verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200/actor
```

---

## ✅ 解决方案

### 方案1: 修改脚本路径 (推荐)

**正确路径应该是实验根目录**:

```bash
# 错误 ❌
export RESUME_FROM_CHECKPOINT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200

# 正确 ✅
export RESUME_FROM_CHECKPOINT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl
```

代码会自动:
1. 找到 `actor/` 目录
2. 扫描所有 `global_step_XXX/`
3. 选择最新的 (即 global_step_500)

### 方案2: 指定特定step恢复

如果您想从特定step (如200) 恢复,需要:

#### 选项A: 临时创建软链接

```bash
# 创建一个临时目录结构
mkdir -p verl_checkpoints/resume_from_step_200/actor
mkdir -p verl_checkpoints/resume_from_step_200/critic

# 创建软链接指向step_200
ln -s $(pwd)/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200 \
      verl_checkpoints/resume_from_step_200/actor/global_step_200

ln -s $(pwd)/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/critic/global_step_200 \
      verl_checkpoints/resume_from_step_200/critic/global_step_200

# 然后设置
export RESUME_FROM_CHECKPOINT=verl_checkpoints/resume_from_step_200
```

#### 选项B: 修改verl代码支持直接指定step路径

(不推荐,需要改源码)

---

## 🛠️ 快速修复

### 修复您的训练脚本

我帮您创建一个修复版本:

```bash
#!/usr/bin/env bash
# train_ppo_resume_step200_improved_fixed.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='data/nq_hotpot_search'
export NQ_TEST='data/nq_search/test.parquet'
export HOTPOT_TEST='data/hotpot_search/test.parquet'

WAND_PROJECT='Search-R1'
export BASE_MODEL='/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct'

# ═══════════════════════════════════════════════════════════════
# 方式1: 从最新checkpoint (step 500) 恢复 - 推荐
# ═══════════════════════════════════════════════════════════════
export EXPERIMENT_NAME=nq-hp-ppo-step500-improved-reward
export RESUME_FROM_CHECKPOINT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl
# 代码会自动找到最新的 global_step_500

# ═══════════════════════════════════════════════════════════════
# 方式2: 从特定step (200) 恢复 - 需要创建临时目录
# ═══════════════════════════════════════════════════════════════
# export EXPERIMENT_NAME=nq-hp-ppo-step200-improved-reward-v2
# 
# # 创建临时目录结构
# TEMP_RESUME_DIR=verl_checkpoints/temp_resume_step_200
# mkdir -p $TEMP_RESUME_DIR/actor
# mkdir -p $TEMP_RESUME_DIR/critic
# 
# # 创建软链接
# ln -sf $(pwd)/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200 \
#        $TEMP_RESUME_DIR/actor/global_step_200
# ln -sf $(pwd)/verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/critic/global_step_200 \
#        $TEMP_RESUME_DIR/critic/global_step_200
# 
# export RESUME_FROM_CHECKPOINT=$TEMP_RESUME_DIR

... (其他配置保持不变)
```

---

## 📋 对比三种情况

| 设置 | Resume路径 | 结果 | 实际step |
|------|-----------|------|---------|
| **当前(错误)** | `.../actor/global_step_200` | ❌ 失败,从头训练 | 0→150 |
| **修复后(方式1)** | `.../nq-ppo-sft-noformat-v2-lowvarkl` | ✅ 从step 500恢复 | 500→650 |
| **修复后(方式2)** | `.../temp_resume_step_200` | ✅ 从step 200恢复 | 200→350 |

---

## 🎯 推荐方案

### 建议: 使用方式1 - 从Step 500恢复

**理由**:
1. ✅ **最简单**: 只需改一行路径
2. ✅ **最新状态**: Step 500是最好的checkpoint
3. ✅ **避免问题**: Step 200性能已下降(0.311)

**Step 500的优势**:
- 经过了完整的500步训练
- 虽然原训练不稳定,但可能有可用的知识
- 用新reward继续训练,可能快速改善

**如果不行**:
- 当前训练(从头开始)到Step 150已达0.416
- 已经超过原训练最好成绩
- 建议继续当前训练就好

---

## 🚀 立即修复

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 修改脚本第27行
# 从: export RESUME_FROM_CHECKPOINT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl/actor/global_step_200
# 改为: export RESUME_FROM_CHECKPOINT=verl_checkpoints/nq-ppo-sft-noformat-v2-lowvarkl

# 修改实验名避免冲突
# 从: export EXPERIMENT_NAME=nq-hp-ppo-step200-improved-reward
# 改为: export EXPERIMENT_NAME=nq-hp-ppo-step500-improved-reward

# 重新启动训练
bash train_ppo_resume_step200_improved.sh
```

---

**总结**: 问题是路径多了 `/actor/global_step_200`,应该指向实验根目录!
