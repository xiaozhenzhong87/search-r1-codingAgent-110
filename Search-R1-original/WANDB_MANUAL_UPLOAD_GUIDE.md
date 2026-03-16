# 训练完成后手动上传WandB的方法

钟老师您好!

## ✅ 可以的!训练完后手动上传WandB

训练时WandB会在本地保存所有数据,即使没有实时同步,训练完成后也可以手动上传。

---

## 方法1: 使用 wandb sync (最简单,推荐)

WandB会将所有run数据保存在本地 `wandb/` 目录下,可以随时同步:

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 查看所有本地run
ls -lht wandb/

# 同步最新的run
wandb sync wandb/run-20260306_012613-b6q2bvxz

# 或者同步所有未上传的run
wandb sync --sync-all
```

### 优点
- ✅ **最简单**: 一条命令搞定
- ✅ **安全**: 不影响正在训练的进程
- ✅ **完整**: 包含所有metrics、logs、system info

---

## 方法2: 离线训练 + 训练后上传

如果担心网络影响训练速度,可以完全离线训练:

### 训练前设置离线模式

```bash
# 在训练脚本开头添加
export WANDB_MODE=offline

# 或者修改train_ppo_from_step200_improved.sh
cd /ssd1/zz/AI_efficency/RAG/Search-R1
```

修改训练脚本第18行附近:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 添加这行 - 离线模式
export WANDB_MODE=offline

export EXPERIMENT_NAME=nq-ppo-sft-noformat-v2-lowvarkl-from-step200
```

### 训练完成后上传

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 上传特定run
wandb sync wandb/run-20260306_XXXXXX-XXXXXXXX

# 或者上传所有离线run
wandb sync --sync-all
```

---

## 方法3: 从日志重新创建WandB run

如果本地WandB数据丢失,可以从训练日志手动提取metrics重新上传:

### 创建上传脚本

```python
# upload_metrics_to_wandb.py
import wandb
import re
import json

# 初始化新的WandB run
wandb.init(
    project="Search-R1",
    name="nq-ppo-sft-noformat-v2-lowvarkl-from-step200-manual",
    config={
        "model": "Qwen2.5-7B-Instruct",
        "kl_type": "low_var_kl",
        "kl_coef": 0.001,
        "resumed_from": "step_200",
        "reward_version": "v2_penalty",
    }
)

# 解析日志文件
log_file = "logs/improved_reward/nq-ppo-sft-noformat-v2-lowvarkl-from-step200.log"

with open(log_file, 'r') as f:
    for line in f:
        # 提取step信息
        step_match = re.search(r'epoch \d+, step (\d+)', line)
        if step_match:
            current_step = int(step_match.group(1))
        
        # 提取metrics (需要根据实际日志格式调整)
        if 'critic/score/mean' in line:
            score_match = re.search(r'critic/score/mean[:\s]+([0-9.]+)', line)
            if score_match:
                wandb.log({"critic/score/mean": float(score_match.group(1))}, step=current_step)
        
        # ... 添加其他metrics的解析 ...

wandb.finish()
```

**不推荐**这个方法,因为太繁琐,且可能丢失信息。

---

## 方法4: 让当前训练继续,不管WandB的事

**最简单的方式**: 让训练继续跑,训练完成后:

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 1. 找到当前run的目录
ls -lht wandb/

# 假设是 run-20260306_012613-b6q2bvxz

# 2. 检查同步状态
wandb sync --view wandb/run-20260306_012613-b6q2bvxz

# 3. 如果显示未同步,手动同步
wandb sync wandb/run-20260306_012613-b6q2bvxz
```

---

## 📊 查看当前WandB状态

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 查看所有本地run
ls -lht wandb/

# 查看最新run的详细信息
ls -la wandb/latest-run/

# 检查哪些run还没上传
wandb sync --sync-all --view
```

---

## 🎯 推荐方案(针对您的情况)

### 选项A: 让训练继续跑,完成后手动上传

**当前训练不用管**,虽然WandB resume到了旧run,但数据还是在记录的。训练完成后:

```bash
# 训练完成后执行
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 同步所有数据
wandb sync --sync-all

# 或者只同步当前run
wandb sync wandb/run-20260306_012613-b6q2bvxz
```

**优点**:
- ✅ 不影响当前训练
- ✅ 所有数据都会被记录
- ✅ 训练完成后随时可以上传

**缺点**:
- ❌ 新数据会被添加到旧run中(step 201+会接在原来的曲线后面)
- ❌ 不是独立的新曲线

---

### 选项B: 停止训练,修复后重启(独立新曲线)

如果您希望在WandB中看到**独立的新曲线**,需要:

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 1. 停止当前训练
pkill -f "train_ppo_from_step200_improved.sh"

# 2. 修复trainer_state.json
bash fix_wandb_resume.sh

# 3. 清理WandB缓存
rm -rf wandb/run-*

# 4. 重新启动
nohup bash train_ppo_from_step200_improved.sh > nohup_newrun.log 2>&1 &
```

**优点**:
- ✅ 独立的新WandB run
- ✅ 清晰的曲线对比

**缺点**:
- ❌ 需要重启训练(会浪费一些时间)

---

## 💡 我的建议

**选项A - 让训练继续**,原因:

1. **训练已经在跑了**: 当前在Step 201,不要中断
2. **数据不会丢**: 虽然resume到旧run,但新数据会正常记录
3. **可以后期处理**: 训练完成后,可以用WandB API分离step 201+的数据创建新曲线

### 训练完成后分离数据的方法

```python
# 训练完成后执行
import wandb

api = wandb.Api()
old_run = api.run("xiaozhenzhong87-uestc/Search-R1/b6q2bvxz")

# 创建新run,只包含step 201+的数据
new_run = wandb.init(
    project="Search-R1",
    name="nq-ppo-from-step200-improved-reward",
    resume=False
)

# 提取step 201+的数据
history = old_run.history()
filtered_data = history[history['_step'] >= 201]

# 重新记录到新run (调整step从0开始)
for idx, row in filtered_data.iterrows():
    metrics = {k: v for k, v in row.items() if not k.startswith('_')}
    new_run.log(metrics, step=row['_step'] - 201)

new_run.finish()
```

---

## ⚡ 快速决策

**如果您只是想记录数据,不在意是否独立曲线**:
```bash
# 什么都不用做,让训练继续跑
# 训练完成后执行: wandb sync --sync-all
```

**如果您一定要独立的新曲线**:
```bash
# 需要重启训练
# 告诉我,我帮您执行停止、修复、重启的完整流程
```

**您更倾向于哪种?** 🤔
