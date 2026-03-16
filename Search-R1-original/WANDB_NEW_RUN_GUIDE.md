# WandB 新曲线设置方案

钟老师您好!有几种方式可以在WandB中建立新曲线,从Step 201开始:

---

## 方案1: 修改实验名 (最简单,推荐)

### 原理
使用完全不同的实验名,WandB会创建新的run,从头开始记录。

### 操作步骤

**停止当前训练**:
```bash
# 如果有训练在运行
kill $(cat train_step200_improved.pid)
```

**修改实验名**:
```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1
```

编辑 `train_ppo_from_step200_improved.sh` 第20行:

**从**:
```bash
export EXPERIMENT_NAME=nq-hp-ppo-step200-improved-reward-v2
```

**改为**:
```bash
export EXPERIMENT_NAME=nq-hp-ppo-step200-improved-reward-from201
```

**重新启动**:
```bash
nohup bash train_ppo_from_step200_improved.sh > nohup_step200_v3.log 2>&1 &
echo $! > train_step200_v3.pid
```

### 效果

✅ WandB会创建新的run: `nq-hp-ppo-step200-improved-reward-from201`
✅ 从Step 201开始记录,不会有警告
✅ 曲线从201开始显示

---

## 方案2: 使用 WandB 的 step offset

### 原理
让WandB记录的step从自定义值开始

### 操作步骤

**修改训练脚本**,在Python训练代码中添加step offset。

需要修改verl的trainer代码:

```python
# 在 verl/trainer/ppo/ray_trainer.py 中
# 找到 wandb.log() 的地方

# 添加 step offset
WANDB_STEP_OFFSET = 0  # 如果从201开始,设为0;如果想显示为201,设为201

wandb.log({
    'metric': value,
    ...
}, step=current_step + WANDB_STEP_OFFSET)
```

但这需要修改verl源码,**不推荐**。

---

## 方案3: 创建新的 WandB run ID

### 原理
强制WandB创建新的run,不resume之前的

### 方法A: 删除本地WandB缓存

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 删除wandb缓存
rm -rf wandb/run-*
```

然后使用**新的实验名**重新启动训练。

### 方法B: 在训练配置中禁用resume

修改训练脚本,添加环境变量:

```bash
export WANDB_RESUME="never"  # 强制不resume
export WANDB_RUN_ID=""        # 清空run ID
```

---

## 方案4: 手动创建新run (通过脚本修改)

### 创建修改版训练脚本

```bash
#!/usr/bin/env bash
# train_ppo_from_step200_improved_newrun.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ... (其他配置保持不变) ...

# 关键修改: 新实验名 + 强制新run
export EXPERIMENT_NAME=nq-hp-ppo-from-step200-newrun
export WANDB_RESUME="never"
export WANDB_RUN_ID=""

# ... (其他配置保持不变) ...
```

---

## 🎯 推荐方案

### 方案1 + 修改实验名 (最简单有效)

**优点**:
- ✅ 最简单,只改一行
- ✅ 完全隔离,不影响原有数据
- ✅ WandB自动创建新曲线
- ✅ 从Step 201开始记录

**缺点**:
- 实验名变了(但这也是优点,更清晰)

### 具体操作

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 1. 停止当前训练 (如果在运行)
kill $(cat train_step200_improved.pid) 2>/dev/null || true

# 2. 修改实验名
sed -i 's/nq-hp-ppo-step200-improved-reward-v2/nq-hp-ppo-from-step200-newrun/g' train_ppo_from_step200_improved.sh

# 3. 重新启动
nohup bash train_ppo_from_step200_improved.sh > nohup_from_step200_newrun.log 2>&1 &
echo $! > train_from_step200_newrun.pid

# 4. 监控日志
tail -f nohup_from_step200_newrun.log
```

---

## 📊 WandB 显示效果

### 原来的问题

```
Step 0 ──> Step 500 (原训练)
             ↓
           Step 201 (新训练,被WandB忽略)
```

### 方案1效果 (新实验名)

**Run 1**: `nq-ppo-sft-noformat-v2-lowvarkl` (原训练)
```
Step 0 ──> Step 500
```

**Run 2**: `nq-hp-ppo-from-step200-newrun` (新训练)
```
Step 201 ──> Step 700  (正常记录)
```

### 对比显示

在WandB界面:
1. 选择两个run对比
2. 可以看到:
   - Run 1: 0-500的完整曲线
   - Run 2: 201-700的新曲线 (使用改进的reward)

---

## 🔧 如果想让Step显示从0开始

如果您希望新训练的Step **显示为0-500**(实际是201-700),可以:

### 选项A: 修改verl代码

修改 `verl/trainer/ppo/ray_trainer.py`:

```python
# 找到 resume_step 的地方
self.resume_step = self._setup_resume_from_checkpoint()

# 添加: 重置wandb的step计数
if self.resume_step > 0 and 'wandb' in self.config.trainer.logger:
    # 让wandb从0开始计数,但训练从resume_step继续
    self.wandb_step_offset = -self.resume_step
```

然后在log的时候:
```python
wandb.log({...}, step=self.global_steps + self.wandb_step_offset)
```

### 选项B: 后处理WandB数据

训练完成后,在WandB界面使用"custom charts"功能,手动调整x轴。

---

## 💡 我的建议

### 立即可用的方案

**使用方案1 - 新实验名**:

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 创建新的训练脚本(自动修改实验名)
cat train_ppo_from_step200_improved.sh | \
  sed 's/nq-hp-ppo-step200-improved-reward-v2/nq-hp-ppo-from-step200-improved/g' \
  > train_ppo_from_step200_newrun.sh

chmod +x train_ppo_from_step200_newrun.sh

# 启动新训练
nohup bash train_ppo_from_step200_newrun.sh > nohup_newrun.log 2>&1 &
echo $! > train_newrun.pid

# 查看日志
tail -f nohup_newrun.log
```

### 为什么推荐这个方案

1. ✅ **最简单**: 只改实验名,1分钟搞定
2. ✅ **最安全**: 不影响原有数据和代码
3. ✅ **最清晰**: 新的run名字明确表示"从step200改进"
4. ✅ **立即生效**: 不需要等待,马上可以看到新曲线

### WandB中的效果

在WandB项目 `Search-R1` 下会看到:

| Run名称 | Steps | 说明 |
|---------|-------|------|
| `nq-ppo-sft-noformat-v2-lowvarkl` | 0-500 | 原训练 |
| `nq-hp-ppo-from-step200-improved` | 201-700 | 新训练(改进reward) |

可以在WandB中对比这两条曲线,清楚地看到改进的效果!

---

**您想用哪个方案?我可以帮您立即实施!** 🚀
