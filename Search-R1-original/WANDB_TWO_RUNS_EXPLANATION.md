# WandB 出现两个Run的原因说明

钟老师您好!

## 🔍 问题分析

您的 `wandb/` 目录下出现了两个run:

```
run-20260306_013721-orxnpk0f  (最新, PID 914595)
run-20260306_013716-geqfjy7s  (稍早, PID 872799)
```

## ✅ 原因解释

### 第一个run: `geqfjy7s` (01:37:16启动)

从训练日志看到:

```
wandb: setting up run geqfjy7s
wandb: Syncing run nq-ppo-sft-noformat-v2-lowvarkl-from-step200
wandb: 🚀 View run at https://wandb.ai/.../runs/geqfjy7s
```

**这是第一次启动训练时创建的run**,但可能因为某些原因(比如训练脚本重启)被中断了。

### 第二个run: `orxnpk0f` (01:37:21启动,当前活跃)

从最新的nohup日志看到:

```
wandb: setting up run orxnpk0f
```

**这是当前正在运行的训练run**。

---

## 🎯 为什么会出现两个?

### 情况1: 训练脚本被快速重启

最可能的原因:
1. **01:37:16**: 第一次启动,创建了run `geqfjy7s`
2. **几秒后**: 训练因某原因中断(可能是手动停止,或自动重启)
3. **01:37:21**: 训练再次启动,创建了新run `orxnpk0f`

### 情况2: Ray的分布式进程创建了多个WandB实例

在分布式训练中,如果多个进程都尝试初始化WandB,可能会创建多个run目录(但通常只有一个是主run)。

---

## 🔍 验证哪个是活跃的

```bash
# 查看最新run的最后修改时间
ls -lht /ssd1/zz/AI_efficency/RAG/Search-R1/wandb/run-*/run-*.wandb

# 查看进程对应的run
ps aux | grep "914595\|872799"
```

从您的进程列表中看到:
- **PID 914595**: `ray::main_task` - **正在运行**
- **PID 872799**: 已经不存在了(第一次启动的进程)

所以:
- ✅ **`orxnpk0f`** 是**当前活跃的run**
- ❌ **`geqfjy7s`** 是**已中断的旧run** (可以删除)

---

## ✅ 这样没问题吗?

### 完全没问题! ✅

1. **当前训练正常运行**: run `orxnpk0f` 正在记录数据
2. **旧run不影响**: `geqfjy7s` 只是一个未完成的run,不会影响当前训练

---

## 🧹 清理建议 (可选)

如果您想保持目录整洁,可以删除旧的未完成run:

```bash
cd /ssd1/zz/AI_efficency/RAG/Search-R1

# 删除旧的run (geqfjy7s)
rm -rf wandb/run-20260306_013716-geqfjy7s

# 或者移动到备份目录
mkdir -p wandb_backup
mv wandb/run-20260306_013716-geqfjy7s wandb_backup/
```

**注意**: **不要删除** `run-20260306_013721-orxnpk0f`,这是当前正在使用的!

---

## 📊 关于新run的状态

### 重要发现!

看训练日志,**新run `orxnpk0f` 没有resume到旧run** `b6q2bvxz`!

```
# 旧训练(01:26启动)
wandb: Resuming run nq-ppo-sft-noformat-v2-lowvarkl  ← resume到旧run
wandb: run ID: b6q2bvxz

# 新训练(01:37启动)  
wandb: setting up run orxnpk0f  ← 创建了新run!
```

**这意味着您之前的重启操作成功了!** 🎉

现在的训练:
- ✅ 使用新的run ID: `orxnpk0f`
- ✅ 实验名: `nq-ppo-sft-noformat-v2-lowvarkl-from-step200`
- ✅ 是独立的新曲线!

---

## 🎯 总结

### 出现两个run的原因
训练脚本在短时间内重启了一次(可能是您执行了重启脚本):
- 第一次启动(01:37:16)创建了 `geqfjy7s`
- 第二次启动(01:37:21)创建了 `orxnpk0f` (当前活跃)

### 当前状态
- ✅ 当前训练正常运行
- ✅ 使用新run ID `orxnpk0f`
- ✅ **这是一条独立的新曲线**,不会追加到旧run上!
- ✅ 旧run `geqfjy7s` 可以安全删除

### 验证方法

等训练完成后,在WandB上应该能看到:

| Run名称 | Run ID | Steps | 说明 |
|---------|--------|-------|------|
| `nq-ppo-sft-noformat-v2-lowvarkl` | `b6q2bvxz` | 0-500+ | 原训练 |
| `nq-ppo-sft-noformat-v2-lowvarkl-from-step200` | `orxnpk0f` | 201-700 | **新训练**(独立曲线) |

**恭喜!您现在已经有独立的新曲线了!** 🎉
