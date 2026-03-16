# GRPO / PPO 训练指标完全参考手册

> 基于 verl 框架的 Search-R1 项目，随时查阅

---

## 一、共有指标详解（GRPO 和 PPO 都有）

### 1. 序列长度负载均衡指标（global_seqlen）

反映**多 GPU 之间序列长度分配的均衡性**，直接影响训练效率。

| 指标 | 含义 | 示例值 |
|------|------|--------|
| `global_seqlen/min` | 所有 GPU 中，分配到的最短总序列长度 | 236399 |
| `global_seqlen/max` | 所有 GPU 中，分配到的最长总序列长度 | 272404 |
| `global_seqlen/minmax_diff` | max - min，差值越大说明负载越不均衡 | 36005 |
| `global_seqlen/balanced_min` | 负载均衡后的最小值 | 253016 |
| `global_seqlen/balanced_max` | 负载均衡后的最大值 | 253016 |
| `global_seqlen/mean` | 所有 GPU 的平均序列总长度 | 253016 |

**看点**：`minmax_diff` 越小越好；`balanced_min ≈ balanced_max` 说明均衡策略生效。

---

### 2. 状态 token 指标（state_tokens）

| 指标 | 含义 | 示例值 |
|------|------|--------|
| `state_tokens/total` | 当前 batch 中所有 state token（搜索返回的信息文本 token）总数 | 414448 |
| `state_tokens/coverage` | state token 占总生成 token 的比例 | 0.255 |

**看点**：`coverage` 反映模型生成内容中**检索信息占比**，过高说明模型过度依赖检索，过低说明检索利用不足。

---

### 3. Actor（策略网络）训练指标

GRPO 和 PPO 的**核心指标**，两者完全相同。

| 指标 | 含义 | 示例值 | 关注要点 |
|------|------|--------|----------|
| `actor/pg_loss` | 策略梯度损失（Policy Gradient Loss） | 0.008 | 优化的主要目标函数，正常应在 0 附近小幅波动 |
| `actor/pg_clipfrac` | 策略梯度裁剪比例（被 clip 的 token 占比） | 0.000 | 过高（>0.2）说明新旧策略差异太大，训练不稳定 |
| `actor/kl_loss` | KL 散度损失（当前策略 vs **参考模型 π_ref**） | 0.001 | 衡量策略漂移程度，采用 low_var_kl：r - log(r) - 1 |
| `actor/kl_coef` | KL 损失的系数（超参） | 0.001 | 固定值，控制 KL 惩罚力度 |
| `actor/ppo_kl` | 每 token 的平均 KL 散度（当前策略 vs **旧策略 π_old**） | 0.000 | 纯监控，反映一个 step 内策略偏移 |
| `actor/entropy_loss` | 策略的熵（输出分布的不确定性） | 0.611 | **越高 = 探索越多**，持续下降说明策略在收敛（或坍塌） |
| `actor/grad_norm` | 梯度范数 | 0.981 | 过大（>10）可能梯度爆炸，过小可能梯度消失 |
| `actor/lr` | 当前学习率 | 0.000xx | 随 warmup/decay 变化 |

**总 loss 公式**：`policy_loss = pg_loss - entropy_coeff × entropy_loss + kl_coef × kl_loss`

**`actor/kl_loss` vs `actor/ppo_kl` 的区别**：

| | `actor/ppo_kl` | `actor/kl_loss` |
|---|---|---|
| 比较对象 | π_new vs π_old（rollout 时的策略） | π_new vs π_ref（冻结的初始模型） |
| 刷新时机 | π_old 每个 step 重新算 | π_ref 永远不变 |
| 作用 | **纯监控**，不参与 loss | **参与 loss 反向传播** |
| 意义 | 一个 step 内策略变了多少 | 整体偏离初始模型多远 |

---

### 4. MFU 指标

| 指标 | 含义 | GRPO 示例 | PPO 示例 |
|------|------|-----------|----------|
| `mfu/actor` | Actor 的 Model FLOPs Utilization | 0.362 | 0.351 |

$$\text{MFU} = \frac{\text{实际每秒浮点运算量}}{\text{GPU 理论峰值浮点运算量}}$$

| MFU 范围 | 评价 |
|----------|------|
| 0.3～0.5 | **正常水平** |
| 0.5～0.6 | 优秀 |
| > 0.6 | 极高，通常只有纯预训练才能达到 |
| < 0.2 | 偏低，有瓶颈 |

---

### 5. Score 与 Reward 指标

| 指标 | 含义 |
|------|------|
| `critic/score/mean,max,min` | reward 函数的**原始输出**（QA Exact Match，0 或 1） |
| `critic/rewards/mean,max,min` | 送入 RL 优势计算的**最终奖励** |

**GRPO 与 PPO 在此处的关键区别见下方第五节。**

---

### 6. Advantage 与 Return 指标

| 指标 | 含义 |
|------|------|
| `critic/advantages/mean,max,min` | 优势函数值 |
| `critic/returns/mean,max,min` | 回报值（未来累计奖励） |

**GRPO**：advantage = (reward - group_mean) / group_std，returns = advantages（无 Critic，两者相同）
**PPO**：advantage = GAE(rewards, values)，returns = rewards 的折扣累加（Critic 用 returns 作学习目标）

---

### 7. 序列长度指标

| 指标 | 含义 |
|------|------|
| `response_length/mean,max,min` | 模型生成的 response token 长度 |
| `response_length/clip_ratio` | response 被截断的比例 |
| `prompt_length/mean,max,min` | 输入 prompt token 长度 |
| `prompt_length/clip_ratio` | prompt 被截断的比例 |

**看点**：`response_length/mean` 增长趋势反映模型是否倾向于生成更长的回答；`clip_ratio` 过高说明 `max_response_length` 设置偏小。

---

### 8. 环境交互指标（env）— Search-R1 特有

| 指标 | 含义 |
|------|------|
| `env/number_of_actions/mean,max,min` | 每条样本的动作次数（think/search/answer） |
| `env/finish_ratio` | 正常结束（输出了 `<answer>`）的样本比例 |
| `env/number_of_valid_action` | 平均有效动作数 |
| `env/ratio_of_valid_action` | 有效动作占总动作的比例 |
| `env/number_of_valid_search` | 平均有效搜索次数 |

**看点**：`finish_ratio` 应接近 1.0；`ratio_of_valid_action` 反映格式合规率。

---

### 9. 验证指标

| 指标 | 含义 |
|------|------|
| `val/test_score/nq` | NQ 测试集上的 QA Exact Match 准确率 |

**最终评价指标**，每隔一定步数评估一次。

---

### 10. 计时指标

| 指标 | 含义 | GRPO 有 | PPO 有 |
|------|------|---------|--------|
| `timing_s/gen` | Rollout 生成耗时 | ✓ | ✓ |
| `timing_s/ref` | 参考模型推理耗时 | ✓ | ✓ |
| `timing_s/adv` | 优势函数计算耗时 | ✓ | ✓ |
| `timing_s/update_actor` | Actor 更新耗时 | ✓ | ✓ |
| `timing_s/values` | Critic 前向计算 V(s) 耗时 | ✗ | ✓ |
| `timing_s/update_critic` | Critic 更新耗时 | ✗ | ✓ |
| `timing_s/testing` | 验证评估耗时 | ✓ | ✓ |
| `timing_s/save_checkpoint` | 保存 checkpoint 耗时 | ✓ | ✓ |
| `timing_s/step` | 整个 step 总耗时 | ✓ | ✓ |
| `timing_per_token_ms/*` | 各阶段每 token 耗时 | ✓ | ✓ |

---

## 二、PPO 独有指标详解

以下指标只在 PPO 中出现，GRPO 没有（因为 GRPO 没有 Critic 网络）。

### 1. Critic 网络训练指标

| 指标 | 含义 | 示例值 | 关注要点 |
|------|------|--------|----------|
| `critic/vf_loss` | 价值函数损失（Critic 学习预测 return） | 0.128 | 应逐步下降，初始会很大（~10） |
| `critic/vf_clipfrac` | Critic 预测值被裁剪的比例 | 0.000 | 过高说明值函数波动大 |
| `critic/vpred_mean` | Critic 预测的平均值 V(s) | 0.534 | 应逐步趋近真实 return |
| `critic/values/mean,max,min` | Critic 对每条样本的价值估计 | 0.404 | 理想情况应接近 score/mean |
| `critic/vf_explained_var` | Critic 解释方差比 | 0.064 | 1.0=完美预测，<0=比随机差 |
| `critic/grad_norm` | Critic 梯度范数 | 28.4 | PPO 中 Critic 梯度通常比 Actor 大 |
| `critic/lr` | Critic 学习率 | — | 与 Actor 可独立设置 |
| `mfu/critic` | Critic 的 GPU 利用率 | 0.177 | 通常低于 Actor |

### 2. PPO 的 KL 控制指标

| 指标 | 含义 | 示例值 |
|------|------|--------|
| `critic/kl` | 从 reward 中扣减的 KL 惩罚均值 | 0.102 |
| `critic/kl_coeff` | KL 惩罚系数 | 0.001 |

**注意**：PPO 中 KL 惩罚是**从 reward 里扣减**的（`reward = score - kl_coeff × KL`），而 GRPO 的 KL 是**作为单独 loss 项加入**的。

---

## 三、GRPO 独有的设计特点

GRPO 没有独有的"指标名"（指标名是 PPO 的子集），但以下**计算逻辑**不同：

| 方面 | GRPO | PPO |
|------|------|-----|
| Advantage 计算 | (reward - group_mean) / group_std | GAE(rewards, values, γ, λ) |
| Returns | = Advantages（无 Critic，直接复制） | 折扣累计回报（Critic 的学习目标） |
| Reward | = Score（KL 不扣减 reward） | = Score - KL 惩罚 |
| n_agent | 通常 ≥ 5（需要组内对比） | 通常 = 1 |

---

## 四、Score 与 Reward 的区别（GRPO vs PPO）

### GRPO 中：reward ≡ score

配置中所有附加奖励为 0，且 KL 不从 reward 扣减：

```
reward = score + 0 (format) + 0 (retrieval) + 0 (structure) = score
```

KL 惩罚单独作为 loss 项：`policy_loss += kl_coef × kl_loss`

### PPO 中：reward ≠ score

PPO 的 reward 包含 KL 惩罚（从 reward 中扣减）：

$$\text{reward}_t = \text{score}_t - \text{kl\_coeff} \times (\log\pi_{old}(a_t|s_t) - \log\pi_{ref}(a_t|s_t))$$

实测对比：

| step | score/mean | rewards/mean | 差值 | 说明 |
|------|-----------|-------------|------|------|
| PPO step 1 | 0.324 | 0.324 | 0 | 初始 KL≈0 |
| PPO step 50 | 0.404 | 0.392 | -0.012 | reward < score |
| PPO step 250 | 0.402 | 0.579 | +0.177 | reward > score（KL 为负） |
| PPO step 450 | 0.461 | 0.618 | +0.157 | reward > score |
| GRPO 任意 step | 0.xxx | 0.xxx | **0** | 始终相等 |

---

## 五、GRPO vs PPO 全面对比

### 架构差异

| 维度 | GRPO | PPO |
|------|------|-----|
| Critic 网络 | **无** | 有（与 Actor 同等大小） |
| 显存占用 | 较低 | 约 2 倍（Actor + Critic） |
| Advantage 来源 | 组内归一化（无偏，零方差假设） | GAE + Critic（有估计偏差） |
| 采样策略 | n_agent ≥ 5（需要组内对比） | n_agent = 1 即可 |
| KL 惩罚方式 | 加入 loss 项 | 从 reward 中扣减 |

### 训练行为差异（实测）

| 维度 | GRPO（paperarg_v3） | PPO（paperarg_v1） |
|------|------|-----|
| **最高 val_score** | **0.492**（step 150） | 0.471（step 450） |
| **达到 0.40 所需步数** | ~50 步 | ~100 步 |
| **训练稳定性** | step 180 崩溃 | **全程 500 步无崩溃** |
| **每 step 总序列数** | 512×5=2560 | 512×1=512 |
| **每 step 耗时** | ~325s | ~180s |
| **response 长度趋势** | 稳定 ~700 | 持续增长到 ~1590 |
| **entropy 下降速度** | 快（0.61→0.30 @150步） | 慢（0.64→0.21 @450步） |
| **actor grad_norm** | 0.6~1.2（正常时） | 2~25（大得多） |
| **Critic vf_explained_var** | 无 Critic | 0→0.286（学得慢） |

### 结论

- **GRPO 学得快但不稳**：收敛速度约为 PPO 的 2~3 倍，但 KL 失控后直接崩溃
- **PPO 学得慢但更稳**：Critic 提供 baseline 降低了梯度方差，不易崩溃
- **PPO 的 Critic 学得不好**（explained_var 最高 0.286），限制了 PPO 的上限
- **PPO 的 response 长度失控**（1590 vs GRPO 的 700），模型学会"多搜几次碰运气"

---

## 六、GRPO 指标变化趋势速查

| 步数 | val_score | score/mean | kl_loss | entropy | pg_clipfrac |
|------|-----------|-----------|---------|---------|-------------|
| 0 | 0.295 | — | — | — | — |
| 1 | — | 0.303 | 0.001 | 0.611 | 0.000 |
| 50 | 0.401 | 0.406 | 0.096 | 0.520 | 0.002 |
| 100 | 0.467 | 0.430 | 0.151 | 0.376 | 0.001 |
| 150 | 0.492 | 0.465 | 0.370 | 0.345 | 0.001 |
| 200 | 0.000（崩溃） | 0.000 | nan | nan | 0.000 |

**崩溃原因**：kl_loss 持续增长（0.001→0.7），kl_loss_coef=0.001 太小无法约束；step 180 梯度爆炸（grad_norm: 0.7 → 59 → 2127 → 1.5亿），step 187 NaN 传入权重，模型永久死亡。

---

## 七、PPO 指标变化趋势速查

| 步数 | val_score | score/mean | rewards/mean | vf_loss | vf_explained_var | entropy | resp_len | grad_norm |
|------|-----------|-----------|-------------|---------|-----------------|---------|----------|-----------|
| 0 | 0.295 | — | — | — | — | — | — | — |
| 1 | — | 0.324 | 0.324 | 10.527 | -90.87 | 0.636 | 647 | 26.8 |
| 50 | 0.349 | 0.404 | 0.392 | 0.128 | 0.064 | 0.620 | 676 | 2.3 |
| 100 | 0.400 | 0.393 | 0.400 | 0.146 | 0.057 | 0.441 | 674 | 2.1 |
| 150 | 0.414 | 0.402 | 0.428 | 0.152 | 0.120 | 0.327 | 763 | 3.1 |
| 200 | 0.422 | 0.408 | 0.507 | 0.220 | 0.122 | 0.315 | 940 | 8.2 |
| 250 | 0.411 | 0.402 | 0.579 | 0.160 | 0.177 | 0.299 | 1371 | 11.3 |
| 300 | 0.432 | 0.406 | 0.607 | 0.105 | 0.244 | 0.217 | 1537 | 23.1 |
| 350 | 0.451 | 0.420 | 0.630 | 0.139 | 0.286 | 0.189 | 1479 | 25.2 |
| 400 | 0.455 | 0.428 | 0.594 | 0.103 | 0.270 | 0.249 | 1518 | 5.9 |
| 450 | 0.471 | 0.461 | 0.618 | 0.086 | 0.250 | 0.212 | 1590 | 4.3 |
| 500 | 0.464 | — | — | — | — | — | — | — |

**特点**：全程稳定无崩溃；Critic 从 vf_loss=10.5 冷启动逐步收敛到 0.086；response 长度从 647 增长到 1590；val_score 在 step 250 有回落后继续上升。

---

## 八、调参速查

### 关键超参数

| 参数 | 作用 | GRPO 建议 | PPO 建议 |
|------|------|-----------|----------|
| `kl_loss_coef` | 控制策略偏离参考模型的惩罚力度 | **0.01~0.05**（防崩溃） | 0.001（PPO 已有 reward 内的 KL） |
| `clip_ratio` | 限制单次更新的 ratio 范围 | 0.1~0.2 | 0.2 |
| `entropy_coeff` | 鼓励探索的力度 | 0.001~0.01 | 0.001 |
| `lr` | 学习率 | 5e-7~1e-6 | 1e-6 |
| `n_agent` | 每 prompt 采样次数 | ≥ 5（必须，用于组内对比） | 1 即可 |
| `grad_clip` | 梯度裁剪 | 1.0 | 1.0 |
| `ppo_epochs` | 每批 rollout 数据复用次数 | 1 | 1~4 |

### 异常排查速查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| kl_loss 持续上升 > 0.5 | kl_loss_coef 太小 | 增大 kl_loss_coef |
| entropy 快速下降 < 0.2 | 探索不足，策略过早收敛 | 增大 entropy_coeff |
| grad_norm 突然飙升 > 100 | 梯度爆炸前兆 | 降低 lr，增大 kl_loss_coef |
| finish_ratio < 0.9 | 模型格式退化 | 检查是否需要 SFT 冷启动 |
| score=0 且 clip_ratio=1 | 策略坍塌，模型输出全乱 | 从最近的 checkpoint 恢复 |
| vf_explained_var < 0 | Critic 未收敛（PPO 初期正常） | 等待更多步数，或调大 Critic lr |
| rewards ≫ score（PPO） | KL 项为负值，策略过度集中 | 检查 KL 系数和 entropy |


value_loss的更新，其实就是让vpred更加准确，但是returns 并没有真正的标注（ground truth），从实际 reward + Critic 的末尾估计值拼出来的"伪 GT"。
Critic 本质上是在"自我迭代"：用自己不太准的预测算出一个比纯预测稍好的目标（因为混入了真实 reward），然后学习这个目标，逐步逼近真实的价值函数。