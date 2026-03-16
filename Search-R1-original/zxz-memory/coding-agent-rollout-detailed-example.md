# Coding Agent vs Search-R1 Rollout 流程对比

## 核心逻辑对比

**结论**: Coding Agent 的 rollout 逻辑与 Search-R1 **完全一致**，只是把环境交互从 "HTTP 检索服务" 改为 "Docker 沙箱命令执行"。

---

## 完整 Rollout 流程示例

### 假设输入

**Task**: 实现一个 Verilog 模块 `top_64b66b_codec.sv`

**Initial Context Files** (已在 Docker 中):
```
/code/docs/specification.md
/code/rtl/encoder_data_64b66b.sv
/code/rtl/encoder_control_64b66b.sv
/code/rtl/decoder_data_control_64b66b.sv
```

**Prompt** (initial_input_ids):
```
<|system|>
You are a coding agent...
<bash>command</bash> to execute
<done>summary</done> when finished

<|user|>
Task: Implement top_64b66b_codec module...
Context files: /code/docs/specification.md, /code/rtl/...
```

---

## Turn-by-Turn Rollout 过程

### 初始化阶段

```python
# 1. 创建 Docker 沙箱 (每个样本一个容器)
self._init_sandboxes(batch_size=1)
# Docker 容器 sandbox-batch_0_0 创建成功
# 容器内已写入 context files 到 /code/

# 2. 初始化 rolling state
rollings = gen_batch  # 包含 initial prompt 的 input_ids
active_mask = [True]  # 该样本还在活跃状态
```

---

### Turn 1: 探索文件

```python
# Step 1.1: LLM 生成
gen_output = self._generate_with_gpu_padding(rollings_active)
# 生成 token ids，解码后得到:
response_str = "Let me first check what files are available.\n<bash>ls /code/rtl/</bash>"

# Step 1.2: 后处理 - 识别到 </bash> 标签
responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
# 截断到 </bash>，得到完整的 action

# Step 1.3: 解析 action
actions, contents = self.postprocess_predictions(responses_str)
# actions[0] = 'bash'
# contents[0] = 'ls /code/rtl/'

# Step 1.4: 执行 action - 进入 Docker 沙箱！
next_obs, dones, valid_action, is_search = self.execute_predictions(...)

# 内部执行:
if action == 'bash':
    sandbox = self.sandbox_pool.get_sandbox(sandbox_id)
    result = sandbox.exec('ls /code/rtl/')
    # Docker 容器执行: docker exec sandbox-batch_0_0 bash -c "ls /code/rtl/"
    # result.stdout = "encoder_data_64b66b.sv\nencoder_control_64b66b.sv\n..."
    
    next_obs[0] = '\n\n<observation>encoder_data_64b66b.sv\nencoder_control_64b66b.sv\n...</observation>\n\n'
    dones[0] = 0      # 未结束
    valid_action[0] = 1
    is_search[0] = 1  # 算作有效的"搜索"动作

# Step 1.5: 更新 rolling state
next_obs_ids = self._process_next_obs(next_obs)  # tokenize observation
rollings = self._update_rolling_state(
    rollings,
    responses_ids,      # LLM 生成的 <bash>ls /code/rtl/</bash>
    next_obs_ids        # <observation>...</observation>
)
# 现在 rollings 的 input_ids 包含了:
# [initial_prompt] + [<bash>ls /code/rtl/</bash>] + [<observation>encoder_data_64b66b.sv\n...</observation>]
```

**此时模型的上下文**:
```
<|system|>You are a coding agent...
<|user|>Task: Implement top_64b66b_codec...
<|assistant|>Let me first check what files are available.
<bash>ls /code/rtl/</bash>

<observation>encoder_data_64b66b.sv
encoder_control_64b66b.sv
decoder_data_control_64b66b.sv</observation>

```

---

### Turn 2: 读取规范

```python
# Step 2.1: LLM 继续生成 (基于上面的完整上下文)
gen_output = self._generate_with_gpu_padding(rollings_active)
response_str = "Now let me read the specification.\n<bash>cat /code/docs/specification.md</bash>"

# Step 2.2-2.3: 解析到 'bash' action，内容是 'cat /code/docs/specification.md'

# Step 2.4: 执行 - 再次进入 Docker 沙箱
result = sandbox.exec('cat /code/docs/specification.md')
# 返回 spec 内容
next_obs[0] = '\n\n<observation>## 64b66b Codec Specification\n...(2KB spec 内容)...</observation>\n\n'

# Step 2.5: 更新 rolling state
rollings = [prompt] + [turn1_bash] + [turn1_obs] + [turn2_bash] + [turn2_obs]
```

**此时上下文**:
```
<|assistant|>Let me first check...
<bash>ls /code/rtl/</bash>
<observation>encoder_data_64b66b.sv...</observation>

Now let me read the specification.
<bash>cat /code/docs/specification.md</bash>
<observation>## 64b66b Codec Specification
...</observation>

```

---

### Turn 3: 写代码

```python
# Step 3.1: LLM 生成
response_str = '''Based on the specification, I'll create the top module.
<bash>cat > /code/rtl/top_64b66b_codec.sv << 'EOF'
module top_64b66b_codec(
  input wire clk,
  input wire rst_n,
  ...
);
  encoder_data_64b66b enc_data(...);
  ...
endmodule
EOF
</bash>'''

# Step 3.4: 执行 - Docker 沙箱写文件
result = sandbox.exec('cat > /code/rtl/top_64b66b_codec.sv << \'EOF\'\\nmodule...')
next_obs[0] = '\n\n<observation>[exit code: 0]</observation>\n\n'  # 成功创建文件
```

---

### Turn 4: 编译验证

```python
# Step 4.1: LLM 生成
response_str = '''Let me compile to check for syntax errors.
<bash>iverilog -o /code/rundir/top.out -g2012 /code/rtl/*.sv</bash>'''

# Step 4.4: 执行 - Docker 编译
result = sandbox.exec('iverilog -o /code/rundir/top.out -g2012 /code/rtl/*.sv')
# 如果有错误，stderr 会包含编译错误信息
next_obs[0] = '\n\n<observation>/code/rtl/top_64b66b_codec.sv:15: syntax error\n...</observation>\n\n'
```

---

### Turn 5: 修复并完成

```python
# Step 5.1: LLM 看到错误，生成修复
response_str = '''I see the issue, let me fix line 15.
<bash>sed -i '15s/wire clk/wire clk,/' /code/rtl/top_64b66b_codec.sv</bash>'''

# 执行修复...

# Step 5.2: 再次编译确认
response_str = '''<bash>iverilog -o /code/rundir/top.out -g2012 /code/rtl/*.sv</bash>'''
# 成功: next_obs = '<observation>[exit code: 0]</observation>'

# Step 5.3: 完成
response_str = '''The module is implemented and compiles successfully.
<done>Implemented top_64b66b_codec module with encoder and decoder integration</done>'''

# Step 5.4: 识别到 <done>
actions[0] = 'done'
dones[0] = 1  # 标记为完成
active_mask[0] = False  # 该样本不再活跃
```

---

### 循环结束 - 运行测试

```python
# 所有样本都完成或达到 max_turns
# 此时 Docker 容器还在运行，文件都已修改完成

# 运行测试 harness
print("[CodingAgent] Running test harness...")
self.test_scores = self._run_tests_for_batch(batch_size)

# 对每个样本:
for i in range(batch_size):
    sandbox = self.sandbox_pool.get_sandbox(sandbox_ids[i])
    
    # 1. 写入 harness 文件
    for filepath, content in harness.items():
        sandbox._write_file(f'/code/harness/{filepath}', content)
    
    # 2. 执行 pytest
    result = sandbox.exec('cd /code/rundir && pytest -v /code/harness/src/test_runner.py')
    
    # 3. 解析输出
    # result.stdout = "...test_top_64b66b_codec.py::test_basic PASSED...5 passed, 0 failed..."
    pass_rate = self._parse_pytest_output(result.stdout)
    # pass_rate = 5/5 = 1.0
    
    test_scores[i] = pass_rate

print(f"[CodingAgent] Test scores: mean={np.mean(test_scores):.3f}")
# 输出: [CodingAgent] Test scores: mean=1.000, nonzero=1/1

# 清理容器
self._destroy_sandboxes()
```

---

## Search-R1 的对应流程

完全相同的循环结构，只是 `execute_predictions()` 不同：

### Search-R1 的 execute_predictions

```python
def execute_predictions(self, predictions, pad_token, active_mask, do_search=True):
    cur_actions, contents = self.postprocess_predictions(predictions)
    
    # 识别所有 search action
    search_queries = [content for action, content in zip(cur_actions, contents) 
                      if action == 'search']
    
    if do_search:
        # 调用 HTTP 检索服务
        search_results = self.batch_search(search_queries)
    
    for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
        if action == 'answer':
            next_obs.append('')
            dones.append(1)     # 结束
            
        elif action == 'search':
            # 获取检索结果
            result = search_results.pop(0)
            next_obs.append(f'\n\n<information>{result}</information>\n\n')
            dones.append(0)     # 继续
            is_search.append(1)
            
        else:  # invalid
            next_obs.append('Invalid action. Use <search> or <answer>.')
            dones.append(0)
            is_search.append(0)
    
    return next_obs, dones, valid_action, is_search
```

### Coding Agent 的 execute_predictions

```python
def execute_predictions(self, predictions, pad_token, active_mask, do_search=True):
    cur_actions, contents = self.postprocess_predictions(predictions)
    
    for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
        if action == 'done':
            next_obs.append('')
            dones.append(1)     # 结束
            
        elif action == 'bash':
            # 获取 Docker 沙箱
            sandbox = self.sandbox_pool.get_sandbox(sandbox_ids[i])
            
            # 执行命令
            result = sandbox.exec(contents[i])
            
            # 构造 observation
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            next_obs.append(f'\n\n<observation>{output}</observation>\n\n')
            dones.append(0)     # 继续
            is_search.append(1)
            
        else:  # invalid
            next_obs.append('Invalid action. Use <bash> or <done>.')
            dones.append(0)
            is_search.append(0)
    
    return next_obs, dones, valid_action, is_search
```

---

## 关键对应关系

| Search-R1 | Coding Agent | 说明 |
|-----------|--------------|------|
| `<search>query</search>` | `<bash>command</bash>` | Action 标签 |
| `<answer>result</answer>` | `<done>summary</done>` | 终止标签 |
| `<information>docs</information>` | `<observation>output</observation>` | Environment 返回 |
| `batch_search()` HTTP 调用 | `sandbox.exec()` Docker 调用 | 环境交互 |
| 检索服务返回文档 | Docker 返回 stdout/stderr | Observation 来源 |
| (无) | `run_tests()` pytest 执行 | Rollout 后的测试 |

---

## 时序图对比

### Search-R1 时序

```
LLM → <search>query</search>
    ↓ postprocess_responses (截断到 </search>)
    ↓ execute_predictions
    ↓ batch_search(query) → HTTP Server
    ↓ 返回 documents
    ↓ 构造 <information>docs</information>
    ↓ tokenize & update rolling state
LLM → (继续基于新上下文生成)
```

### Coding Agent 时序

```
LLM → <bash>ls /code/</bash>
    ↓ postprocess_responses (截断到 </bash>)
    ↓ execute_predictions
    ↓ sandbox.exec('ls /code/') → Docker Container
    ↓ 返回 stdout/stderr
    ↓ 构造 <observation>stdout</observation>
    ↓ tokenize & update rolling state
LLM → (继续基于新上下文生成)

# Rollout 结束后额外步骤:
    ↓ run_tests() → Docker pytest
    ↓ 返回 pass_rate (0-1)
    ↓ 注入到 batch.non_tensor_batch['test_scores']
```

---

## 总结

### 相同之处 (核心逻辑 100% 一致)

1. **Multi-turn loop**: 都是 `for step in range(max_turns)` 循环
2. **生成截断**: 都在识别到特定标签时截断 (`</search>` vs `</bash>`)
3. **环境交互**: 都是 `execute_predictions()` 调用外部环境
4. **状态更新**: 都是 `_update_rolling_state(responses, observations)`
5. **终止条件**: 都是识别到终止标签 (`</answer>` vs `<done>`)
6. **State masking**: 都用 `responses_with_info_mask` 标记环境返回的 token

### 不同之处 (仅环境接口)

1. **环境类型**: HTTP 检索 vs Docker 沙箱
2. **Action 内容**: 检索 query vs bash 命令
3. **Observation 来源**: 文档片段 vs stdout/stderr
4. **额外步骤**: Search-R1 无 / Coding Agent 有 pytest 测试

### 关键问答

**Q: Rollout 完如果识别到新的 `<bash>command</bash>` 就会进入沙箱操作吗？**

A: **不是 "rollout 完"，而是在 rollout 过程中，每识别到一个 `</bash>` 标签，就立即执行一次沙箱操作**。

具体时机:
```python
# 每一轮 (turn) 内:
1. LLM 生成 → 识别到 </bash> → 截断
2. 解析出 bash command
3. 立即调用 sandbox.exec(command)  ← 这里就进沙箱了
4. 获取 stdout/stderr
5. 构造 <observation>
6. 更新 rolling state
7. 继续下一轮生成 (LLM 会看到之前的 observation)
```

**Rollout 完成后 (所有 turn 结束)，才运行 pytest 测试获取最终 reward**。

---

## 完整数据流示例

```
Prompt (1024 tokens)
  ↓
Turn 1: LLM生成(512 tokens) → <bash>ls</bash> 
  → Docker exec → <observation>3 files</observation> (20 tokens)
  ↓
Turn 2: LLM生成(512 tokens) → <bash>cat spec.md</bash>
  → Docker exec → <observation>spec content</observation> (200 tokens)
  ↓
Turn 3: LLM生成(512 tokens) → <bash>cat > top.sv</bash>
  → Docker exec → <observation>[exit code: 0]</observation> (10 tokens)
  ↓
Turn 4: LLM生成(512 tokens) → <bash>iverilog</bash>
  → Docker exec → <observation>compiled</observation> (10 tokens)
  ↓
Turn 5: LLM生成(100 tokens) → <done>finished</done>
  ↓
Loop 结束，总 token 数: 1024 + (512+20) + (512+200) + (512+10) + (512+10) + 100 = 3412 tokens

最后运行测试:
  → Docker pytest → pass_rate = 0.8
  → Reward = 0.8 (注入到 batch)
```

这个 0.8 的 reward 会在后续的 GRPO 更新中，分配到整个 trajectory 的最后一个 token 上。
