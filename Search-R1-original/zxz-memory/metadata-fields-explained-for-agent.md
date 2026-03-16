# Coding Agent 的"新人入职包"：metadata 字段详解

**视角**: 你是一个 Coding Agent，刚加入一个 Verilog 项目团队

---

## 场景设定

你收到一个任务 ticket：
```
Task ID: cvdp_agentic_64b66b_codec_0001
需求: 实现 64b/66b 编解码器的顶层模块
```

你的 PM 给你准备了一个 `metadata` 包，里面有三样东西：

---

## 1. `context` - 你的工作环境 (已有代码库)

**作用**: 这是你加入项目时，代码库里**已经存在的文件**

### 类比
就像你入职第一天，PM 说："这是我们的代码仓库，你先熟悉一下现有的模块"

### 实际内容

在这个例子中，`context` 包含 4 个文件：

```
/code/
├── docs/
│   └── specification.md          # 需求文档 (12KB)
└── rtl/
    ├── encoder_data_64b66b.sv    # 数据编码器 (已实现)
    ├── encoder_control_64b66b.sv # 控制编码器 (已实现)
    └── decoder_data_control_64b66b.sv  # 解码器 (已实现)
```

**这些文件会在你开始工作前，就已经写入到 Docker 沙箱的 `/code/` 目录下了**。

### 你能对 context 做什么？

1. **读取**: `<bash>cat /code/docs/specification.md</bash>` 
   - 了解需求规格
   
2. **查看**: `<bash>ls /code/rtl/</bash>`
   - 看看有哪些现成的模块可以复用
   
3. **分析**: `<bash>grep "module encoder" /code/rtl/*.sv</bash>`
   - 找出已有模块的接口定义
   
4. **编译测试**: `<bash>iverilog /code/rtl/encoder_data_64b66b.sv</bash>`
   - 验证现有模块是否能正常编译

### 完整示例 (specification.md 片段)

```markdown
# 64b/66b Codec Specification Document

## 1. Overview
The 64b/66b encoding scheme is a line coding technique defined by 
IEEE 802.3 standard for high-speed serial communication.

## 2. Architecture
- Encoder Data Module: Encodes 64-bit data to 66-bit
- Encoder Control Module: Handles control block encoding
- Decoder Module: Decodes 66-bit back to 64-bit

## 3. Top-Level Requirements
You need to implement `top_64b66b_codec` that:
- Instantiates encoder_data_64b66b
- Instantiates encoder_control_64b66b
- Instantiates decoder_data_control_64b66b
- Connects them properly
```

---

## 2. `patch` - 你的任务目标 (需要实现的文件)

**作用**: 这是你**需要创建/实现的文件**，也是**标准答案 (Ground Truth)**

### 类比
PM 给你的任务清单："你需要实现这个文件，完成后要能通过测试"

### 实际内容

在这个例子中，`patch` 包含 1 个文件：

```
📝 需要你实现:
rtl/top_64b66b_codec.sv  (当前是空的，或者不存在)
```

### 重要理解

1. **对于 Agent (你)**: 
   - 你**看不到** `patch` 的内容
   - 你只知道"需要实现 `top_64b66b_codec.sv`"
   - 你需要根据 `specification.md` 和现有模块的接口自己写代码

2. **对于训练系统**:
   - `patch` 是标准答案
   - 用于对比你写的代码和标准实现的差异 (可选的相似度 reward)
   - **但我们当前不用它计算 reward**，我们用测试通过率

3. **在这个 CVDP 数据集中**: 
   - `patch` 实际是空的 (size=0)
   - 因为 Verilog 实现方式很多，没有唯一标准答案
   - 评判标准是**测试是否通过**，而不是代码是否和标准答案一致

### 如果 patch 不为空的情况

假设在某些任务中，`patch` 包含标准实现：

```verilog
// patch['rtl/top_64b66b_codec.sv'] 的内容:
module top_64b66b_codec(
    input  logic clk_in,
    input  logic rst_in,
    // ...
);
    encoder_data_64b66b enc_data(...);
    encoder_control_64b66b enc_ctrl(...);
    decoder_data_control_64b66b dec(...);
endmodule
```

你完成任务后，系统会：
- 计算你的代码和 `patch` 的相似度 (如用 diff、编辑距离)
- 作为额外的 reward signal (可选)

---

## 3. `harness` - 你的验收测试 (Test Harness)

**作用**: PM 给你的**自动化测试套件**，用来验证你的实现是否正确

### 类比
PM 说："写完后，运行这些测试。所有测试通过，你的任务就完成了"

### 实际内容

在这个例子中，`harness` 包含 4 个文件：

```
/code/harness/
├── docker-compose.yml           # Docker 配置 (我们已内置到镜像)
├── src/
│   ├── .env                     # 测试环境变量
│   ├── test_runner.py           # pytest 运行器
│   └── test_top_64b66b_codec.py # 实际测试用例 (28KB!)
```

### 详细解析

#### 3.1 `.env` - 测试配置

```bash
SIM=icarus                        # 使用 Icarus Verilog 仿真器
TOPLEVEL_LANG=verilog             # 顶层语言是 Verilog
VERILOG_SOURCES=/code/rtl/top_64b66b_codec.sv /code/rtl/encoder_data_64b66b.sv ...
                                  # 需要编译的所有源文件
TOPLEVEL=top_64b66b_codec         # 顶层模块名
MODULE=test_top_64b66b_codec      # Python 测试模块名
PYTHONPATH=/src
HASH=5ae28b08...                  # 任务 hash (用于缓存)
```

**作用**: 告诉 cocotb 如何编译和运行仿真

#### 3.2 `test_runner.py` - pytest 入口

```python
import os
from cocotb_tools.runner import get_runner
import pytest

# 从环境变量读取配置
verilog_sources = os.getenv("VERILOG_SOURCES").split()
toplevel_lang = os.getenv("TOPLEVEL_LANG")
sim = os.getenv("SIM")
toplevel = os.getenv("TOPLEVEL")
module = os.getenv("MODULE")

def test_top_64b66b_codec():
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        ...
    )
    runner.test(
        toplevel=toplevel,
        test_module=module,
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**作用**: 
1. 编译你写的 Verilog 代码 (使用 iverilog)
2. 运行仿真测试

#### 3.3 `test_top_64b66b_codec.py` - 核心测试用例

这是一个 28KB 的 Python 文件，包含多个测试：

```python
import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
import random

async def dut_initialization(dut):
    """初始化 DUT (Device Under Test) 输入"""
    dut.rst_in.value = 1
    await Timer(100, units='ns')
    dut.rst_in.value = 0

@cocotb.test()
async def test_basic_encoding(dut):
    """测试基本编码功能"""
    # 启动时钟
    cocotb.start_soon(Clock(dut.clk_in, 10, units='ns').start())
    
    # 初始化
    await dut_initialization(dut)
    
    # 测试 1: 发送 64-bit 数据
    dut.data_in.value = 0x1234567890ABCDEF
    await RisingEdge(dut.clk_in)
    
    # 验证编码后是 66-bit
    assert len(bin(dut.encoded_out.value)) == 68  # 66 bits + '0b'
    
    # 验证同步头
    sync_header = (dut.encoded_out.value >> 64) & 0x3
    assert sync_header in [0b01, 0b10]

@cocotb.test()
async def test_control_block_encoding(dut):
    """测试控制块编码"""
    # ... 更多测试逻辑

@cocotb.test()
async def test_decoder_output(dut):
    """测试解码器输出"""
    # ... 验证解码后数据是否正确

@cocotb.test()
async def test_random_patterns(dut):
    """随机模式测试"""
    for _ in range(100):
        random_data = random.randint(0, 2**64-1)
        dut.data_in.value = random_data
        await RisingEdge(dut.clk_in)
        # 验证编解码一致性
        decoded = dut.decoded_out.value
        assert decoded == random_data
```

**作用**: 
- 通过 cocotb 驱动 Verilog 仿真
- 测试各种场景 (基本功能、边界情况、随机测试)
- 每个 `@cocotb.test()` 是一个测试用例

---

## Coding Agent 的完整工作流程

### 阶段 1: 环境准备 (Rollout 开始前)

```python
# 系统自动执行:
sandbox = DockerSandbox('task_0001')
sandbox.create(
    context_files=metadata['context'],  # 写入 4 个已有文件
    harness_files=metadata['harness']   # 写入测试框架
)
```

**此时 Docker 容器内的文件结构**:
```
/code/
├── docs/
│   └── specification.md          ← context
├── rtl/
│   ├── encoder_data_64b66b.sv    ← context
│   ├── encoder_control_64b66b.sv ← context
│   ├── decoder_data_control_64b66b.sv ← context
│   └── (top_64b66b_codec.sv 不存在，需要你创建)
├── harness/
│   ├── docker-compose.yml        ← harness
│   └── src/
│       ├── .env                  ← harness
│       ├── test_runner.py        ← harness
│       └── test_top_64b66b_codec.py ← harness
└── rundir/                        (空目录，用于编译输出)
```

### 阶段 2: Agent 工作 (Rollout 进行中)

**Turn 1: 探索环境**
```
Agent: <bash>ls -la /code/</bash>
Env:   <observation>
       docs/  rtl/  harness/  rundir/
       </observation>

Agent: <bash>cat /code/docs/specification.md | head -50</bash>
Env:   <observation>
       # 64b/66b Codec Specification
       ...需求内容...
       </observation>
```

**Turn 2: 查看现有模块**
```
Agent: <bash>grep "module encoder_data" /code/rtl/encoder_data_64b66b.sv</bash>
Env:   <observation>
       module encoder_data_64b66b (
           input  logic clk_in,
           input  logic rst_in,
           input  logic [63:0] data_in,
           output logic [65:0] data_out
       );
       </observation>
```

**Turn 3-5: 实现代码**
```
Agent: <bash>cat > /code/rtl/top_64b66b_codec.sv << 'EOF'
       module top_64b66b_codec(
           input  logic clk_in,
           input  logic rst_in,
           ...
       );
           encoder_data_64b66b enc_data(
               .clk_in(clk_in),
               .rst_in(rst_in),
               ...
           );
           ...
       endmodule
       EOF
       </bash>
Env:   <observation>[exit code: 0]</observation>
```

**Turn 6: 编译验证**
```
Agent: <bash>iverilog -o /code/rundir/top.vvp -g2012 /code/rtl/*.sv</bash>
Env:   <observation>
       /code/rtl/top_64b66b_codec.sv:15: syntax error
       </observation>
       
Agent: <bash>sed -i '15s/wire/logic/' /code/rtl/top_64b66b_codec.sv</bash>
Agent: <bash>iverilog -o /code/rundir/top.vvp -g2012 /code/rtl/*.sv</bash>
Env:   <observation>[exit code: 0]</observation>  ← 编译成功!
```

**Turn 7: 完成**
```
Agent: <done>Implemented top_64b66b_codec module integrating encoder and decoder</done>
```

### 阶段 3: 自动化测试 (Rollout 结束后)

```python
# 系统自动执行:
pass_rate, output = sandbox.run_tests(metadata['harness'])

# 内部流程:
# 1. cd /code/rundir
# 2. 设置环境变量 (从 .env 读取)
# 3. pytest -v /code/harness/src/test_runner.py
# 4. 解析输出:
#    test_basic_encoding PASSED
#    test_control_block_encoding PASSED  
#    test_decoder_output PASSED
#    test_random_patterns PASSED
#    =================== 4 passed ===================
# 5. pass_rate = 4/4 = 1.0

print(f"Test pass rate: {pass_rate}")  # 输出: 1.0
```

**Reward 计算**:
```python
reward = pass_rate  # 1.0 表示满分，0.0 表示全挂
```

---

## 关键理解

### 1. Context ≠ 你能看到的全部

**你能看到**: 通过 `<bash>cat /code/docs/specification.md</bash>` 读取

**你看不到**: `patch` (标准答案) 的内容

这就像真实工作：
- PM 给你需求文档 (specification)
- 告诉你代码库结构 (context files)
- 但不会直接给你答案
- 你需要自己实现

### 2. Harness = 你的 CI/CD 测试

就像 GitHub 上的 CI pipeline:
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: pytest tests/
```

你的代码 push 后，自动跑测试：
- ✅ 全部通过 → Merge
- ❌ 有失败 → 需要修复

在 RL 训练中:
- ✅ pass_rate = 1.0 → reward = 1.0 → 强化这个 trajectory
- ❌ pass_rate = 0.2 → reward = 0.2 → 削弱这个 trajectory

### 3. Patch 的双重角色

**训练时** (Agent 视角):
- Agent 完全不知道 `patch` 存在
- 只能根据 `context` 和需求文档自己写

**评估时** (可选):
- 可以用 `patch` 计算代码相似度
- 作为额外的 reward shaping
- 但我们当前只用 `pass_rate`

---

## 完整数据流图

```
[数据集] cvdp_v1.0.2_agentic_code_generation.jsonl
    ↓
[预处理] preprocess_cvdp_coding.py
    ↓
[Parquet] train.parquet
    ├── question: "实现 64b/66b codec..."
    ├── metadata: JSON {
    │       context: {已有代码},
    │       patch: {标准答案},
    │       harness: {测试框架}
    │   }
    ↓
[训练时]
    ├─→ [Docker 初始化] 写入 context + harness
    ├─→ [Agent Rollout] 
    │      LLM 看到: question
    │      LLM 生成: <bash>...</bash>
    │      环境执行: Docker exec
    │      环境返回: <observation>...</observation>
    │      循环 max_turns 次
    ├─→ [测试执行] pytest harness
    └─→ [Reward] pass_rate (0-1)
```

---

## 类比总结

想象你是一个远程工作的工程师：

| 现实场景 | Coding Agent 对应 |
|---------|------------------|
| PM 发给你的 Jira ticket | `question` (任务描述) |
| 你 clone 的代码仓库 | `context` (已有文件) |
| 你需要实现的功能 | `patch` (需要创建的文件) |
| CI/CD 测试套件 | `harness` (pytest + cocotb) |
| 你的开发环境 (VSCode + Terminal) | Docker 沙箱 + bash |
| 你提交 PR 后的测试结果 | `pass_rate` (reward) |
| Code Review 对比标准实现 | `patch` 相似度 (可选 reward) |

你的工作流程：
1. 读需求 (`cat specification.md`)
2. 看现有代码 (`ls rtl/`, `grep module`)
3. 写新模块 (`cat > top.sv`)
4. 本地编译 (`iverilog`)
5. 修 bug (根据 stderr)
6. 提交 (`<done>`)
7. 等待 CI 测试结果 (pytest)
8. 获得反馈 (pass_rate → reward → policy update)

---

希望这样解释清楚了！从 Agent 的视角看，这就是一个标准的软件开发流程。
