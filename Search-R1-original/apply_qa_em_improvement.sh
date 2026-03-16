#!/usr/bin/env bash
#
# 应用 Reward 改进到实际训练代码 (qa_em.py)
#
# 使用方法:
#   cd /ssd1/zz/AI_efficency/RAG/Search-R1
#   ./apply_qa_em_improvement.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TARGET_FILE="verl/utils/reward_score/qa_em.py"
BACKUP_FILE="verl/utils/reward_score/qa_em.py.backup"
IMPROVED_FILE="verl/utils/reward_score/qa_em_improved.py"

echo "========================================================================"
echo "  应用 qa_em.py Reward 改进 - 惩罚 answer 后继续输出"
echo "========================================================================"
echo ""

# 检查文件是否存在
if [ ! -f "$TARGET_FILE" ]; then
    echo "✗ 错误: 找不到 $TARGET_FILE"
    exit 1
fi

if [ ! -f "$IMPROVED_FILE" ]; then
    echo "✗ 错误: 找不到 $IMPROVED_FILE"
    exit 1
fi

# 1. 备份原文件
if [ ! -f "$BACKUP_FILE" ]; then
    echo "[1/4] 备份原文件..."
    cp "$TARGET_FILE" "$BACKUP_FILE"
    echo "  ✓ 已备份到: $BACKUP_FILE"
else
    echo "[1/4] 备份已存在: $BACKUP_FILE"
fi

echo ""
echo "[2/4] 添加 has_continuation_after_answer 函数..."

# 2. 在 extract_solution 函数后添加新函数
/ssd1/zz/envs/searchr1/bin/python << 'EOF'
import re

# 读取文件
with open('verl/utils/reward_score/qa_em.py', 'r') as f:
    content = f.read()

# 要插入的新函数
new_function = '''

def has_continuation_after_answer(solution_str):
    """
    检查 </answer> 后是否有多余输出 (关键改进!)
    
    Returns:
        bool: True 表示有多余输出, False 表示干净
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    # 需要至少3个match (2个在prompt, 1个是模型输出)
    if len(matches) <= 2:
        return False
    
    # 获取最后一个 </answer> 的结束位置
    last_answer_end = matches[-1].end()
    
    # 检查后面的内容
    after_answer = solution_str[last_answer_end:].strip()
    
    # 检查是否有 <search>, <think>, <information> 标签
    has_extra_tags = bool(
        re.search(r'<(search|think|information)>', after_answer)
    )
    
    return has_extra_tags
'''

# 找到 extract_solution 函数的结束位置 (在第一个 def compute_score 之前)
pattern = r'(def extract_solution.*?return matches\[-1\]\.group\(1\)\.strip\(\)\s*\n)'
match = re.search(pattern, content, re.DOTALL)

if match:
    insert_pos = match.end()
    new_content = content[:insert_pos] + new_function + '\n' + content[insert_pos:]
    
    with open('verl/utils/reward_score/qa_em.py', 'w') as f:
        f.write(new_content)
    
    print("  ✓ 已添加 has_continuation_after_answer 函数")
else:
    print("  ✗ 未找到插入位置")
    exit(1)

EOF

echo ""
echo "[3/4] 修改 compute_score_em 函数..."

# 3. 修改 compute_score_em 和 compute_score_subem 函数
/ssd1/zz/envs/searchr1/bin/python << 'EOF'
import re

with open('verl/utils/reward_score/qa_em.py', 'r') as f:
    content = f.read()

# 修改 compute_score_em 函数
# 1. 添加 penalty_ratio 参数
content = re.sub(
    r'def compute_score_em\(solution_str, ground_truth, method=\'strict\', format_score=0\., score=1\.\):',
    'def compute_score_em(solution_str, ground_truth, method=\'strict\', format_score=0., score=1., penalty_ratio=0.2):',
    content
)

# 2. 在 docstring 后添加 has_continuation 检查
# 找到 "answer = extract_solution" 这一行,在前面插入
pattern = r'(    """.*?""")\s*(answer = extract_solution\(solution_str=solution_str\))'
replacement = r'\1\n    answer = extract_solution(solution_str=solution_str)\n    has_continuation = has_continuation_after_answer(solution_str)'

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 3. 修改打印部分,添加 has_continuation 信息
content = re.sub(
    r'(print\(f"Extracted answer: \{answer\}"\))',
    r'\1\n        print(f"Has continuation after answer: {has_continuation}")',
    content
)

# 4. 修改返回逻辑 - 找到 em_check 部分
old_logic = r'''    if answer is None:
        return 0
    else:
        if em_check\(answer, ground_truth\['target'\]\):
            return score
        else:
            return format_score'''

new_logic = '''    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            # 答案正确
            if has_continuation:
                # 有多余输出,给予惩罚
                return score * (1 - penalty_ratio)  # 默认: 1.0 * 0.8 = 0.8
            else:
                # 完美输出
                return score  # 1.0
        else:
            # 答案错误
            return format_score  # 0.0 (您的设置)'''

content = re.sub(old_logic, new_logic, content)

# 5. 同样修改 compute_score_subem
# 添加 penalty_ratio 参数
content = re.sub(
    r'def compute_score_subem\(solution_str, ground_truth, method=\'strict\', format_score=0\., score=1\.\):',
    'def compute_score_subem(solution_str, ground_truth, method=\'strict\', format_score=0., score=1., penalty_ratio=0.2):',
    content
)

# 为 subem 添加 has_continuation 检查 (找第二个 answer = extract_solution)
# 先找到 compute_score_subem 的位置
subem_start = content.find('def compute_score_subem')
if subem_start != -1:
    # 在这个范围内找 answer = extract_solution
    subem_section = content[subem_start:]
    subem_pattern = r'(answer = extract_solution\(solution_str=solution_str\))'
    if re.search(subem_pattern, subem_section):
        # 在第二次出现的地方插入
        parts = content.split('def compute_score_subem')
        if len(parts) == 2:
            subem_part = parts[1]
            subem_part = re.sub(
                r'(answer = extract_solution\(solution_str=solution_str\))',
                r'\1\n    has_continuation = has_continuation_after_answer(solution_str)',
                subem_part,
                count=1
            )
            # 也要修改 subem 的打印
            subem_part = re.sub(
                r'(print\(f"Extracted answer: \{answer\}"\))',
                r'\1\n        print(f"Has continuation after answer: {has_continuation}")',
                subem_part,
                count=1
            )
            # 修改 subem 的返回逻辑
            subem_old_logic = r'''    if answer is None:
        return 0
    else:
        if subem_check\(answer, ground_truth\['target'\]\):
            return score
        else:
            return format_score'''
            
            subem_new_logic = '''    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            # 答案正确
            if has_continuation:
                return score * (1 - penalty_ratio)
            else:
                return score
        else:
            return format_score'''
            
            subem_part = re.sub(subem_old_logic, subem_new_logic, subem_part)
            content = parts[0] + 'def compute_score_subem' + subem_part

with open('verl/utils/reward_score/qa_em.py', 'w') as f:
    f.write(content)

print("  ✓ 已修改 compute_score_em 和 compute_score_subem 函数")

EOF

echo ""
echo "[4/4] 验证语法..."

# 4. 验证Python语法
if /ssd1/zz/envs/searchr1/bin/python -m py_compile "$TARGET_FILE" 2>/dev/null; then
    echo "  ✓ Python 语法检查通过"
else
    echo "  ✗ 语法错误!正在恢复备份..."
    cp "$BACKUP_FILE" "$TARGET_FILE"
    echo "  ✓ 已恢复原文件"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  修改内容对比 (部分)"
echo "========================================================================"
echo ""
diff -u "$BACKUP_FILE" "$TARGET_FILE" | head -80 || true

echo ""
echo "========================================================================"
echo "  应用完成!"
echo "========================================================================"
echo ""
echo "✅ 主要改进:"
echo "  1. 添加了 has_continuation_after_answer() 函数"
echo "  2. compute_score_em 添加 penalty_ratio 参数 (默认0.2)"
echo "  3. 答案正确但有多余输出: 从 1.0 → 0.8"
echo "  4. 答案错误: 保持 0.0 (您的设置)"
echo ""
echo "📊 新的Reward规则:"
echo "  - 答案正确 + 无多余输出: 1.0 ✅"
echo "  - 答案正确 + 有多余输出: 0.8 ⚠️ (惩罚-0.2)"
echo "  - 答案错误: 0.0 ❌ (保持原设置)"
echo ""
echo "下一步:"
echo "  1. 查看详细diff: diff -u $BACKUP_FILE $TARGET_FILE | less"
echo "  2. 使用新reward重新训练"
echo "  3. 监控训练日志中的 'Has continuation' 信息"
echo ""
echo "如需恢复: cp $BACKUP_FILE $TARGET_FILE"
echo ""
