#!/usr/bin/env python3
import json
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "./output/test_step400_100samples/rejection_sampling.jsonl"

samples = []
with open(input_file) as f:
    for line in f:
        samples.append(json.loads(line))

print("=" * 60)
print("  模型输出效果分析")
print("=" * 60)
print()

# 整体统计
n = len(samples)
qids = set(s['question_id'] for s in samples)
correct = sum(1 for s in samples if s['reward'] >= 1.0)
has_think = sum(1 for s in samples if '<think>' in s.get('response', ''))
has_answer = sum(1 for s in samples if '<answer>' in s.get('response', ''))
avg_turns = sum(s['num_turns'] for s in samples) / max(n, 1)
q_correct = set(s['question_id'] for s in samples if s['reward'] >= 1.0)

print(f"问题数:         {len(qids)}")
print(f"总样本数:       {n} (每题{n//len(qids)}次采样)")
print(f"正确样本:       {correct} / {n} ({100*correct/n:.1f}%)")
print(f"至少1次答对:    {len(q_correct)} / {len(qids)} ({100*len(q_correct)/len(qids):.1f}%)")
print(f"格式正确率:")
print(f"  - 含<think>:  {has_think} / {n} ({100*has_think/n:.1f}%)")
print(f"  - 含<answer>: {has_answer} / {n} ({100*has_answer/n:.1f}%)")
print(f"平均轮数:       {avg_turns:.2f}")
print()

# 查看几个正确和错误的样本
print("=" * 60)
print("  示例样本")
print("=" * 60)
print()

correct_samples = [s for s in samples if s['reward'] >= 1.0]
incorrect_samples = [s for s in samples if s['reward'] < 1.0]

if correct_samples:
    print("【正确样本示例 1】")
    s = correct_samples[0]
    print(f"问题: {s['question']}")
    print(f"模型答案: {s['answer']}")
    print(f"标准答案: {s['ground_truth']}")
    print(f"轮数: {s['num_turns']}")
    print()

if len(correct_samples) > 1:
    print("【正确样本示例 2】")
    s = correct_samples[1]
    print(f"问题: {s['question']}")
    print(f"模型答案: {s['answer']}")
    print(f"标准答案: {s['ground_truth']}")
    print(f"轮数: {s['num_turns']}")
    print()

if incorrect_samples:
    print("【错误样本示例 1】")
    s = incorrect_samples[0]
    print(f"问题: {s['question']}")
    print(f"模型答案: {s['answer']}")
    print(f"标准答案: {s['ground_truth']}")
    print(f"轮数: {s['num_turns']}")
    print(f"响应片段: {s['response'][:500]}...")
    print()

# 按reward分布
print("=" * 60)
print("  Reward 分布")
print("=" * 60)
reward_dist = {}
for s in samples:
    r = s['reward']
    reward_dist[r] = reward_dist.get(r, 0) + 1

for r in sorted(reward_dist.keys(), reverse=True):
    print(f"Reward {r:.1f}: {reward_dist[r]} 样本 ({100*reward_dist[r]/n:.1f}%)")
print()

# 轮数分布
print("=" * 60)
print("  轮数分布")
print("=" * 60)
turn_dist = {}
for s in samples:
    t = s['num_turns']
    turn_dist[t] = turn_dist.get(t, 0) + 1

for t in sorted(turn_dist.keys()):
    print(f"{t} 轮: {turn_dist[t]} 样本 ({100*turn_dist[t]/n:.1f}%)")
print()
