"""
准备RL训练数据：将答错的题目转换为RL训练所需的格式
"""
import json
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Any


def convert_to_rl_format(wrong_samples_file: str, output_file: str):
    """
    将答错的题目转换为RL训练所需的parquet格式
    
    Args:
        wrong_samples_file: 答错题目的jsonl文件
        output_file: 输出的parquet文件路径
    """
    samples = []
    
    print(f"读取答错的题目: {wrong_samples_file}")
    with open(wrong_samples_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            samples.append(sample)
    
    print(f"共 {len(samples)} 个答错的题目")
    
    # 转换为RL训练所需的格式（与原始数据格式完全一致）
    rl_data = []
    for sample in samples:
        # 确保prompt格式正确：numpy数组包含字典列表
        prompt_content = sample.get('prompt', '')
        if isinstance(prompt_content, str):
            prompt_array = np.array([{'content': prompt_content, 'role': 'user'}], dtype=object)
        else:
            prompt_array = np.array(prompt_content, dtype=object) if isinstance(prompt_content, list) else prompt_content
        
        # 确保ground_truth格式正确
        ground_truth = sample.get('ground_truth', [])
        if isinstance(ground_truth, list):
            ground_truth_array = np.array(ground_truth, dtype=object)
        else:
            ground_truth_array = np.array([ground_truth], dtype=object) if not isinstance(ground_truth, np.ndarray) else ground_truth
        
        # 构建与原始数据格式完全一致的数据结构
        rl_data.append({
            'id': sample.get('id', ''),
            'question': sample.get('question', ''),
            'golden_answers': ground_truth_array,  # numpy数组格式
            'data_source': sample.get('data_source', 'nq'),
            'prompt': prompt_array,  # numpy数组格式，包含字典列表
            'ability': sample.get('ability', 'fact-reasoning'),
            'reward_model': {
                'ground_truth': {
                    'target': ground_truth_array  # numpy数组格式
                },
                'style': 'rule'
            },
            'extra_info': sample.get('extra_info', {'split': 'train'})
        })
    
    # 保存为parquet格式
    df = pd.DataFrame(rl_data)
    df.to_parquet(output_file, index=False)
    print(f"RL训练数据保存到: {output_file}")
    print(f"共 {len(df)} 条数据")
    
    # 验证格式
    print("\n验证数据格式...")
    test_row = df.iloc[0]
    print(f"  prompt type: {type(test_row['prompt'])}")
    print(f"  reward_model type: {type(test_row['reward_model'])}")
    print(f"  golden_answers type: {type(test_row['golden_answers'])}")
    if isinstance(test_row['prompt'], np.ndarray):
        print(f"  prompt content: {test_row['prompt']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备RL训练数据")
    parser.add_argument("--wrong_samples_file", type=str, required=True,
                       help="答错题目的jsonl文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                       help="输出的parquet文件路径")
    
    args = parser.parse_args()
    convert_to_rl_format(args.wrong_samples_file, args.output_file)
