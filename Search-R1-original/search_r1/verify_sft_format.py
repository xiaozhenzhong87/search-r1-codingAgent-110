"""
验证SFT数据格式：检查是否包含必要的格式标签
"""
import pandas as pd
import argparse
from typing import List, Dict


def verify_sft_format(data_file: str):
    """
    验证SFT数据格式
    
    Args:
        data_file: SFT数据集文件路径（parquet格式）
    """
    df = pd.read_parquet(data_file)
    
    print(f"验证SFT数据集: {data_file}")
    print(f"总样本数: {len(df)}")
    print()
    
    # 必需的格式标签
    required_tags = ['<think>', '<search>', '<information>', '<answer>']
    
    # 统计信息
    missing_tags_count = {tag: 0 for tag in required_tags}
    has_all_tags_count = 0
    has_no_tags_count = 0
    
    # 检查每个样本
    issues = []
    for idx, row in df.iterrows():
        response = row.get('response', '')
        prompt = row.get('prompt', '')
        
        if not isinstance(response, str):
            response = str(response)
        
        # 检查格式标签
        missing_tags = [tag for tag in required_tags if tag not in response]
        
        if missing_tags:
            for tag in missing_tags:
                missing_tags_count[tag] += 1
            
            if len(missing_tags) == len(required_tags):
                has_no_tags_count += 1
                issues.append({
                    'id': row.get('id', idx),
                    'issue': '缺少所有格式标签',
                    'response_preview': response[:200] + '...' if len(response) > 200 else response
                })
            else:
                issues.append({
                    'id': row.get('id', idx),
                    'issue': f'缺少标签: {missing_tags}',
                    'response_preview': response[:200] + '...' if len(response) > 200 else response
                })
        else:
            has_all_tags_count += 1
        
        # 检查response是否包含prompt（不应该包含）
        if prompt and isinstance(prompt, str) and prompt in response:
            issues.append({
                'id': row.get('id', idx),
                'issue': 'response中包含了prompt内容',
                'response_preview': response[:200] + '...' if len(response) > 200 else response
            })
    
    # 打印统计信息
    print("=" * 60)
    print("格式标签统计")
    print("=" * 60)
    print(f"包含所有格式标签的样本: {has_all_tags_count} ({has_all_tags_count/len(df)*100:.2f}%)")
    print(f"缺少所有格式标签的样本: {has_no_tags_count} ({has_no_tags_count/len(df)*100:.2f}%)")
    print()
    print("缺少标签统计:")
    for tag, count in missing_tags_count.items():
        if count > 0:
            print(f"  {tag}: {count} 个样本缺少")
    print()
    
    # 打印问题样本
    if issues:
        print("=" * 60)
        print(f"发现 {len(issues)} 个问题样本（显示前10个）:")
        print("=" * 60)
        for i, issue in enumerate(issues[:10]):
            print(f"\n样本 {issue['id']}:")
            print(f"  问题: {issue['issue']}")
            print(f"  响应预览: {issue['response_preview']}")
    else:
        print("=" * 60)
        print("✓ 所有样本格式正确！")
        print("=" * 60)
    
    # 显示示例
    print("\n" + "=" * 60)
    print("示例样本（包含所有格式标签）:")
    print("=" * 60)
    for idx, row in df.iterrows():
        response = row.get('response', '')
        if isinstance(response, str):
            has_all = all(tag in response for tag in required_tags)
            if has_all:
                print(f"\n样本ID: {row.get('id', idx)}")
                print(f"Prompt: {row.get('prompt', '')[:100]}...")
                print(f"Response: {response[:500]}...")
                break
    
    return len(issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证SFT数据格式")
    parser.add_argument("--data_file", type=str, required=True,
                       help="SFT数据集文件路径")
    
    args = parser.parse_args()
    is_valid = verify_sft_format(args.data_file)
    
    if not is_valid:
        print("\n⚠️  数据格式存在问题，请检查！")
        exit(1)
    else:
        print("\n✓ 数据格式验证通过！")
