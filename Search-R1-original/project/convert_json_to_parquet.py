#!/usr/bin/env python3
"""
将 JSON 格式的 Search-R1 数据转换为 Parquet 格式
用于 VERL 训练框架

用法:
    python convert_json_to_parquet.py <input.json> <output.parquet>
"""

import json
import sys
import pandas as pd
from pathlib import Path


def convert_json_to_parquet(json_path: str, parquet_path: str):
    """
    将 JSON 数据转换为 Parquet 格式
    
    Args:
        json_path: 输入 JSON 文件路径
        parquet_path: 输出 Parquet 文件路径
    """
    print(f"📖 读取 JSON 文件: {json_path}")
    
    # 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 成功读取 {len(data)} 条数据")
    
    # 转换为 DataFrame
    print("🔄 转换为 DataFrame...")
    df = pd.DataFrame(data)
    
    # 显示数据统计
    print(f"\n📊 数据统计:")
    print(f"  - 总样本数: {len(df)}")
    print(f"  - 列名: {list(df.columns)}")
    print(f"  - 数据来源: {df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else '未知'}")
    
    # 保存为 Parquet
    print(f"\n💾 保存为 Parquet: {parquet_path}")
    output_path = Path(parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    # 检查文件大小
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ 转换完成! 文件大小: {file_size:.2f} MB")
    
    # 验证读取
    print("\n🔍 验证转换结果...")
    df_verify = pd.read_parquet(parquet_path)
    print(f"  - 验证样本数: {len(df_verify)}")
    print(f"  - 验证列名: {list(df_verify.columns)}")
    
    if len(df_verify) == len(df):
        print("✅ 验证通过!")
    else:
        print(f"⚠️  警告: 数据条数不匹配 (原始: {len(df)}, 转换后: {len(df_verify)})")
    
    return df


def main():
    if len(sys.argv) != 3:
        print("用法: python convert_json_to_parquet.py <input.json> <output.parquet>")
        print("\n示例:")
        print("  python convert_json_to_parquet.py data/qa_2000_searchr1_format.json data/qa_2000_searchr1_format.parquet")
        sys.exit(1)
    
    json_path = sys.argv[1]
    parquet_path = sys.argv[2]
    
    # 检查输入文件
    if not Path(json_path).exists():
        print(f"❌ 错误: 找不到输入文件: {json_path}")
        sys.exit(1)
    
    try:
        convert_json_to_parquet(json_path, parquet_path)
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
