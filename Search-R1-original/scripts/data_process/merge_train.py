"""
Merge multiple processed parquet train files into one unified training set.

Usage:
  # Merge NQ + HotpotQA (bridge only)
  python scripts/data_process/merge_train.py \
      --input_files data/nq_search/train.parquet data/hotpot_search/train.parquet \
      --output_dir  data/nq_hotpot_search \
      --shuffle

  # Check sizes before merging (dry run)
  python scripts/data_process/merge_train.py \
      --input_files data/nq_search/train.parquet data/hotpot_search/train.parquet \
      --dry_run
"""

import os
import argparse
from collections import Counter
import datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge multiple parquet train files.")
    parser.add_argument(
        '--input_files',
        nargs='+',
        required=True,
        help='Paths to parquet files to merge (order matters for logging).'
    )
    parser.add_argument(
        '--output_dir',
        default='./data/nq_hotpot_search',
        help='Output directory. Saves as train.parquet.'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle merged dataset (recommended for training).'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling.'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only print sizes, do not write output.'
    )
    args = parser.parse_args()

    all_datasets = []
    total = 0

    for path in args.input_files:
        ds = datasets.load_dataset('parquet', data_files=path, split='train')
        print(f"  {path}: {len(ds):>7,} rows")

        # 打印 data_source 分布（如果存在该列）
        if 'data_source' in ds.column_names:
            src_counter = Counter(ds['data_source'])
            print(f"    data_source: {dict(src_counter)}")
        else:
            print("    data_source: <column not found>")

        all_datasets.append(ds)
        total += len(ds)

    print(f"\n合并后总量: {total:,} 条")

    if args.dry_run:
        print("dry_run 模式，不写文件。")
        raise SystemExit(0)

    merged = datasets.concatenate_datasets(all_datasets)

    if args.shuffle:
        merged = merged.shuffle(seed=args.seed)
        print(f"已 shuffle（seed={args.seed}）")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'train.parquet')
    merged.to_parquet(out_path)

    print(f"\n已保存 → {out_path}  ({len(merged):,} 行)")