"""
Preprocess the HotpotQA (distractor) dataset to parquet format
for Search-R1 training/evaluation.

Key differences from nq_search.py:
  - Loads from local disk (arrow format) or HuggingFace Hub
  - answer field is a plain string → wrapped into list for reward_model
  - Supports filtering by question type (bridge / comparison)
  - yes/no answers filtered out by default (--keep_yn to retain)

Usage:
  # Generate train + test from local data (default)
  python hotpot_search.py --local_dir ./data/hotpot_search

  # Keep yes/no answers (comparison type)
  python hotpot_search.py --keep_yn

  # Use only bridge-type questions
  python hotpot_search.py --question_type bridge

  # Load from HuggingFace Hub instead of local disk
  python hotpot_search.py --from_hub
"""

import os
import argparse
import datasets

from verl.utils.hdfs_io import copy, makedirs


def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'base':
        prefix = (
            f"Answer the given question. "
            f"You must conduct reasoning inside <think> and </think> first every time you get new information. "
            f"After reasoning, if you find you lack some knowledge, you can call a search engine by "
            f"<search> query </search> and it will return the top searched results between "
            f"<information> and </information>. "
            f"You can search as many times as your want. "
            f"If you find no further external knowledge needed, you can directly provide the answer inside "
            f"<answer> and </answer>, without detailed illustrations. "
            f"For example, <answer> Beijing </answer>. Question: {question}\n"
        )
    else:
        raise NotImplementedError(f"template_type '{template_type}' is not supported.")
    return prefix


def make_map_fn(split, template_type):
    def process_fn(example, idx):
        question = example['question'].strip()
        if not question.endswith('?'):
            question += '?'
        example['question'] = question

        prompt = make_prefix(example, template_type=template_type)

        # HotpotQA answer is a plain string; wrap as list to match NQ format
        answer = example['answer'].strip()

        data = {
            "data_source": "hotpotqa",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": [answer]},
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "type": example.get("type", ""),
                "level": example.get("level", ""),
            },
        }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess HotpotQA for Search-R1 training.")
    parser.add_argument('--local_dir', default='./data/hotpot_search',
                        help='Output directory for parquet files.')
    parser.add_argument('--hdfs_dir', default=None,
                        help='Optional HDFS destination.')
    parser.add_argument('--template_type', type=str, default='base',
                        help='Prompt template type.')
    parser.add_argument('--data_path', type=str,
                        default='./data/zxz-other-data/hotpotqa_hotpot_qa_distractor',
                        help='Local path to the HotpotQA arrow dataset.')
    parser.add_argument('--from_hub', action='store_true',
                        help='Load from HuggingFace Hub instead of local disk.')
    parser.add_argument('--question_type', type=str, default=None,
                        choices=['bridge', 'comparison'],
                        help='Filter to a specific question type. Default: keep all.')
    parser.add_argument('--keep_yn', action='store_true',
                        help='Keep yes/no answers (default: filtered out).')
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if args.from_hub:
        print("Loading HotpotQA from HuggingFace Hub (hotpotqa/hotpot_qa, distractor)...")
        ds = datasets.load_dataset('hotpotqa/hotpot_qa', 'distractor')
    else:
        print(f"Loading HotpotQA from local disk: {args.data_path}")
        ds = datasets.load_from_disk(args.data_path)

    train_dataset = ds['train']
    # HotpotQA uses 'validation' split; treat it as the test set
    test_dataset  = ds['validation']

    print(f"Raw sizes  — train: {len(train_dataset)}, validation: {len(test_dataset)}")

    # ── Filter ────────────────────────────────────────────────────────────────
    def should_keep(example):
        if not args.keep_yn and example['answer'].strip().lower() in ('yes', 'no'):
            return False
        if args.question_type and example['type'] != args.question_type:
            return False
        return True

    train_dataset = train_dataset.filter(should_keep)
    test_dataset  = test_dataset.filter(should_keep)

    print(f"After filtering — train: {len(train_dataset)}, validation: {len(test_dataset)}")
    if args.question_type:
        print(f"  question_type filter: {args.question_type}")
    if not args.keep_yn:
        print("  yes/no answers removed")

    # ── Map ───────────────────────────────────────────────────────────────────
    train_dataset = train_dataset.map(
        function=make_map_fn('train', args.template_type),
        with_indices=True,
        desc="Processing train",
    )
    test_dataset = test_dataset.map(
        function=make_map_fn('test', args.template_type),
        with_indices=True,
        desc="Processing validation",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, 'train.parquet')
    test_path  = os.path.join(args.local_dir, 'test.parquet')

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Saved → {train_path}  ({len(train_dataset)} rows)")
    print(f"Saved → {test_path}   ({len(test_dataset)} rows)")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
