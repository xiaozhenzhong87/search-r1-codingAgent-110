"""
Preprocess PubMedQA into Search-R1 parquet format.

- Training set: pqa_artificial (211,269 samples, yes/no labels)
- Test set:     pqa_labeled   (1,000 samples, yes/no/maybe expert labels)

Usage:
  python pubmedqa_search.py --local_dir ./data/pubmedqa_search
"""

import os
import argparse
import pandas as pd
import datasets


def make_prefix(question, template_type):
    if template_type == "base":
        prefix = (
            f"Answer the given question. "
            f"You must conduct reasoning inside <think> and </think> first every time you get new information. "
            f"After reasoning, if you find you lack some knowledge, you can call a search engine by "
            f"<search> query </search> and it will return the top searched results between "
            f"<information> and </information>. "
            f"You can search as many times as your want. "
            f"If you find no further external knowledge needed, you can directly provide the answer inside "
            f"<answer> and </answer>. "
            f"The answer MUST be one of: yes, no, or maybe. "
            f"For example, <answer> yes </answer>. Question: {question}\n"
        )
    else:
        raise NotImplementedError(f"template_type '{template_type}' is not supported.")
    return prefix


def parquet_to_hf_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)
    return datasets.Dataset.from_pandas(df)


def make_map_fn(split, template_type):
    def process_fn(example, idx):
        question = example["question"].strip()
        if not question.endswith("?"):
            question += "?"

        prompt = make_prefix(question, template_type=template_type)
        answer = example["final_decision"].strip().lower()

        data = {
            "data_source": "pubmedqa",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": [answer]},
            },
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PubMedQA for Search-R1.")
    parser.add_argument(
        "--local_dir",
        default="./data/pubmedqa_search",
        help="Output directory (also contains raw pqa_* subdirs).",
    )
    parser.add_argument("--template_type", type=str, default="base")
    args = parser.parse_args()

    train_parquet = os.path.join(
        args.local_dir, "pqa_artificial", "train-00000-of-00001.parquet"
    )
    test_parquet = os.path.join(
        args.local_dir, "pqa_labeled", "train-00000-of-00001.parquet"
    )

    print(f"Loading training data from {train_parquet}")
    train_dataset = parquet_to_hf_dataset(train_parquet)
    print(f"Loading test data from {test_parquet}")
    test_dataset = parquet_to_hf_dataset(test_parquet)

    print(f"Raw sizes — train: {len(train_dataset)}, test: {len(test_dataset)}")

    train_dataset = train_dataset.map(
        function=make_map_fn("train", args.template_type),
        with_indices=True,
        desc="Processing train",
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", args.template_type),
        with_indices=True,
        desc="Processing test",
    )

    train_path = os.path.join(args.local_dir, "train.parquet")
    test_path = os.path.join(args.local_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Saved → {train_path}  ({len(train_dataset)} rows)")
    print(f"Saved → {test_path}   ({len(test_dataset)} rows)")
