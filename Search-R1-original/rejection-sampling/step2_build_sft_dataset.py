"""
Step 2: Build SFT dataset from Rejection Sampling results.
For each question pick the best response, output verl-format parquet.

Usage:
    python step2_build_sft_dataset.py \
        --input_file  ./output/rejection_sampling_results.jsonl \
        --output_file ./output/sft_dataset.parquet \
        --strategy best
"""

import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_TAGS = ["<think>", "<answer>"]


def select_best(samples: List[Dict]) -> List[Dict]:
    return [max(samples, key=lambda x: x["reward"])]


def select_random_best(samples: List[Dict]) -> List[Dict]:
    perfect = [s for s in samples if s["reward"] >= 1.0]
    if perfect:
        return [random.choice(perfect)]
    return select_best(samples)


def select_all_correct(samples: List[Dict]) -> List[Dict]:
    perfect = [s for s in samples if s["reward"] >= 1.0]
    if perfect:
        return perfect
    return select_best(samples)


STRATEGIES = {
    "best": select_best,
    "random_best": select_random_best,
    "all_correct": select_all_correct,
}


def check_format_tags(response: str) -> bool:
    return all(tag in response for tag in REQUIRED_TAGS)


def build_sft_record(sample: Dict) -> Dict:
    """Convert a sampling result to verl SFT format."""
    prompt_text = sample["prompt"]
    response_text = sample["response"]
    golden = sample.get("ground_truth", [])

    return {
        "data_source": sample.get("data_source", "nq"),
        "prompt": [{"role": "user", "content": prompt_text}],
        "response": response_text,
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": golden},
        },
        "extra_info": {
            "question_id": sample.get("question_id", ""),
            "question": sample.get("question", ""),
            "reward": sample.get("reward", 0.0),
            "answer": sample.get("answer", ""),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Step 2: Build SFT dataset")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Step 1 output jsonl file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output parquet for SFT")
    parser.add_argument("--strategy", type=str, default="best",
                        choices=list(STRATEGIES.keys()),
                        help="Selection strategy: best / random_best / all_correct")
    parser.add_argument("--min_reward", type=float, default=0.0,
                        help="Only include samples with reward >= this")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Group by question_id
    groups = defaultdict(list)
    logger.info("Reading %s ...", args.input_file)
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            sample = json.loads(line.strip())
            groups[sample["question_id"]].append(sample)
    logger.info("Loaded %d questions, %d total samples",
                len(groups), sum(len(v) for v in groups.values()))

    # Select best per question
    selector = STRATEGIES[args.strategy]
    sft_records = []
    format_ok_count = 0
    skip_low_reward = 0

    skip_no_answer = 0
    for qid, samples in tqdm(groups.items(), desc="Selecting"):
        selected = selector(samples)
        for s in selected:
            if s["reward"] < args.min_reward:
                skip_low_reward += 1
                continue
            if not check_format_tags(s["response"]):
                fallback = [x for x in samples if check_format_tags(x["response"])]
                if fallback:
                    s = max(fallback, key=lambda x: x["reward"])
                else:
                    skip_no_answer += 1
                    continue
            format_ok_count += 1
            sft_records.append(build_sft_record(s))

    logger.info("Selected %d SFT samples (strategy=%s)", len(sft_records), args.strategy)
    if skip_low_reward:
        logger.info("Skipped %d samples with reward < %.2f", skip_low_reward, args.min_reward)
    if skip_no_answer:
        logger.info("Skipped %d questions where no sample has <think>+<answer> tags", skip_no_answer)
    logger.info("Format-tag check: %d / %d (%.1f%%)",
                format_ok_count, len(sft_records),
                100 * format_ok_count / max(len(sft_records), 1))

    # Save parquet
    df = pd.DataFrame(sft_records)
    df.to_parquet(args.output_file, index=False)
    logger.info("Saved to %s (%d rows)", args.output_file, len(df))

    # Stats
    rewards = [r["extra_info"]["reward"] for r in sft_records]
    if rewards:
        logger.info("--- Stats ---")
        logger.info("  Total:    %d", len(rewards))
        logger.info("  Avg rew:  %.4f", sum(rewards) / len(rewards))
        correct = sum(1 for r in rewards if r >= 1.0)
        logger.info("  Correct:  %d (%.1f%%)", correct, 100 * correct / len(rewards))


if __name__ == "__main__":
    main()
