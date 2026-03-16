#!/usr/bin/env python3
"""
Preprocess CVDP JSONL into verl training format (parquet).
Creates the data format expected by RLHFDataset with proper chat template structure.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


SYSTEM_PROMPT = """You are a coding agent that writes Verilog/SystemVerilog code. You have access to a Docker sandbox with the following tools:

- Execute any bash command: <bash>command</bash>
- Signal task completion: <done>summary of what was done</done>

Available commands in the sandbox:
- `ls`, `tree` - list files
- `cat <file>` - read file contents
- `echo "content" > <file>` or use heredoc - write files
- `iverilog -o out.vvp -g2012 file.sv` - compile Verilog
- `vvp out.vvp` - run simulation

Workflow:
1. First explore the existing files to understand the codebase
2. Read the specification documents
3. Write the required Verilog module(s)
4. Compile and test your code
5. Fix any errors
6. When done, use <done>summary</done>

Always think step by step. Use <bash>...</bash> for every command you want to execute."""


def format_context_files(context: dict) -> str:
    """Format context file listing for the user prompt."""
    if not context:
        return "No existing files."
    lines = ["The following files are already in the /code/ directory:"]
    for filepath in sorted(context.keys()):
        lines.append(f"  - /code/{filepath}")
    return "\n".join(lines)


def convert_cvdp_to_verl(input_jsonl: str, output_dir: str):
    print(f"Reading CVDP dataset: {input_jsonl}")

    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                context = data.get('context', {})
                file_listing = format_context_files(context)

                user_message = f"""{data['prompt']}

{file_listing}

Start by exploring the available files, then implement the required module(s)."""

                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]

                metadata = {
                    'id': data['id'],
                    'categories': data['categories'],
                    'context': data.get('context', {}),
                    'patch': data.get('patch', {}),
                    'harness': data.get('harness', {}),
                }

                reward_model = {
                    'ground_truth': {
                        'target': list(data.get('patch', {}).values()),
                    },
                    'style': 'rule',
                }

                sample = {
                    'id': data['id'],
                    'question': data['prompt'][:200],
                    'data_source': 'cvdp',
                    'prompt': prompt,
                    'ability': 'coding',
                    'reward_model': reward_model,
                    'extra_info': {'index': line_no - 1, 'split': 'train'},
                    'metadata': json.dumps(metadata, ensure_ascii=False),
                }

                samples.append(sample)

            except Exception as e:
                print(f"Error on line {line_no}: {e}")
                continue

    print(f"Processed {len(samples)} samples")

    df = pd.DataFrame(samples)

    # Split: 85 train, 7 val (roughly 92% / 8%)
    n_val = max(5, len(df) // 13)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df = df.iloc[-n_val:].reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / 'train.parquet'
    val_path = output_path / 'test.parquet'

    train_df.to_parquet(train_path, index=False, engine='pyarrow')
    val_df.to_parquet(val_path, index=False, engine='pyarrow')

    print(f"\nDataset statistics:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Saved to: {output_path}")

    # Verify
    verify_df = pd.read_parquet(train_path)
    print(f"\nVerification:")
    print(f"  Columns: {list(verify_df.columns)}")
    row = verify_df.iloc[0]
    print(f"  data_source: {row['data_source']}")
    print(f"  prompt type: {type(row['prompt'])}")
    print(f"  prompt[0]: {row['prompt'][0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
        default='/ssd1/zz/AI_efficency/RAG/search/data/cvdp-benchmark-dataset/cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl')
    parser.add_argument('--output', type=str,
        default='/ssd1/zz/AI_efficency/RAG/Search-R1/data/cvdp_coding')
    args = parser.parse_args()
    convert_cvdp_to_verl(args.input, args.output)
    print("\nDone!")


if __name__ == '__main__':
    main()
