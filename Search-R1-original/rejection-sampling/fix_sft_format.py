"""
Fix SFT dataset format issues:
  1. Insert </think> before the first <search> tag
  2. Wrap intermediate reasoning (after </information>) in <think>...</think>
  3. Truncate everything after the first </answer>

Usage:
    python fix_sft_format.py \
        --input  output/vllm_rs_10k/sft_dataset.parquet \
        --output output/vllm_rs_10k/sft_dataset_fixed.parquet
"""

import argparse
import re
import pandas as pd


def fix_response(text: str) -> str:
    # ── Step 1: Truncate after first </answer> ──
    m = re.search(r'</answer>', text)
    if m:
        text = text[:m.end()]

    # ── Step 2: Fix opening <think>...</think> before first <search> ──
    first_search = text.find('<search>')
    if first_search > 0:
        before_search = text[:first_search]
        after_search = text[first_search:]

        if '<think>' not in before_search:
            # No <think> at all - add <think>...</think> wrapper
            text = '<think> I need to search this.</think>\n\n' + after_search
        elif '</think>' not in before_search:
            # Has <think> but no </think> - close it
            before_search = before_search.rstrip()
            before_search += '</think>\n\n'
            text = before_search + after_search
    elif first_search == 0:
        # Response starts directly with <search>
        text = '<think> I need to search this.</think>\n\n' + text

    # ── Step 3: Wrap intermediate reasoning in <think>...</think> ──
    # After </information>\n\n, any text before <search> or <answer> should be wrapped
    def wrap_reasoning(match):
        info_end = match.group(1)       # "</information>\n\n"
        reasoning = match.group(2)      # intermediate text
        next_tag = match.group(3)       # "<search>" or "<answer>"

        reasoning_stripped = reasoning.strip()
        if not reasoning_stripped:
            return info_end + next_tag

        if '<think>' in reasoning_stripped and '</think>' in reasoning_stripped:
            return info_end + reasoning + next_tag

        return info_end + '<think>' + reasoning_stripped + '</think>\n\n' + next_tag

    text = re.sub(
        r'(</information>\s*\n*\s*)(.*?)(<search>|<answer>)',
        wrap_reasoning,
        text,
        flags=re.DOTALL,
    )

    return text


def validate(text: str) -> dict:
    """Check format tags in a fixed response."""
    return {
        'has_think_open': '<think>' in text,
        'has_think_close': '</think>' in text,
        'has_search': '<search>' in text and '</search>' in text,
        'has_answer': '<answer>' in text and '</answer>' in text,
        'has_info': '<information>' in text,
        'has_post_answer_junk': bool(re.search(r'</answer>.{10,}', text, re.DOTALL)),
        'think_before_search': '</think>' in text[:text.find('<search>')] if '<search>' in text else True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} samples from {args.input}')

    # ── Before stats ──
    print('\n=== BEFORE FIX ===')
    before_close_think = sum(1 for r in df['response'] if '</think>' in r)
    before_post_answer = sum(1 for r in df['response']
                            if re.search(r'</answer>.{10,}', r, re.DOTALL))
    print(f'  Has </think>:           {before_close_think}/{len(df)}')
    print(f'  Has post-answer junk:   {before_post_answer}/{len(df)}')

    # ── Apply fix ──
    df['response'] = df['response'].apply(fix_response)

    # ── After stats ──
    print('\n=== AFTER FIX ===')
    stats = {k: 0 for k in ['has_think_open', 'has_think_close', 'has_search',
                              'has_answer', 'has_info', 'has_post_answer_junk',
                              'think_before_search']}
    for r in df['response']:
        v = validate(r)
        for k in stats:
            stats[k] += int(v[k])

    total = len(df)
    print(f'  <think>:               {stats["has_think_open"]}/{total}')
    print(f'  </think>:              {stats["has_think_close"]}/{total}')
    print(f'  </think> before <search>: {stats["think_before_search"]}/{total}')
    print(f'  <search>...</search>:  {stats["has_search"]}/{total}')
    print(f'  <answer>...</answer>:  {stats["has_answer"]}/{total}')
    print(f'  <information>:         {stats["has_info"]}/{total}')
    print(f'  Post-answer junk:      {stats["has_post_answer_junk"]}/{total}')

    # ── Show examples ──
    print('\n=== EXAMPLE (sample 0) ===')
    print(df.iloc[0]['response'][:600])
    print('...')

    # ── Avg response length comparison ──
    orig = pd.read_parquet(args.input)
    avg_before = orig['response'].str.len().mean()
    avg_after = df['response'].str.len().mean()
    print(f'\nAvg response length: {avg_before:.0f} → {avg_after:.0f} chars '
          f'({100*avg_after/avg_before:.0f}%)')

    # ── Save ──
    df.to_parquet(args.output, index=False)
    print(f'\nSaved to {args.output} ({len(df)} rows)')


if __name__ == '__main__':
    main()
