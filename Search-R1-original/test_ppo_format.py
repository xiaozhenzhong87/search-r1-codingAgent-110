"""
Test PPO checkpoint output format stability.

Checks:
  - Tag completeness: <think>, <search>, <answer> paired correctly
  - Tag ordering: think -> search/answer flow
  - Multi-turn behavior: search -> information -> think pattern
  - Invalid action rate
  - Nested/broken tags
  - Overall accuracy

Usage:
  python test_ppo_format.py -n 30
  python test_ppo_format.py -n 30 --save results.jsonl -v
"""

import argparse
import json
import time
import re
import torch
import transformers
import pandas as pd
import requests

SEARCH_URL = "http://127.0.0.1:8000/retrieve"
MAX_TURNS = 5
MAX_NEW_TOKENS = 512
TOPK = 3
QWEN_EOS_IDS = [151645, 151643]

DEFAULT_MODEL = "/ssd1/zz/AI_efficency/RAG/Search-R1/verl_checkpoints/nq-hp-ppo-sft-fixed-7b/actor/global_step_400"


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in target_sequences
        ]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(self.target_ids):
            t = torch.as_tensor(target, device=input_ids.device)
            if torch.equal(input_ids[0, -self.target_lengths[i]:], t):
                return True
        return False


def batch_search(query):
    payload = {"queries": [query], "topk": TOPK, "return_scores": True}
    try:
        results = requests.post(SEARCH_URL, json=payload, timeout=30).json()['result']
    except Exception as e:
        return f"[Search failed: {e}]"
    formatted = ''
    for idx, doc_item in enumerate(results[0]):
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        formatted += f"Doc {idx+1}(Title: {title}) {text}\n"
    return formatted


def postprocess_response(text):
    if '</search>' in text:
        text = text.split('</search>')[0] + '</search>'
    elif '</answer>' in text:
        text = text.split('</answer>')[0] + '</answer>'
    pattern = r'<(search|answer)>(.*?)</\1>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return text, match.group(1)
    return text, None


def run_inference(model, tokenizer, chat_prompt, stopping_criteria, device):
    prompt = tokenizer.apply_chat_template(
        chat_prompt, add_generation_prompt=True, tokenize=False
    )
    all_model_outputs = []
    turns = []

    for turn in range(MAX_TURNS):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        generated_tokens = outputs[0][input_ids.shape[1]:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        all_model_outputs.append(raw_output)
        response_text, action = postprocess_response(raw_output)

        if action == 'answer':
            turns.append({"turn": turn + 1, "action": "answer", "raw": raw_output})
            break
        elif action == 'search':
            query_match = re.search(r'<search>(.*?)</search>', response_text, re.DOTALL)
            query = query_match.group(1).strip() if query_match else ""
            search_results = batch_search(query)
            next_obs = f'\n\n<information>{search_results.strip()}</information>\n\n'
            prompt = prompt + response_text + next_obs
            all_model_outputs.append(next_obs)
            turns.append({"turn": turn + 1, "action": "search", "query": query, "raw": raw_output})
        else:
            turns.append({"turn": turn + 1, "action": "no_valid_action", "raw": raw_output})
            if outputs[0][-1].item() in QWEN_EOS_IDS:
                break
            prompt = prompt + response_text
            break

    full_output = ''.join(all_model_outputs)
    return full_output, turns


def analyze_format(full_output, turns):
    """Detailed format analysis of a single sample output."""
    result = {}

    think_opens = len(re.findall(r'<think>', full_output))
    think_closes = len(re.findall(r'</think>', full_output))
    search_opens = len(re.findall(r'<search>', full_output))
    search_closes = len(re.findall(r'</search>', full_output))
    answer_opens = len(re.findall(r'<answer>', full_output))
    answer_closes = len(re.findall(r'</answer>', full_output))
    info_opens = len(re.findall(r'<information>', full_output))
    info_closes = len(re.findall(r'</information>', full_output))

    result['think_paired'] = think_opens == think_closes and think_opens > 0
    result['search_paired'] = search_opens == search_closes
    result['answer_paired'] = answer_opens == answer_closes and answer_opens > 0
    result['info_paired'] = info_opens == info_closes

    result['think_count'] = think_opens
    result['search_count'] = search_opens
    result['answer_count'] = answer_opens

    result['has_think'] = think_opens > 0 and think_closes > 0
    result['has_search'] = search_opens > 0 and search_closes > 0
    result['has_answer'] = answer_opens > 0 and answer_closes > 0

    nested_think = len(re.findall(r'<think>[^<]*<think>', full_output))
    result['nested_think'] = nested_think > 0

    result['think_before_action'] = True
    for t in turns:
        raw = t.get('raw', '')
        if t['action'] in ('search', 'answer'):
            if '<think>' not in raw:
                result['think_before_action'] = False
                break

    result['ends_with_answer'] = turns[-1]['action'] == 'answer' if turns else False

    n_search_turns = sum(1 for t in turns if t['action'] == 'search')
    n_answer_turns = sum(1 for t in turns if t['action'] == 'answer')
    n_invalid = sum(1 for t in turns if t['action'] == 'no_valid_action')
    result['n_turns'] = len(turns)
    result['n_search_turns'] = n_search_turns
    result['n_answer_turns'] = n_answer_turns
    result['n_invalid_turns'] = n_invalid

    result['valid_flow'] = (
        result['has_think'] and
        result['ends_with_answer'] and
        n_invalid == 0 and
        not result['nested_think']
    )

    empty_think = len(re.findall(r'<think>\s*</think>', full_output))
    result['empty_think_count'] = empty_think

    repetitive_search = False
    queries = [t.get('query', '') for t in turns if t['action'] == 'search']
    if len(queries) != len(set(queries)):
        repetitive_search = True
    result['repetitive_search'] = repetitive_search

    return result


def check_match(predicted, golden):
    if not predicted or len(golden) == 0:
        return False
    pred_lower = predicted.lower().strip()
    for g in golden:
        if isinstance(g, str):
            gl = g.lower().strip()
            if gl in pred_lower or pred_lower in gl:
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-n", "--num_samples", type=int, default=30)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(f"  PPO Format Stability Test")
    print(f"  Model: {args.model_path}")
    print(f"  Samples: {args.num_samples}")
    print("=" * 80)

    device = torch.device("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    target_sequences = [
        "</search>", " </search>",
        "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n",
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    nq_df = pd.read_parquet("data/nq_search/test.parquet")
    hotpot_df = pd.read_parquet("data/hotpot_search/test.parquet")

    n_nq = args.num_samples // 2
    n_hp = args.num_samples - n_nq

    test_samples = []
    step_nq = max(1, len(nq_df) // n_nq) if n_nq > 0 else 1
    for i in range(0, len(nq_df), step_nq):
        if len(test_samples) >= n_nq:
            break
        row = nq_df.iloc[i]
        golden = row['golden_answers']
        if hasattr(golden, 'tolist'):
            golden = golden.tolist()
        elif isinstance(golden, str):
            golden = [golden]
        test_samples.append(("NQ", i, row['prompt'].tolist(), row['question'], golden))

    hotpot_answer_col = 'golden_answers' if 'golden_answers' in hotpot_df.columns else 'answer'
    step_hp = max(1, len(hotpot_df) // n_hp) if n_hp > 0 else 1
    for i in range(0, len(hotpot_df), step_hp):
        if len(test_samples) >= args.num_samples:
            break
        row = hotpot_df.iloc[i]
        ans = row[hotpot_answer_col]
        if isinstance(ans, str):
            ans = [ans]
        elif hasattr(ans, 'tolist'):
            ans = ans.tolist()
        test_samples.append(("HotpotQA", i, row['prompt'].tolist(), row['question'], ans))

    print(f"\nLoaded {len(test_samples)} samples "
          f"(NQ: {sum(1 for s in test_samples if s[0]=='NQ')}, "
          f"HotpotQA: {sum(1 for s in test_samples if s[0]=='HotpotQA')})")
    print("-" * 80)

    all_format_results = []
    results_log = []
    t0 = time.time()

    for idx, (dataset, sample_idx, chat_prompt, question, golden) in enumerate(test_samples):
        t_start = time.time()
        full_output, turns = run_inference(
            model, tokenizer, chat_prompt, stopping_criteria, device
        )
        elapsed = time.time() - t_start

        predicted_answer = None
        ans_match = re.search(r'<answer>(.*?)</answer>', full_output, re.DOTALL)
        if ans_match:
            predicted_answer = ans_match.group(1).strip()

        is_correct = check_match(predicted_answer, golden)
        fmt = analyze_format(full_output, turns)
        fmt['correct'] = is_correct
        all_format_results.append(fmt)

        status = "OK" if fmt['valid_flow'] else "!!"
        corr = "Y" if is_correct else "N"
        issues = []
        if not fmt['has_think']:
            issues.append("no_think")
        if not fmt['ends_with_answer']:
            issues.append("no_answer_end")
        if fmt['n_invalid_turns'] > 0:
            issues.append(f"invalid_x{fmt['n_invalid_turns']}")
        if fmt['nested_think']:
            issues.append("nested_think")
        if fmt['empty_think_count'] > 0:
            issues.append(f"empty_think_x{fmt['empty_think_count']}")
        if fmt['repetitive_search']:
            issues.append("repeat_search")
        if not fmt['think_before_action']:
            issues.append("no_think_before_act")

        issue_str = ",".join(issues) if issues else "clean"
        print(f"[{idx+1:3d}/{len(test_samples)}] {dataset:9s} [{status}] "
              f"turns={fmt['n_turns']} search={fmt['n_search_turns']} "
              f"correct={corr} | {issue_str} | {elapsed:.1f}s")

        if args.verbose:
            print(f"    Q: {question[:80]}")
            print(f"    Gold: {golden}")
            print(f"    Pred: {(predicted_answer or 'N/A')[:60]}")
            print(f"    Output preview: {full_output[:150].replace(chr(10), ' ')}...")
            print()

        results_log.append({
            "dataset": dataset, "sample_idx": sample_idx,
            "question": question, "golden": golden,
            "predicted": predicted_answer, "correct": is_correct,
            "format": fmt, "full_output": full_output,
        })

    total_time = time.time() - t0
    total = len(all_format_results)

    # Aggregate
    def pct(count):
        return f"{count}/{total} ({100*count/total:.0f}%)"

    n_valid_flow = sum(1 for f in all_format_results if f['valid_flow'])
    n_has_think = sum(1 for f in all_format_results if f['has_think'])
    n_has_search = sum(1 for f in all_format_results if f['has_search'])
    n_has_answer = sum(1 for f in all_format_results if f['has_answer'])
    n_think_paired = sum(1 for f in all_format_results if f['think_paired'])
    n_answer_paired = sum(1 for f in all_format_results if f['answer_paired'])
    n_ends_answer = sum(1 for f in all_format_results if f['ends_with_answer'])
    n_nested_think = sum(1 for f in all_format_results if f['nested_think'])
    n_invalid = sum(1 for f in all_format_results if f['n_invalid_turns'] > 0)
    n_think_before = sum(1 for f in all_format_results if f['think_before_action'])
    n_empty_think = sum(1 for f in all_format_results if f['empty_think_count'] > 0)
    n_repeat_search = sum(1 for f in all_format_results if f['repetitive_search'])
    n_correct = sum(1 for f in all_format_results if f['correct'])

    avg_turns = sum(f['n_turns'] for f in all_format_results) / total
    avg_searches = sum(f['n_search_turns'] for f in all_format_results) / total
    avg_think = sum(f['think_count'] for f in all_format_results) / total

    print("\n" + "=" * 80)
    print("  FORMAT STABILITY REPORT")
    print("=" * 80)

    print(f"\n[Overall]")
    print(f"  Valid flow (think+answer, no errors):  {pct(n_valid_flow)}")
    print(f"  Answer correct (fuzzy match):          {pct(n_correct)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/total:.1f}s/sample)")

    print(f"\n[Tag Presence]")
    print(f"  Has <think>...</think>:       {pct(n_has_think)}")
    print(f"  Has <search>...</search>:     {pct(n_has_search)}")
    print(f"  Has <answer>...</answer>:     {pct(n_has_answer)}")

    print(f"\n[Tag Correctness]")
    print(f"  <think> properly paired:      {pct(n_think_paired)}")
    print(f"  <answer> properly paired:     {pct(n_answer_paired)}")
    print(f"  Ends with <answer>:           {pct(n_ends_answer)}")
    print(f"  Think before every action:    {pct(n_think_before)}")

    print(f"\n[Issues]")
    print(f"  Nested <think> tags:          {pct(n_nested_think)}")
    print(f"  Empty <think></think>:        {pct(n_empty_think)}")
    print(f"  Invalid action turns:         {pct(n_invalid)}")
    print(f"  Repetitive search queries:    {pct(n_repeat_search)}")

    print(f"\n[Behavior Stats]")
    print(f"  Avg turns/sample:             {avg_turns:.2f}")
    print(f"  Avg searches/sample:          {avg_searches:.2f}")
    print(f"  Avg think blocks/sample:      {avg_think:.2f}")

    turn_dist = {}
    for f in all_format_results:
        t = f['n_turns']
        turn_dist[t] = turn_dist.get(t, 0) + 1
    print(f"\n[Turn Distribution]")
    for t in sorted(turn_dist.keys()):
        bar = "#" * turn_dist[t]
        print(f"  {t} turns: {turn_dist[t]:3d} {bar}")

    search_dist = {}
    for f in all_format_results:
        s = f['n_search_turns']
        search_dist[s] = search_dist.get(s, 0) + 1
    print(f"\n[Search Count Distribution]")
    for s in sorted(search_dist.keys()):
        bar = "#" * search_dist[s]
        print(f"  {s} searches: {search_dist[s]:3d} {bar}")

    print("\n" + "=" * 80)
    verdict = "STABLE" if n_valid_flow / total >= 0.9 else "UNSTABLE" if n_valid_flow / total < 0.7 else "MARGINAL"
    print(f"  VERDICT: Format is {verdict} ({100*n_valid_flow/total:.0f}% valid flow)")
    print("=" * 80)

    if args.save:
        with open(args.save, 'w') as f:
            for r in results_log:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nDetailed results saved to: {args.save}")


if __name__ == "__main__":
    main()
