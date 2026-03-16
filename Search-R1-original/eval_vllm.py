"""
Evaluate models via vLLM Chat Completions API with multi-turn search.

Usage:
  python eval_vllm.py --url http://localhost:8002 --model qwen-base --label "Base" -n 100
  python eval_vllm.py --url http://localhost:8003 --model qwen-sft-fixed --label "SFT-fixed" -n 100
"""

import argparse
import json
import time
import re
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

SEARCH_URL = "http://127.0.0.1:8000/retrieve"
MAX_TURNS = 5
MAX_TOKENS = 512
TOPK = 3
STOP_SEQUENCES = ["</search>", "</answer>"]


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


def chat_completion(api_url, model, messages, stop=None, temperature=0.0):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stop": stop or [],
    }
    resp = requests.post(f"{api_url}/v1/chat/completions", json=payload,
                         headers={"Authorization": "Bearer EMPTY"}, timeout=180)
    data = resp.json()
    if "choices" in data and data["choices"]:
        ch = data["choices"][0]
        content = ch.get("message", {}).get("content", "")
        reason = ch.get("finish_reason", "stop")
        if reason == "stop" and stop:
            for s in stop:
                tag_open = s.replace("/", "")
                if tag_open in content and s not in content:
                    content = content + s
                    break
        return content, reason
    return "", "error"


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


def run_sample(api_url, model, chat_prompt, question, golden):
    messages = list(chat_prompt)
    all_outputs = []
    turns = []

    for turn in range(MAX_TURNS):
        content, finish_reason = chat_completion(
            api_url, model, messages, stop=STOP_SEQUENCES, temperature=0.0
        )
        if not content and finish_reason == "error":
            turns.append({"turn": turn + 1, "action": "error"})
            break

        response_text, action = postprocess_response(content)
        all_outputs.append(response_text)

        if action == 'answer':
            turns.append({"turn": turn + 1, "action": "answer"})
            break
        elif action == 'search':
            query_match = re.search(r'<search>(.*?)</search>', response_text, re.DOTALL)
            query = query_match.group(1).strip() if query_match else ""
            search_results = batch_search(query)
            info_text = f'\n\n<information>{search_results.strip()}</information>\n\n'
            all_outputs.append(info_text)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": info_text})
            turns.append({"turn": turn + 1, "action": "search", "query": query})
        else:
            turns.append({"turn": turn + 1, "action": "no_valid_action"})
            break

    full_output = ''.join(all_outputs)
    return full_output, turns


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
    parser.add_argument("--url", required=True, help="vLLM API base URL")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--label", default=None, help="display label")
    parser.add_argument("-n", "--num_samples", type=int, default=100)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("-w", "--workers", type=int, default=8)
    args = parser.parse_args()

    label = args.label or args.model

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

    print("=" * 80)
    print(f"Model:       {label}")
    print(f"API:         {args.url} / {args.model}")
    print(f"Num samples: {len(test_samples)} (NQ: {sum(1 for s in test_samples if s[0]=='NQ')}, "
          f"HotpotQA: {sum(1 for s in test_samples if s[0]=='HotpotQA')})")
    print(f"Workers:     {args.workers}")
    print("=" * 80)

    stats = {
        "has_think": 0, "has_search": 0, "has_answer": 0,
        "has_info": 0, "correct": 0, "total": len(test_samples),
        "total_turns": 0, "search_count": 0,
    }
    results_log = [None] * len(test_samples)
    print_lock = threading.Lock()
    stats_lock = threading.Lock()
    completed = [0]
    t0 = time.time()

    def process_sample(idx):
        dataset, sample_idx, chat_prompt, question, golden = test_samples[idx]
        t_start = time.time()
        full_output, turns = run_sample(args.url, args.model, chat_prompt, question, golden)
        elapsed = time.time() - t_start

        predicted_answer = None
        ans_match = re.search(r'<answer>(.*?)</answer>', full_output, re.DOTALL)
        if ans_match:
            predicted_answer = ans_match.group(1).strip()

        is_correct = check_match(predicted_answer, golden)
        has_think = "<think>" in full_output and "</think>" in full_output
        has_search = "<search>" in full_output and "</search>" in full_output
        has_answer = "<answer>" in full_output and "</answer>" in full_output
        has_info = "<information>" in full_output
        n_searches = sum(1 for t in turns if t.get('action') == 'search')

        with stats_lock:
            stats["has_think"] += int(has_think)
            stats["has_search"] += int(has_search)
            stats["has_answer"] += int(has_answer)
            stats["has_info"] += int(has_info)
            stats["correct"] += int(is_correct)
            stats["total_turns"] += len(turns)
            stats["search_count"] += n_searches
            completed[0] += 1
            cnt = completed[0]

        result = {
            "dataset": dataset, "sample_idx": sample_idx,
            "question": question, "golden": golden,
            "predicted": predicted_answer, "correct": is_correct,
            "turns": len(turns), "searches": n_searches,
            "has_think": has_think, "has_search": has_search,
            "has_answer": has_answer, "has_info": has_info,
            "full_output": full_output,
        }
        results_log[idx] = result

        status = "✓" if is_correct else "✗"
        with print_lock:
            print(f"[{cnt:3d}/{len(test_samples)}] {dataset:9s} #{sample_idx:<4d} | "
                  f"turns={len(turns)} searches={n_searches} | "
                  f"think={'Y' if has_think else 'N'} answer={'Y' if has_answer else 'N'} | "
                  f"{status} pred=\"{(predicted_answer or '')[:50]}\" | {elapsed:.1f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_sample, i) for i in range(len(test_samples))]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Error: {e}")

    total_time = time.time() - t0
    total = stats["total"]

    print("\n" + "=" * 80)
    print(f"SUMMARY — {label}")
    print("=" * 80)
    print(f"Samples:                    {total}")
    print(f"Total time:                 {total_time:.0f}s ({total_time/total:.1f}s/sample)")
    print(f"Avg turns/sample:           {stats['total_turns']/total:.1f}")
    print(f"Avg searches/sample:        {stats['search_count']/total:.1f}")
    print(f"---")
    print(f"<think>...</think>:         {stats['has_think']}/{total} ({100*stats['has_think']/total:.0f}%)")
    print(f"<search>...</search>:       {stats['has_search']}/{total} ({100*stats['has_search']/total:.0f}%)")
    print(f"<answer>...</answer>:       {stats['has_answer']}/{total} ({100*stats['has_answer']/total:.0f}%)")
    print(f"<information>:              {stats['has_info']}/{total} ({100*stats['has_info']/total:.0f}%)")
    print(f"---")
    print(f"Answer correct (fuzzy):     {stats['correct']}/{total} ({100*stats['correct']/total:.0f}%)")

    if args.save:
        with open(args.save, 'w') as f:
            for r in results_log:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nDetailed results saved to: {args.save}")


if __name__ == "__main__":
    main()
