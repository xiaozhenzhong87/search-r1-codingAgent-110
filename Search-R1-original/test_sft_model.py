"""
Test model output format with NQ and HotpotQA samples.

Usage:
  python test_sft_model.py                        # SFT model, 8 samples
  python test_sft_model.py --base                 # base model, 8 samples
  python test_sft_model.py --base -n 50           # base model, 50 samples
  python test_sft_model.py -n 50 --save result.jsonl  # SFT, 50 samples, save details
"""

import argparse
import json
import time
import torch
import transformers
import pandas as pd
import requests
import re

SEARCH_URL = "http://127.0.0.1:8000/retrieve"
MAX_TURNS = 5
MAX_NEW_TOKENS = 512
TOPK = 3
QWEN_EOS_IDS = [151645, 151643]


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
        response_text, action = postprocess_response(raw_output)
        all_model_outputs.append(response_text)

        if action == 'answer':
            turns.append({"turn": turn + 1, "action": "answer"})
            break
        elif action == 'search':
            query_match = re.search(r'<search>(.*?)</search>', response_text, re.DOTALL)
            query = query_match.group(1).strip() if query_match else ""
            search_results = batch_search(query)
            next_obs = f'\n\n<information>{search_results.strip()}</information>\n\n'
            prompt = prompt + response_text + next_obs
            all_model_outputs.append(next_obs)
            turns.append({"turn": turn + 1, "action": "search", "query": query})
        else:
            turns.append({"turn": turn + 1, "action": "no_valid_action"})
            if outputs[0][-1].item() in QWEN_EOS_IDS:
                break
            prompt = prompt + response_text
            break

    full_output = ''.join(all_model_outputs)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("-n", "--num_samples", type=int, default=8,
                        help="Number of test samples (split evenly between NQ and HotpotQA)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save per-sample results to JSONL file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print full model output for each sample")
    return parser.parse_args()


def main():
    args = parse_args()

    BASE_MODEL = "/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct"
    SFT_DIR = "/ssd1/zz/AI_efficency/RAG/Search-R1/sft_output/sft_1148_rs"
    SFT_CKPT = SFT_DIR + "/checkpoint-108"

    if args.model_path:
        model_path = args.model_path
        tokenizer_path = args.tokenizer_path or args.model_path
        model_label = f"Custom: {model_path}"
    elif args.base:
        model_path = BASE_MODEL
        tokenizer_path = BASE_MODEL
        model_label = "Base: Qwen2.5-7B-Instruct"
    else:
        model_path = SFT_CKPT
        tokenizer_path = SFT_DIR
        model_label = "SFT: sft_1148_rs/checkpoint-108"

    print("=" * 80)
    print(f"Model:       {model_label}")
    print(f"Num samples: {args.num_samples}")
    print("=" * 80)

    device = torch.device("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
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

    print(f"Actual samples: {len(test_samples)} (NQ: {sum(1 for s in test_samples if s[0]=='NQ')}, "
          f"HotpotQA: {sum(1 for s in test_samples if s[0]=='HotpotQA')})")
    print("=" * 80)

    stats = {
        "has_think": 0, "has_search": 0, "has_answer": 0,
        "has_info": 0, "correct": 0, "total": len(test_samples),
        "total_turns": 0, "search_count": 0,
    }
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

        has_think = "<think>" in full_output and "</think>" in full_output
        has_search = "<search>" in full_output and "</search>" in full_output
        has_answer = "<answer>" in full_output and "</answer>" in full_output
        has_info = "<information>" in full_output
        n_searches = sum(1 for t in turns if t['action'] == 'search')

        stats["has_think"] += int(has_think)
        stats["has_search"] += int(has_search)
        stats["has_answer"] += int(has_answer)
        stats["has_info"] += int(has_info)
        stats["correct"] += int(is_correct)
        stats["total_turns"] += len(turns)
        stats["search_count"] += n_searches

        status = "✓" if is_correct else "✗"
        print(f"[{idx+1:3d}/{len(test_samples)}] {dataset:9s} #{sample_idx:<4d} | "
              f"turns={len(turns)} searches={n_searches} | "
              f"think={'Y' if has_think else 'N'} answer={'Y' if has_answer else 'N'} | "
              f"{status} pred=\"{(predicted_answer or '')[:50]}\" | {elapsed:.1f}s")

        if args.verbose:
            print(f"    Q: {question[:80]}")
            print(f"    Gold: {golden}")
            print(f"    Output: {full_output[:200]}...")

        results_log.append({
            "dataset": dataset, "sample_idx": sample_idx,
            "question": question, "golden": golden,
            "predicted": predicted_answer, "correct": is_correct,
            "turns": len(turns), "searches": n_searches,
            "has_think": has_think, "has_search": has_search,
            "has_answer": has_answer, "has_info": has_info,
            "full_output": full_output,
        })

    total_time = time.time() - t0
    total = stats["total"]

    print("\n" + "=" * 80)
    print(f"SUMMARY — {model_label}")
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
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nDetailed results saved to: {args.save}")


if __name__ == "__main__":
    main()
