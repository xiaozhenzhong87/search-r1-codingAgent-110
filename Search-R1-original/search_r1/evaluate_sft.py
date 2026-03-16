"""
评估SFT模型，识别答错的题目
"""
import torch
import transformers
import pandas as pd
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
import re
import requests
from verl.utils.reward_score.qa_em import compute_score_em, extract_solution


def extract_answer_from_response(response_str: str) -> str:
    """从响应中提取答案"""
    answer = extract_solution(response_str)
    return answer if answer else ""


def search_query(query: str, search_url: str, topk: int = 3) -> str:
    """执行搜索"""
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True
    }
    results = requests.post(search_url, json=payload).json()['result']
    
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference
    
    return _passages2string(results[0])


def generate_with_search(
    model,
    tokenizer,
    prompt: str,
    search_url: str,
    topk: int = 3,
    max_turns: int = 4,
    max_new_tokens: int = 500,
    temperature: float = 0.7
) -> str:
    """
    使用搜索功能生成答案（模拟infer.py的逻辑）
    """
    curr_eos = [151645, 151643]  # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    # 应用chat template
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
    
    full_response = prompt
    device = next(model.parameters()).device
    
    # 定义停止条件
    class StopOnSequence(transformers.StoppingCriteria):
        def __init__(self, target_sequences, tokenizer):
            self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) 
                             for target_sequence in target_sequences]
            self.target_lengths = [len(target_id) for target_id in self.target_ids]
            self._tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            targets = [torch.as_tensor(target_id, device=input_ids.device) 
                      for target_id in self.target_ids]
            if input_ids.shape[1] < min(self.target_lengths):
                return False
            for i, target in enumerate(targets):
                if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                    return True
            return False
    
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", 
                        "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([
        StopOnSequence(target_sequences, tokenizer)
    ])
    
    def get_query(text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None
    
    # 多轮生成循环
    for turn in range(max_turns):
        input_ids = tokenizer.encode(full_response, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature
        )
        
        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_response += output_text
            break
        
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            search_results = search_query(tmp_query, search_url, topk)
        else:
            search_results = ''
        
        search_text = curr_search_template.format(
            output_text=output_text,
            search_results=search_results
        )
        full_response += search_text
    
    return full_response


def evaluate_sft_model(
    model_path: str,
    test_data_file: str,
    output_file: str,
    search_url: str = "http://127.0.0.1:8000/retrieve",
    topk: int = 3,
    batch_size: int = 1
):
    """
    评估SFT模型，识别答错的题目
    
    Args:
        model_path: SFT模型路径
        test_data_file: 测试数据文件路径（parquet格式）
        output_file: 输出文件路径（jsonl格式，包含答错的题目）
        search_url: 搜索服务URL
        topk: 检索topk结果
        batch_size: 批处理大小（目前只支持1）
    """
    print(f"加载模型: {model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"加载测试数据: {test_data_file}")
    df = pd.read_parquet(test_data_file)
    print(f"测试样本数: {len(df)}")
    
    wrong_samples = []
    correct_count = 0
    total_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估中"):
        question_data = row.to_dict()
        prompt = question_data['prompt']
        ground_truth = question_data['reward_model']['ground_truth']
        
        # 准备prompt
        if isinstance(prompt, list):
            prompt_text = prompt[0]['content']
        else:
            prompt_text = prompt
        
        try:
            # 生成响应
            response_str = generate_with_search(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                search_url=search_url,
                topk=topk,
                max_turns=4
            )
            
            # 计算reward
            reward = compute_score_em(
                solution_str=response_str,
                ground_truth=ground_truth,
                format_score=0.0,
                score=1.0
            )
            
            total_count += 1
            if reward == 1.0:
                correct_count += 1
            else:
            # 答错的题目，保存用于RL训练
            # 确保ground_truth格式正确
            gt_target = ground_truth.get('target', [])
            if hasattr(gt_target, 'tolist'):
                gt_list = gt_target.tolist()
            elif isinstance(gt_target, (list, tuple)):
                gt_list = list(gt_target)
            else:
                gt_list = [gt_target]
            
            wrong_sample = {
                'id': question_data.get('id', f'test_{idx}'),
                'question': question_data.get('question', ''),
                'prompt': prompt_text,
                'response': response_str,
                'answer': extract_answer_from_response(response_str),
                'ground_truth': gt_list,  # 保存为列表格式，prepare_rl_data.py会转换为numpy数组
                'reward': float(reward),
                'data_source': question_data.get('data_source', 'nq'),
                'reward_model': question_data.get('reward_model', {
                    'ground_truth': {'target': gt_list},
                    'style': 'rule'
                }),
                'ability': question_data.get('ability', 'fact-reasoning'),
                'extra_info': question_data.get('extra_info', {'split': 'test'})
            }
                wrong_samples.append(wrong_sample)
        
        except Exception as e:
            print(f"处理问题 {idx} 时出错: {e}")
            continue
    
    # 保存答错的题目
    print(f"\n评估结果:")
    print(f"  总样本数: {total_count}")
    print(f"  正确数: {correct_count}")
    print(f"  错误数: {len(wrong_samples)}")
    print(f"  准确率: {correct_count / total_count if total_count > 0 else 0:.4f}")
    
    if wrong_samples:
        print(f"\n保存 {len(wrong_samples)} 个答错的题目到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in wrong_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        print("\n所有题目都答对了！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估SFT模型，识别答错的题目")
    parser.add_argument("--model_path", type=str, required=True,
                       help="SFT模型路径")
    parser.add_argument("--test_data_file", type=str, required=True,
                       help="测试数据文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                       help="输出文件路径（答错的题目）")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve",
                       help="搜索服务URL")
    parser.add_argument("--topk", type=int, default=3,
                       help="检索topk结果")
    
    args = parser.parse_args()
    
    evaluate_sft_model(
        model_path=args.model_path,
        test_data_file=args.test_data_file,
        output_file=args.output_file,
        search_url=args.search_url,
        topk=args.topk
    )
