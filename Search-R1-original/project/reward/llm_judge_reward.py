#!/usr/bin/env python3
"""
LLM-as-Judge Reward Function for K8s RAG
Uses GPT-5 to evaluate answer quality based on target_answer
"""
import re
import json
import random
import aiohttp
import asyncio
from typing import Dict, List, Optional

# GPT-5 API Configuration
API_ENDPOINT = "http://osagw.simeji.me/gbu/rest/v1/ai_chat/openai_service"
API_KEY = "mediago_platform.oijio3f4893u2898"

# Judge Prompt Template
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for Kubernetes question-answering systems. Your task is to evaluate the quality of generated answers compared to reference answers.

Evaluation Criteria:
1. **Correctness** (0-4 points): Does the answer contain accurate information matching the reference?
2. **Completeness** (0-3 points): Does it cover the key points from the reference?
3. **Clarity** (0-2 points): Is the answer clear and well-structured?
4. **Conciseness** (0-1 point): Is it appropriately concise without unnecessary verbosity?

Total Score: 0-10 points (will be normalized to 0-1)

Output Format:
{
  "correctness": <0-4>,
  "completeness": <0-3>,
  "clarity": <0-2>,
  "conciseness": <0-1>,
  "total_score": <0-10>,
  "reasoning": "<brief explanation>"
}"""

JUDGE_USER_TEMPLATE = """Question: {question}

Reference Answer: {reference_answer}

Generated Answer: {generated_answer}

Evaluate the generated answer and provide scores."""


def extract_answer_from_response(response_text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags"""
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, response_text, re.DOTALL))
    
    # Skip first 2 matches (from prompt examples)
    if len(matches) <= 2:
        return None
    
    return matches[-1].group(1).strip()


async def call_judge_api(question: str, reference: str, generated: str, 
                         session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Dict:
    """Call GPT-5 API for judgment"""
    async with semaphore:
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference,
            generated_answer=generated
        )
        
        payload = {
            "model_name": "deploy_gpt5_chat",
            "context": [
                {"text": JUDGE_SYSTEM_PROMPT, "role_type": "system"},
                {"text": user_prompt, "role_type": "user"}
            ],
            "max_token": 1000,
            "ans_token": 500,
            "temperature": "0.3",  # Lower temperature for more consistent judgment
            "top_p": "1.0",
            "frequency_penalty": "0.0",
            "presence_penalty": "0.0",
            "api_version": "2023-05-15"
        }
        
        headers = {
            "Content-Type": "application/json",
            "apikey": API_KEY,
            "User-Agent": "iAPI/1.0.0 (http://iapi.baidu-int.com)",
            "Accept": "*/*",
            "Host": "gbu.jp02-a30-apisix-sandbox.baidu-int.com",
            "Connection": "keep-alive"
        }
        
        for attempt in range(3):
            try:
                async with session.post(API_ENDPOINT, json=payload, headers=headers, 
                                       timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('errno') == 0:
                            content = result.get('data', {}).get('content', '')
                            
                            # Parse JSON from response
                            if '```json' in content:
                                content = content.split('```json')[1].split('```')[0].strip()
                            elif '```' in content:
                                content = content.split('```')[1].split('```')[0].strip()
                            
                            judgment = json.loads(content)
                            return {
                                'success': True,
                                'judgment': judgment,
                                'normalized_score': judgment['total_score'] / 10.0  # Normalize to [0, 1]
                            }
                        else:
                            if attempt < 2:
                                await asyncio.sleep(1)
                                continue
                            return {'success': False, 'error': result.get('msg', 'API error')}
                    else:
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        return {'success': False, 'error': f'HTTP {response.status}'}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Max retries exceeded'}


def compute_score_llm_judge(solution_str: str, ground_truth: Dict, 
                            format_score: float = 0.0, score: float = 1.0,
                            penalty_ratio: float = 0.2, debug: bool = False) -> float:
    """
    LLM-as-Judge reward function (synchronous wrapper for training)
    
    Args:
        solution_str: Model's generated response
        ground_truth: Dict with 'target' (unused) and extra_info containing 'target_answer'
        format_score: Score for format (default 0)
        score: Max score (default 1.0)
        penalty_ratio: Penalty for extra output after answer (default 0.2)
        debug: Enable debug printing
    
    Returns:
        float: Reward score [0, 1]
    """
    # Extract answer from model output
    answer = extract_answer_from_response(solution_str)
    
    if answer is None:
        # No valid answer format
        return format_score
    
    # Get reference answer from extra_info
    extra_info = ground_truth.get('extra_info', {})
    reference_answer = extra_info.get('target_answer', '')
    question = extra_info.get('question', 'N/A')
    
    if not reference_answer:
        # Fallback: use simple exact match if no reference
        if debug:
            print(f"Warning: No target_answer found, using fallback scoring")
        return 0.5  # Neutral score
    
    # Check for extra output after </answer>
    has_continuation = has_continuation_after_answer(solution_str)
    
    # Call LLM judge (synchronous)
    try:
        judgment = asyncio.run(_async_judge_wrapper(question, reference_answer, answer))
        
        if judgment['success']:
            base_score = judgment['normalized_score'] * score
            
            # Apply penalty for extra output
            if has_continuation:
                final_score = base_score * (1 - penalty_ratio)
            else:
                final_score = base_score
            
            if debug:
                print(f"Question: {question[:100]}...")
                print(f"Reference: {reference_answer[:100]}...")
                print(f"Generated: {answer[:100]}...")
                print(f"Judgment: {judgment['judgment']}")
                print(f"Base Score: {base_score:.3f}, Final Score: {final_score:.3f}")
            
            return final_score
        else:
            # Judge API failed, use fallback
            if debug:
                print(f"Judge API failed: {judgment.get('error')}, using fallback")
            return 0.5
            
    except Exception as e:
        if debug:
            print(f"Exception in LLM judge: {e}")
        return 0.5  # Fallback score on error


async def _async_judge_wrapper(question: str, reference: str, generated: str) -> Dict:
    """Async wrapper for single judgment"""
    semaphore = asyncio.Semaphore(1)
    async with aiohttp.ClientSession() as session:
        return await call_judge_api(question, reference, generated, session, semaphore)


def has_continuation_after_answer(solution_str: str) -> bool:
    """Check if there's extra output after </answer>"""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if len(matches) <= 2:
        return False
    
    last_answer_end = matches[-1].end()
    after_answer = solution_str[last_answer_end:].strip()
    
    # Check for extra tags
    has_extra_tags = bool(re.search(r'<(search|think|information)>', after_answer))
    
    return has_extra_tags


# Batch version for evaluation
async def batch_judge(questions: List[str], references: List[str], 
                     generated: List[str], concurrency: int = 5) -> List[Dict]:
    """Batch judgment with concurrency control"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_judge_api(q, r, g, session, semaphore)
            for q, r, g in zip(questions, references, generated)
        ]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    # Test the judge
    test_question = "What is a Pod in Kubernetes?"
    test_reference = "A Pod is the smallest deployable unit in Kubernetes that contains one or more containers sharing storage and network."
    test_generated = "A Pod is a basic unit in Kubernetes that runs containers."
    
    print("Testing LLM Judge...")
    result = asyncio.run(_async_judge_wrapper(test_question, test_reference, test_generated))
    print(f"Result: {json.dumps(result, indent=2)}")
