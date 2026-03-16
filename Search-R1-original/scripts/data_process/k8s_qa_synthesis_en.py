#!/usr/bin/env python3
"""
K8s QA Synthesis Script - Generate training QA pairs from K8s documentation using GPT-5 API
Approach: Few-shot + Method B (generate 2 QA pairs per chunk)
"""
import os
import json
import random
import argparse
import asyncio
import aiohttp
from typing import List, Dict
from collections import defaultdict
from tqdm.asyncio import tqdm as async_tqdm

# API Configuration
API_ENDPOINT = "http://osagw.simeji.me/gbu/rest/v1/ai_chat/openai_service"
API_KEY = "mediago_platform.oijio3f4893u2898"

# Topic to difficulty weight mapping
TOPIC_DIFFICULTY_WEIGHT = {
    "pods": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    "deployments": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    "services": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    "security": {"easy": 0.2, "medium": 0.6, "hard": 0.2},
    "storage": {"easy": 0.2, "medium": 0.6, "hard": 0.2},
    "default": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
}

# System Prompt (English)
SYNTHESIS_PROMPT_SYSTEM = """You are a Kubernetes expert and AI trainer. Generate high-quality Q&A pairs based on official Kubernetes documentation.

Quality Standards:
- Questions should be specific and realistic, simulating real-world scenarios (configuration, troubleshooting, design choices)
- key_points are the core concepts that must be covered in the answer (3-5 points)
- target_answer should be concise and accurate (1-3 sentences)
- Cover different question types: concept, troubleshoot, best-practice, comparison
- Adjust difficulty based on topic: core concepts (Pods/Deployments) should lean toward medium/hard

Few-shot Examples:
【Example 1 - medium difficulty】
Documentation: "Pods are the smallest deployable units of computing in Kubernetes. A Pod contains one or more containers..."
Output:
{
  "qa_pairs": [{
    "question": "When should you run multiple containers in a single Pod instead of deploying them separately?",
    "key_points": ["Tightly coupled containers", "Shared storage and network", "Sidecar pattern", "Not suitable for independent scaling"],
    "target_answer": "Multiple containers should share a Pod when they need tight coordination, shared volumes, or network resources, such as sidecar log collectors or service mesh proxies.",
    "difficulty": "medium",
    "question_type": "best-practice"
  }]
}

【Example 2 - hard difficulty】
Documentation: "StatefulSet provides stable network identity and persistent storage for stateful applications..."
Output:
{
  "qa_pairs": [{
    "question": "Why do database clusters typically use StatefulSet instead of Deployment? What are the technical considerations?",
    "key_points": ["Stable Pod identity (hostname)", "Ordered deployment and scaling", "PVC binding ensures data persistence", "Stable network identity helps replica discovery"],
    "target_answer": "StatefulSet provides stable network identities (e.g., pod-0.svc) and bound PVCs, ensuring database nodes can access their original data after restart, with ordered startup guaranteeing proper cluster initialization.",
    "difficulty": "hard",
    "question_type": "comparison"
  }]
}"""

# User Prompt Template (English)
SYNTHESIS_PROMPT_USER_TEMPLATE = """Generate 2 high-quality Q&A pairs of different types based on the following Kubernetes documentation:

```
{doc_chunk}
```

Topic: {doc_topic}
Document Path: {doc_path}

Requirements:
1. The 2 questions MUST be of different types (choose from: concept, troubleshoot, best-practice, comparison)
2. Difficulty distribution: Core topics (Pods/Deployments/Services) lean medium/hard, specialized topics (Security/Storage) lean medium, others easy/medium
3. Strictly follow JSON format
4. ALL content (questions, answers, key_points) must be in English

Output Format:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "key_points": ["point1", "point2", "point3"],
      "target_answer": "...",
      "difficulty": "easy/medium/hard",
      "question_type": "concept/troubleshoot/best-practice/comparison"
    }},
    {{
      "question": "...",
      "key_points": ["point1", "point2", "point3"],
      "target_answer": "...",
      "difficulty": "easy/medium/hard",
      "question_type": "concept/troubleshoot/best-practice/comparison"
    }}
  ]
}}"""


async def call_gpt5_api(session: aiohttp.ClientSession, chunk_data: Dict, semaphore: asyncio.Semaphore, retry=3) -> Dict:
    """Asynchronously call GPT-5 API to generate QA pairs"""
    async with semaphore:
        user_prompt = SYNTHESIS_PROMPT_USER_TEMPLATE.format(
            doc_chunk=chunk_data['contents'][:2500],  # Limit length
            doc_topic=chunk_data['metadata']['topic_category'],
            doc_path=chunk_data['metadata']['file_path']
        )
        
        payload = {
            "model_name": "deploy_gpt5_chat",
            "context": [
                {"text": SYNTHESIS_PROMPT_SYSTEM, "role_type": "system"},
                {"text": user_prompt, "role_type": "user"}
            ],
            "max_token": 2000,
            "ans_token": 1500,
            "temperature": "0.7",
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
        
        for attempt in range(retry):
            try:
                async with session.post(API_ENDPOINT, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=40)) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Parse API-specific response format
                        if result.get('errno') == 0:
                            content = result.get('data', {}).get('content', '')
                        else:
                            error_msg = result.get('msg', 'Unknown error')
                            if attempt < retry - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return {'success': False, 'chunk_id': chunk_data['id'], 'error': f'API error: {error_msg}'}
                        
                        # Try parsing JSON
                        # Handle possible markdown code block wrapping
                        if '```json' in content:
                            content = content.split('```json')[1].split('```')[0].strip()
                        elif '```' in content:
                            content = content.split('```')[1].split('```')[0].strip()
                        
                        qa_data = json.loads(content)
                        return {
                            'success': True,
                            'chunk_id': chunk_data['id'],
                            'qa_pairs': qa_data.get('qa_pairs', []),
                            'metadata': chunk_data['metadata']
                        }
                    else:
                        error_text = await response.text()
                        if attempt < retry - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return {'success': False, 'chunk_id': chunk_data['id'], 'error': f'HTTP {response.status}: {error_text[:200]}'}
            except asyncio.TimeoutError:
                if attempt < retry - 1:
                    await asyncio.sleep(2)
                    continue
                return {'success': False, 'chunk_id': chunk_data['id'], 'error': 'timeout'}
            except json.JSONDecodeError as e:
                if attempt < retry - 1:
                    await asyncio.sleep(1)
                    continue
                return {'success': False, 'chunk_id': chunk_data['id'], 'error': f'JSON decode error: {str(e)}'}
            except Exception as e:
                if attempt < retry - 1:
                    await asyncio.sleep(2)
                    continue
                return {'success': False, 'chunk_id': chunk_data['id'], 'error': str(e)}
        
        return {'success': False, 'chunk_id': chunk_data['id'], 'error': 'max retries exceeded'}


async def batch_generate_qa(chunks: List[Dict], concurrency: int = 7) -> List[Dict]:
    """Batch async generation of QA pairs"""
    semaphore = asyncio.Semaphore(concurrency)
    
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [call_gpt5_api(session, chunk, semaphore) for chunk in chunks]
        
        print(f"Starting {len(tasks)} API calls with concurrency={concurrency}...")
        results = []
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if result['success']:
                print(f"✓ {result['chunk_id']}: {len(result['qa_pairs'])} QA pairs")
            else:
                print(f"✗ {result['chunk_id']}: {result.get('error', 'unknown error')}\n")
        
        return results


def sample_chunks(corpus_file: str, num_samples: int) -> List[Dict]:
    """Intelligently sample chunks based on topic priority"""
    chunks_by_topic = defaultdict(list)
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            topic = chunk.get('metadata', {}).get('topic_category', 'default')
            chunks_by_topic[topic].append(chunk)
    
    # Priority weights
    topic_weights = {
        'pods': 0.3,
        'deployments': 0.15,
        'services': 0.20,
        'storage': 0.15,
        'security': 0.15,
        'default': 0.05
    }
    
    sampled = []
    for topic, weight in topic_weights.items():
        topic_chunks = chunks_by_topic.get(topic, [])
        if not topic_chunks:
            continue
        n = int(num_samples * weight)
        if n > len(topic_chunks):
            n = len(topic_chunks)
        sampled.extend(random.sample(topic_chunks, n))
    
    # Fill remaining slots
    if len(sampled) < num_samples:
        remaining = []
        for topic, chunks in chunks_by_topic.items():
            remaining.extend([c for c in chunks if c not in sampled])
        random.shuffle(remaining)
        sampled.extend(remaining[:num_samples - len(sampled)])
    
    random.shuffle(sampled)
    return sampled[:num_samples]


def convert_to_searchr1_format(results: List[Dict]) -> List[Dict]:
    """Convert raw QA pairs to Search-R1 training format"""
    searchr1_data = []
    
    for result in results:
        if not result['success']:
            continue
        
        for i, qa in enumerate(result['qa_pairs']):
            entry = {
                "data_source": "k8s_concepts",
                "prompt": [{
                    "role": "user",
                    "content": f"Answer the given question. You must conduct reasoning inside <think>...</think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {qa['question']}\n"
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": qa.get('key_points', [])
                    }
                },
                "extra_info": {
                    "chunk_id": result['chunk_id'],
                    "index": i,
                    "difficulty": qa.get('difficulty', 'medium'),
                    "question_type": qa.get('question_type', 'concept'),
                    "doc_path": result['metadata'].get('file_path', ''),
                    "target_answer": qa.get('target_answer', '')
                }
            }
            searchr1_data.append(entry)
    
    return searchr1_data


def main():
    parser = argparse.ArgumentParser(description='K8s QA Synthesis with GPT-5')
    parser.add_argument('--corpus_file', type=str, required=True, help='Path to corpus JSONL file with metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for generated QA pairs')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of chunks to sample')
    parser.add_argument('--concurrency', type=int, default=7, help='Number of concurrent API requests')
    parser.add_argument('--api_key', type=str, default=API_KEY, help='GPT-5 API key')
    
    args = parser.parse_args()
    
    print("="*70)
    print("K8s QA Synthesis - GPT-5 API (English)")
    print("="*70)
    print(f"Corpus: {args.corpus_file}")
    print(f"Target: {args.num_samples} chunks → ~{args.num_samples * 2} QA pairs")
    print(f"Concurrency: {args.concurrency} requests")
    print(f"API Endpoint: {API_ENDPOINT}")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Sample
    print("\n[1/4] Sampling chunks...")
    sampled_chunks = sample_chunks(args.corpus_file, args.num_samples)
    print(f"  ✓ Sampled {len(sampled_chunks)} chunks")
    
    # Statistics
    topic_dist = defaultdict(int)
    for chunk in sampled_chunks:
        topic = chunk.get('metadata', {}).get('topic_category', 'unknown')
        topic_dist[topic] += 1
    print(f"  Topic distribution: {dict(topic_dist)}")
    
    # Step 2: Generate
    print(f"\n[2/4] Generating QA pairs via GPT-5 API...")
    results = asyncio.run(batch_generate_qa(sampled_chunks, args.concurrency))
    
    # Step 3: Process results
    print(f"\n[3/4] Processing results...")
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"  Success: {len(successful)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")
    
    if failed:
        print(f"\n  Failure details:")
        for fail in failed[:5]:  # Show first 5
            print(f"    - {fail['chunk_id']}: {fail.get('error', 'unknown')}\n")
    
    # Save raw results
    raw_output = os.path.join(args.output_dir, f"raw_qa_{args.num_samples*2}.jsonl")
    with open(raw_output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"  ✓ Raw results: {raw_output}")
    
    # Step 4: Convert to Search-R1 format
    print(f"\n[4/4] Converting to Search-R1 format...")
    searchr1_data = convert_to_searchr1_format(results)
    
    searchr1_output = os.path.join(args.output_dir, f"qa_{args.num_samples*2}_searchr1_format.json")
    with open(searchr1_output, 'w', encoding='utf-8') as f:
        json.dump(searchr1_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Search-R1 format: {searchr1_output}")
    
    # Statistics report
    print("\n" + "="*70)
    print("Statistics Report")
    print("="*70)
    print(f"Processed chunks: {len(results)}")
    print(f"Successfully generated: {len(successful)}")
    
    total_qa = sum(len(r['qa_pairs']) for r in successful)
    print(f"Total QA pairs: {total_qa}")
    
    # Difficulty distribution
    print(f"\nDifficulty distribution:")
    diff_dist = defaultdict(int)
    for r in successful:
        for qa in r['qa_pairs']:
            diff_dist[qa.get('difficulty', 'unknown')] += 1
    
    for diff in ['easy', 'medium', 'hard']:
        count = diff_dist[diff]
        pct = (count / total_qa * 100) if total_qa > 0 else 0
        print(f"  {diff:10s}: {count:3d} ({pct:5.1f}%)")
    
    # Question type distribution
    print(f"\nQuestion type distribution:")
    type_dist = defaultdict(int)
    for r in successful:
        for qa in r['qa_pairs']:
            type_dist[qa.get('question_type', 'unknown')] += 1
    
    for qtype in ['concept', 'troubleshoot', 'best-practice', 'comparison']:
        count = type_dist[qtype]
        pct = (count / total_qa * 100) if total_qa > 0 else 0
        print(f"  {qtype:15s}: {count:3d} ({pct:5.1f}%)")
    
    print("="*70)
    print("\n✅ Complete!")
    print(f"📁 Output files:")
    print(f"   Raw: {raw_output}")
    print(f"   Search-R1: {searchr1_output}")
    print(f"\n💡 Next step: Review 10-15 samples to verify quality before scaling up\n")


if __name__ == "__main__":
    main()
