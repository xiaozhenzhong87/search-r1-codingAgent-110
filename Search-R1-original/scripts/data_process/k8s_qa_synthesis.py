#!/usr/bin/env python3
"""
K8s QA Synthesis Script - 使用GPT-5 API基于K8s文档生成训练QA对
方案: Few-shot + 方案B (一次生成2个QA对)
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

# API配置
API_ENDPOINT = "http://osagw.simeji.me/gbu/rest/v1/ai_chat/openai_service"
API_KEY = "mediago_platform.oijio3f4893u2898"

# 主题到难度权重映射
TOPIC_DIFFICULTY_WEIGHT = {
    "pods": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    "deployments": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    "services": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    "security": {"easy": 0.2, "medium": 0.6, "hard": 0.2},
    "storage": {"easy": 0.2, "medium": 0.6, "hard": 0.2},
    "default": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
}

# System Prompt
SYNTHESIS_PROMPT_SYSTEM = """你是Kubernetes技术专家和AI训练师。基于K8s官方文档生成高质量问答对。

质量标准:
- 问题真实具体,模拟实际工作场景(如配置、排错、选型)
- key_points是回答必须覆盖的核心要点(3-5个)
- target_answer简洁准确(1-3句话)
- 涵盖不同类型:concept(概念)、troubleshoot(排错)、best-practice(最佳实践)、comparison(对比)
- 根据文档主题自动调整难度:核心概念(Pods/Deployments)适当提高难度

Few-shot示例:
【示例1 - medium难度】
文档: "Pod是K8s中最小的可部署单元,包含一个或多个容器..."
输出:
{
  "qa_pairs": [{
    "question": "在什么场景下应该在一个Pod中运行多个容器而不是分开部署?",
    "key_points": ["紧密耦合的容器", "共享存储和网络", "sidecar模式", "不适合独立扩展的场景"],
    "target_answer": "当多个容器需要紧密协作、共享存储卷或网络资源时,如sidecar日志收集器、服务网格代理。",
    "difficulty": "medium",
    "question_type": "best-practice"
  }]
}

【示例2 - hard难度】
文档: "StatefulSet为有状态应用提供稳定的网络标识和持久化存储..."
输出:
{
  "qa_pairs": [{
    "question": "为什么数据库集群通常使用StatefulSet而不是Deployment?涉及哪些技术考量?",
    "key_points": ["稳定的Pod标识(hostname)", "有序部署和扩展", "PVC绑定保证数据持久化", "网络标识不变利于副本发现"],
    "target_answer": "StatefulSet提供稳定的网络标识(如pod-0.svc)和绑定的PVC,确保数据库节点重启后仍能访问原有数据,且有序启动保证集群初始化顺序。",
    "difficulty": "hard",
    "question_type": "comparison"
  }]
}"""

# User Prompt模板
SYNTHESIS_PROMPT_USER_TEMPLATE = """基于以下K8s文档片段生成2个不同类型的高质量问答对:

```
{doc_chunk}
```

文档主题: {doc_topic}
文档路径: {doc_path}

要求:
1. 2个问题类型必须不同(从concept/troubleshoot/best-practice/comparison中选择)
2. 难度分布参考: 核心主题(Pods/Deployments/Services)偏medium/hard, 专业主题(Security/Storage)偏medium, 其他easy/medium
3. 严格遵循JSON格式输出

输出格式:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "key_points": ["要点1", "要点2", "要点3"],
      "target_answer": "...",
      "difficulty": "easy/medium/hard",
      "question_type": "concept/troubleshoot/best-practice/comparison"
    }},
    {{
      "question": "...",
      "key_points": ["要点1", "要点2"],
      "target_answer": "...",
      "difficulty": "easy/medium/hard",
      "question_type": "concept/troubleshoot/best-practice/comparison"
    }}
  ]
}}"""


async def call_gpt5_api(session: aiohttp.ClientSession, chunk_data: Dict, semaphore: asyncio.Semaphore, retry=3) -> Dict:
    """异步调用GPT-5 API生成QA对"""
    async with semaphore:
        user_prompt = SYNTHESIS_PROMPT_USER_TEMPLATE.format(
            doc_chunk=chunk_data['contents'][:2500],  # 限制长度
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
                        # 解析这个API特有的响应格式
                        if result.get('errno') == 0:
                            content = result.get('data', {}).get('content', '')
                        else:
                            error_msg = result.get('msg', 'Unknown error')
                            if attempt < retry - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return {'success': False, 'chunk_id': chunk_data['id'], 'error': f'API error: {error_msg}'}
                        
                        # 尝试解析JSON
                        # 处理可能的markdown代码块包裹
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
                            await asyncio.sleep(2 ** attempt)  # 指数退避
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
    """批量异步生成QA对"""
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
                print(f"✗ {result['chunk_id']}: {result.get('error', 'unknown error')}")
        
        return results


def sample_chunks(corpus_file: str, num_samples: int) -> List[Dict]:
    """智能采样chunks"""
    with open(corpus_file, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    # 按主题分组
    topic_groups = defaultdict(list)
    for chunk in all_chunks:
        topic = chunk.get('metadata', {}).get('topic_category', 'default')
        topic_groups[topic].append(chunk)
    
    # 优先采样 - Pods/Services/Storage优先
    priority = {
        'pods': 0.30,
        'services': 0.20,
        'storage': 0.15,
        'security': 0.15,
        'deployments': 0.10,
        'default': 0.10
    }
    
    sampled = []
    for topic, ratio in priority.items():
        n = int(num_samples * ratio)
        if topic in topic_groups and len(topic_groups[topic]) > 0:
            available = topic_groups[topic]
            sampled.extend(random.sample(available, min(n, len(available))))
    
    # 补充到目标数量
    if len(sampled) < num_samples:
        remaining = [c for c in all_chunks if c not in sampled]
        if remaining:
            sampled.extend(random.sample(remaining, min(num_samples - len(sampled), len(remaining))))
    
    return sampled[:num_samples]


def convert_to_searchr1_format(qa_pair: Dict, chunk_id: str, metadata: Dict, idx: int) -> Dict:
    """转换为Search-R1训练格式"""
    if not qa_pair:
        return None
    
    question = qa_pair.get('question', '')
    key_points = qa_pair.get('key_points', [])
    
    if not question or not key_points:
        return None
    
    # Search-R1标准格式
    return {
        "data_source": "k8s_concepts",
        "prompt": [{
            "role": "user",
            "content": f"Answer the given question. You must conduct reasoning inside <think>...</think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": key_points}
        },
        "extra_info": {
            "chunk_id": chunk_id,
            "index": idx,
            "difficulty": qa_pair.get('difficulty', 'medium'),
            "question_type": qa_pair.get('question_type', 'concept'),
            "doc_path": metadata.get('file_path', ''),
            "target_answer": qa_pair.get('target_answer', '')
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', default='/ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus_with_metadata.jsonl')
    parser.add_argument('--output_dir', default='/ssd1/zz/AI_efficency/RAG/data/k8s_search')
    parser.add_argument('--num_samples', type=int, default=50, help='采样chunks数量(生成~100个QA对)')
    parser.add_argument('--concurrency', type=int, default=7, help='并发API请求数')
    parser.add_argument('--api_key', type=str, help='GPT-5 API Key (或设置GPT5_API_KEY环境变量)')
    
    args = parser.parse_args()
    
    # 设置API KEY
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    print("="*70)
    print("K8s QA Synthesis - GPT-5 API")
    print("="*70)
    print(f"Corpus: {args.corpus_file}")
    print(f"目标: {args.num_samples} chunks → 约{args.num_samples*2} QA pairs")
    print(f"并发: {args.concurrency} 请求")
    print(f"API Endpoint: {API_ENDPOINT}")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: 采样
    print("\n[1/4] 采样chunks...")
    sampled_chunks = sample_chunks(args.corpus_file, args.num_samples)
    print(f"  ✓ 已采样 {len(sampled_chunks)} chunks")
    
    # 统计
    topic_dist = defaultdict(int)
    for chunk in sampled_chunks:
        topic = chunk.get('metadata', {}).get('topic_category', 'unknown')
        topic_dist[topic] += 1
    print(f"  主题分布: {dict(topic_dist)}")
    
    # Step 2: 生成
    print(f"\n[2/4] 通过GPT-5 API生成QA对...")
    results = asyncio.run(batch_generate_qa(sampled_chunks, args.concurrency))
    
    # Step 3: 统计
    print(f"\n[3/4] 处理结果...")
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"  成功: {len(successful)}/{len(results)}")
    print(f"  失败: {len(failed)}/{len(results)}")
    
    if failed:
        print("\n  失败详情:")
        for f in failed[:5]:
            print(f"    - {f['chunk_id']}: {f.get('error', 'unknown')}")
    
    # 保存原始结果
    raw_output = os.path.join(args.output_dir, 'raw_qa_100.jsonl')
    with open(raw_output, 'w', encoding='utf-8') as f:
        for result in successful:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"  ✓ 原始结果: {raw_output}")
    
    # Step 4: 转换格式
    print(f"\n[4/4] 转换为Search-R1格式...")
    searchr1_data = []
    for result in successful:
        for qa_pair in result.get('qa_pairs', []):
            converted = convert_to_searchr1_format(
                qa_pair, 
                result['chunk_id'], 
                result.get('metadata', {}),
                len(searchr1_data)
            )
            if converted:
                searchr1_data.append(converted)
    
    searchr1_output = os.path.join(args.output_dir, 'qa_100_searchr1_format.json')
    with open(searchr1_output, 'w', encoding='utf-8') as f:
        json.dump(searchr1_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Search-R1格式: {searchr1_output}")
    
    # 统计报告
    print("\n" + "="*70)
    print("统计报告")
    print("="*70)
    print(f"处理chunks: {len(results)}")
    print(f"成功生成: {len(successful)}")
    print(f"QA对总数: {len(searchr1_data)}")
    
    # 难度分布
    difficulty_dist = defaultdict(int)
    type_dist = defaultdict(int)
    for data in searchr1_data:
        difficulty_dist[data['extra_info']['difficulty']] += 1
        type_dist[data['extra_info']['question_type']] += 1
    
    total = len(searchr1_data)
    print(f"\n难度分布:")
    for diff in ['easy', 'medium', 'hard']:
        count = difficulty_dist.get(diff, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {diff:10s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n问题类型分布:")
    for qtype in ['concept', 'troubleshoot', 'best-practice', 'comparison']:
        count = type_dist.get(qtype, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {qtype:15s}: {count:3d} ({pct:5.1f}%)")
    
    print("="*70)
    print(f"\n✅ 完成!")
    print(f"📁 输出文件:")
    print(f"   Raw: {raw_output}")
    print(f"   Search-R1: {searchr1_output}")
    print("\n💡 下一步: 人工审核10-15个样本,确认质量后决定是否继续生成")


if __name__ == '__main__':
    main()
