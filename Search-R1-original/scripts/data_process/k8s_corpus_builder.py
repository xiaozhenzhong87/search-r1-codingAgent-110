#!/usr/bin/env python3
"""
K8s Corpus Builder - 清洗Kubernetes文档并生成corpus.jsonl

功能:
1. 遍历K8s concepts目录下的所有markdown文件
2. 移除Hugo Front Matter和shortcodes
3. 按标题智能分块(500-800 tokens/chunk)
4. 输出Search-R1所需的corpus.jsonl格式
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def clean_hugo_frontmatter(content: str) -> str:
    """移除Hugo Front Matter (YAML配置块)"""
    # 匹配开头的 ---\n...\n---
    pattern = r'^---\n.*?\n---\n'
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL)
    return cleaned


def clean_hugo_shortcodes(content: str) -> str:
    """移除Hugo shortcodes"""
    # 移除 {{< ... >}} 和 {{% ... %}}
    content = re.sub(r'\{\{[<%]\s*.*?\s*[>%]\}\}', '', content, flags=re.DOTALL)
    # 移除 HTML 注释
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    return content


def extract_title_from_file(filepath: str) -> str:
    """从文件路径提取主题名称"""
    # 例如: concepts/workloads/pods/_index.md -> "Pods"
    parts = Path(filepath).parts
    if '_index.md' in filepath and len(parts) >= 2:
        # 使用父目录名
        return parts[-2].replace('-', ' ').title()
    elif len(parts) > 0:
        # 使用文件名
        return Path(filepath).stem.replace('-', ' ').title()
    else:
        return "Unknown"


def extract_topic_category(filepath: str) -> str:
    """提取文档的主题类别 (用于难度权重)"""
    filepath_lower = filepath.lower()
    if 'pod' in filepath_lower and 'workload' in filepath_lower:
        return 'pods'
    elif 'deployment' in filepath_lower:
        return 'deployments'
    elif 'service' in filepath_lower and 'networking' in filepath_lower:
        return 'services'
    elif 'security' in filepath_lower:
        return 'security'
    elif 'storage' in filepath_lower:
        return 'storage'
    else:
        return 'default'


def split_by_headers(content: str, max_tokens: int = 700) -> List[Tuple[str, str]]:
    """
    按markdown标题智能分块
    返回: [(section_title, section_content), ...]
    """
    # 分割成行
    lines = content.split('\n')
    
    chunks = []
    current_title = ""
    current_content = []
    current_length = 0
    
    for line in lines:
        # 检测markdown标题 (## 或 ###)
        header_match = re.match(r'^(#{2,3})\s+(.+)$', line)
        
        if header_match:
            # 遇到新标题,保存之前的chunk
            if current_content:
                chunk_text = '\n'.join(current_content).strip()
                if chunk_text and len(chunk_text) > 100:  # 至少100字符
                    chunks.append((current_title, chunk_text))
            
            # 开始新chunk
            current_title = header_match.group(2).strip()
            current_content = [line]
            current_length = len(line)
        else:
            current_content.append(line)
            current_length += len(line)
            
            # 如果超过max_tokens对应的字符数(约4倍),强制分块
            if current_length > max_tokens * 4:
                chunk_text = '\n'.join(current_content).strip()
                if chunk_text and len(chunk_text) > 100:
                    chunks.append((current_title, chunk_text))
                current_content = []
                current_length = 0
    
    # 保存最后一个chunk
    if current_content:
        chunk_text = '\n'.join(current_content).strip()
        if chunk_text and len(chunk_text) > 100:
            chunks.append((current_title, chunk_text))
    
    return chunks


def process_markdown_file(filepath: str, base_dir: str) -> List[Dict]:
    """处理单个markdown文件,返回chunks列表"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        return []
    
    # 清洗内容
    content = clean_hugo_frontmatter(content)
    content = clean_hugo_shortcodes(content)
    
    # 提取元信息
    relative_path = os.path.relpath(filepath, base_dir)
    file_title = extract_title_from_file(relative_path)
    topic_category = extract_topic_category(relative_path)
    
    # 分块
    chunks = split_by_headers(content)
    
    # 构建corpus entries
    corpus_entries = []
    for idx, (section_title, section_content) in enumerate(chunks):
        # 生成唯一ID
        chunk_id = relative_path.replace('/', '_').replace('.md', f'_chunk{idx:03d}')
        
        # 构建contents字段: "标题"\n内容
        if section_title:
            contents = f'"{section_title}"\n{section_content}'
        else:
            contents = f'"{file_title}"\n{section_content}'
        
        corpus_entries.append({
            'id': chunk_id,
            'contents': contents,
            'metadata': {
                'file_path': relative_path,
                'file_title': file_title,
                'section_title': section_title,
                'topic_category': topic_category,
                'chunk_index': idx
            }
        })
    
    return corpus_entries


def build_corpus(concepts_dir: str, output_file: str):
    """遍历concepts目录,构建完整corpus"""
    print(f"Processing K8s concepts directory: {concepts_dir}")
    
    all_entries = []
    stats = defaultdict(int)
    
    # 遍历所有.md文件
    for root, dirs, files in os.walk(concepts_dir):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                entries = process_markdown_file(filepath, concepts_dir)
                
                if entries:
                    all_entries.extend(entries)
                    stats['files_processed'] += 1
                    stats['chunks_created'] += len(entries)
                    
                    # 统计主题分布
                    topic = entries[0]['metadata']['topic_category']
                    stats[f'topic_{topic}'] += len(entries)
    
    # 保存corpus.jsonl
    print(f"\nWriting corpus to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            # 只保存id和contents字段(Search-R1格式)
            corpus_item = {
                'id': entry['id'],
                'contents': entry['contents']
            }
            f.write(json.dumps(corpus_item, ensure_ascii=False) + '\n')
    
    # 保存带metadata的完整版本(用于后续QA生成)
    metadata_file = output_file.replace('.jsonl', '_with_metadata.jsonl')
    print(f"Writing corpus with metadata to: {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Corpus Building Statistics:")
    print("="*60)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total chunks created: {stats['chunks_created']}")
    print(f"Average chunks per file: {stats['chunks_created'] / stats['files_processed']:.1f}")
    print("\nTopic Distribution:")
    for key in sorted(stats.keys()):
        if key.startswith('topic_'):
            topic_name = key.replace('topic_', '')
            print(f"  {topic_name:15s}: {stats[key]:4d} chunks")
    print("="*60)
    
    return all_entries


def main():
    parser = argparse.ArgumentParser(description='Build K8s corpus from concepts directory')
    parser.add_argument(
        '--concepts_dir',
        default='/ssd1/zz/AI_efficency/RAG/data/concepts',
        help='Path to K8s concepts directory'
    )
    parser.add_argument(
        '--output_file',
        default='/ssd1/zz/AI_efficency/RAG/data/k8s-concepts-corpus.jsonl',
        help='Output corpus file path'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.concepts_dir):
        print(f"Error: Concepts directory not found: {args.concepts_dir}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 构建corpus
    build_corpus(args.concepts_dir, args.output_file)
    
    print(f"\n✅ Corpus building completed successfully!")
    print(f"📁 Output file: {args.output_file}")


if __name__ == '__main__':
    main()
