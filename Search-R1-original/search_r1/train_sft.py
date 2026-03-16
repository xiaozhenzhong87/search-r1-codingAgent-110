"""
SFT训练脚本：使用构建的数据集进行监督微调
支持两种 prompt 格式:
  1. 纯字符串 (str)
  2. chat 格式 (list/ndarray of dicts with 'content'+'role')

会自动 mask 掉 prompt 和 <information>...</information> 块，
使模型只学习自身生成的部分（think/search/answer）。
"""
import re
import os
import torch
import numpy as np
import transformers
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
from datasets import Dataset
import pandas as pd
import argparse
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class SFTDataCollator:
    """Pad batch and preserve custom labels (with -100 masks)."""
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        batch_input_ids, batch_labels, batch_attention = [], [], []

        for f in features:
            ids = f["input_ids"]
            labels = f["labels"]
            attn = f.get("attention_mask", [1] * len(ids))
            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)
            batch_attention.append(attn + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
        }


def extract_prompt_text(prompt):
    """从 prompt 字段提取纯文本，兼容多种存储格式。"""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, np.ndarray)):
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            return prompt[0].get("content", str(prompt[0]))
        if len(prompt) > 0:
            return str(prompt[0])
    return str(prompt)


def extract_chat_messages(prompt) -> List[Dict[str, str]]:
    """从 prompt 字段提取 chat messages 列表。"""
    if isinstance(prompt, (list, np.ndarray)):
        msgs = list(prompt)
        if msgs and isinstance(msgs[0], dict):
            return msgs
    text = extract_prompt_text(prompt)
    return [{"role": "user", "content": text}]


def find_info_char_spans(text: str) -> List[Tuple[int, int]]:
    """找到文本中所有 <information>...</information> 的字符区间 [start, end)。"""
    return [(m.start(), m.end())
            for m in re.finditer(r'<information>.*?</information>', text, re.DOTALL)]


def build_mask_from_char_spans(
    offsets: List[Tuple[int, int]],
    mask_char_spans: List[Tuple[int, int]],
) -> List[bool]:
    """根据 tokenizer offset_mapping 和需要 mask 的字符区间，
    返回每个 token 是否应被 mask（True=mask, False=keep）。
    """
    token_mask = [False] * len(offsets)
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end == 0:
            continue
        for span_start, span_end in mask_char_spans:
            if tok_start < span_end and tok_end > span_start:
                token_mask[i] = True
                break
    return token_mask


def prepare_sft_data(data_file: str, tokenizer, max_length: int = 4096):
    """
    准备SFT训练数据。
    使用 chat template 格式化输入；
    自动 mask prompt 和 <information>...</information> 块（不参与 loss 计算）。
    """
    df = pd.read_parquet(data_file)
    eos_token = tokenizer.eos_token or ""

    def tokenize_function(examples):
        all_input_ids = []
        all_labels = []
        all_attention = []

        for prompt, response in zip(examples['prompt'], examples['response']):
            response_str = response if isinstance(response, str) else str(response)
            messages = extract_chat_messages(prompt)

            prompt_formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_formatted + response_str + eos_token

            prompt_tokenized = tokenizer(
                prompt_formatted, add_special_tokens=False, return_offsets_mapping=True
            )
            full_tokenized = tokenizer(
                full_text, truncation=True, max_length=max_length,
                add_special_tokens=False, return_offsets_mapping=True,
            )

            input_ids = full_tokenized["input_ids"]
            offsets = full_tokenized["offset_mapping"]
            prompt_len = len(prompt_tokenized["input_ids"])
            full_len = len(input_ids)

            labels = list(input_ids)

            for i in range(min(prompt_len, full_len)):
                labels[i] = -100

            info_spans_in_full = find_info_char_spans(full_text)
            if info_spans_in_full:
                info_mask = build_mask_from_char_spans(offsets, info_spans_in_full)
                for i in range(full_len):
                    if info_mask[i]:
                        labels[i] = -100

            kept = sum(1 for l in labels if l != -100)
            if kept == 0:
                labels[-1] = input_ids[-1]

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention.append([1] * full_len)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention,
        }

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    _report_mask_stats(tokenized_dataset)
    return tokenized_dataset


def _report_mask_stats(dataset):
    """打印 mask 统计信息。"""
    total_tokens = 0
    masked_tokens = 0
    trained_tokens = 0
    for example in dataset:
        labels = example['labels']
        total_tokens += len(labels)
        m = sum(1 for l in labels if l == -100)
        masked_tokens += m
        trained_tokens += len(labels) - m
    print(f"  [Mask 统计] 总 tokens: {total_tokens}, "
          f"masked: {masked_tokens} ({100*masked_tokens/max(total_tokens,1):.1f}%), "
          f"训练 tokens: {trained_tokens} ({100*trained_tokens/max(total_tokens,1):.1f}%)")


def train_sft(
    base_model: str,
    data_file: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_length: int = 4096,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 500,
    logging_steps: int = 10,
    warmup_steps: int = 20,
    bf16: bool = True,
):
    print(f"加载模型: {base_model}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print(f"准备训练数据: {data_file}")
    train_dataset = prepare_sft_data(data_file, tokenizer, max_length)
    print(f"训练样本数: {len(train_dataset)}")

    data_collator = SFTDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    fsdp_config = {}
    use_fsdp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_fsdp:
        fsdp_config = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                "fsdp_backward_prefetch": "backward_pre",
                "fsdp_forward_prefetch": False,
                "limit_all_gathers": True,
            },
        }
        print(f"[FSDP] 启用全分片数据并行，world_size={os.environ.get('WORLD_SIZE')}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=bf16,
        fp16=not bf16,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="wandb" if os.environ.get("WANDB_PROJECT") else "none",
        **fsdp_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()

    print(f"保存模型到: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT训练脚本")
    parser.add_argument("--base_model", type=str, required=True,
                       help="基础模型路径")
    parser.add_argument("--data_file", type=str, required=True,
                       help="SFT数据集文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="模型输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="per-device 批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="最大序列长度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="梯度累积步数")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="使用 bfloat16 (A100/H100 推荐)")
    parser.add_argument("--no_bf16", action="store_true",
                       help="禁用 bf16，使用 fp16")

    args = parser.parse_args()

    train_sft(
        base_model=args.base_model,
        data_file=args.data_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=not args.no_bf16,
    )
