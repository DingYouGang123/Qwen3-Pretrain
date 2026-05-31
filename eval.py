#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估脚本：用于计算预训练模型的 Perplexity (PPL) 和其他指标。
支持在验证集上进行评估，也可用于单独测试某个 checkpoint。
"""

import os
import sys
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from config import get_args, setup_logging

# 设置日志
logger = logging.getLogger(__name__)

def compute_perplexity(model, dataloader, device, max_batches=None):
    """
    计算模型在给定数据集上的 Perplexity。
    
    Args:
        model: 加载的模型
        dataloader: 验证集 DataLoader
        device: 计算设备
        max_batches: 最大评估 batch 数，None 表示全部
    
    Returns:
        float: 平均 Perplexity
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100) # 假设 padding token 被 mask 为 -100
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating PPL")
        for i, batch in enumerate(pbar):
            if max_batches and i >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            loss = outputs.loss
            # 累加 loss * token数量 (为了加权平均)
            # labels 中有效 token 的数量 (非 -100 的数量)
            valid_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    
    if total_tokens == 0:
        logger.warning("No valid tokens found for evaluation.")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def main():
    args = get_args()
    setup_logging(args.log_level, args.log_file)
    
    logger.info(f"Starting evaluation for {args.model_name}")
    logger.info(f"Checkpoint: {args.output_dir}") # 这里假设 output_dir 指向具体的 checkpoint 文件夹
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. 加载 Tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # 2. 加载模型
    logger.info("Loading model...")
    # 如果 args.output_dir 存在且包含模型文件，则从本地加载，否则从 hub 加载
    model_path = args.output_dir if os.path.exists(os.path.join(args.output_dir, "config.json")) else args.model_name
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)

    # 3. 准备数据 (验证集)
    logger.info("Loading validation dataset...")
    # 实际项目中，验证集路径应通过参数传入，这里复用 data_path 逻辑，实际需区分 train/val
    # 假设数据集中有 'validation' split，或者我们取一部分作为验证
    try:
        # 这里简化处理，实际应根据 args.val_data_path 加载
        # 如果是本地文件，可能需要修改加载逻辑
        dataset = load_dataset("json", data_files=args.data_path, split="train") # 占位，实际应加载 val
        
        # 简单的预处理函数 (需与训练时保持一致)
        def preprocess(example):
            encodings = tokenizer(
                example['text'],
                truncation=True,
                max_length=args.context_length,
                padding=False, # eval 时通常不动态 padding，或者使用 collator
                return_tensors=None
            )
            # 构建 labels，copy input_ids
            input_ids = encodings['input_ids']
            labels = input_ids.copy()
            
            # 如果需要 padding (batch 化时需要)
            # 这里为了演示简单逻辑，实际建议使用 DataCollatorForLanguageModeling
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': encodings.get('attention_mask', [1]*len(input_ids))
            }
        
        # 由于 load_dataset 返回的是列表式数据，我们需要一个简单的 collate_fn 来 batching
        # 这里为了代码简洁，直接使用 HuggingFace Dataset 的 map 和 collator
        
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.context_length)
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_datasets = tokenized_datasets.add_column("labels", tokenized_datasets["input_ids"])
        
        # 如果没有验证集切分，这里强行切分 10% 作为验证用于演示
        if "validation" not in tokenized_datasets:
            split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
            eval_dataset = split_dataset["test"]
        else:
            eval_dataset = tokenized_datasets["validation"]
            
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=args.batch_size, 
            collate_fn=data_collator
        )
        
    except Exception as e:
        logger.error(f"Failed to load or process dataset: {e}")
        sys.exit(1)

    # 4. 执行评估
    logger.info("Calculating Perplexity...")
    ppl = compute_perplexity(model, eval_dataloader, device, max_batches=None)
    
    logger.info("="*30)
    logger.info(f"Evaluation Results:")
    logger(f"Model: {model_path}")
    logger(f"Perplexity: {ppl:.4f}")
    logger(f"Log Loss: {torch.log(torch.tensor(ppl)).item():.4f}")
    logger("="*30)
    
    # 保存结果
    result_file = os.path.join(args.output_dir, "eval_results.json")
    # 确保目录存在
    os.makedirs(os.path.dirname(result_file) if os.path.dirname(result_file) else ".", exist_ok=True)
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "perplexity": ppl,
            "log_loss": torch.log(torch.tensor(ppl)).item()
        }, f, indent=4)
    
    logger.info(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
