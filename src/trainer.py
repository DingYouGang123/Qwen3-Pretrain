"""
训练器模块
封装 HuggingFace Trainer 并提供额外功能
"""
import logging
from typing import Optional, List
import transformers
from transformers import TrainingArguments
from datasets import Dataset

logger = logging.getLogger(__name__)


def create_training_args(
    output_dir: str = "./checkpoints",
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    eval_strategy: str = "steps",
    eval_steps: int = 500,
    logging_steps: int = 50,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 2,
    weight_decay: float = 0.1,
    warmup_steps: int = 200,
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    learning_rate: float = 5e-4,
    save_steps: int = 500,
    save_total_limit: int = 10,
    bf16: bool = True,
    fp16: bool = False,
    seed: int = 42,
    dataloader_num_workers: int = 4,
    remove_unused_columns: bool = False,
    report_to: str = "none",
) -> TrainingArguments:
    """创建训练参数配置
    
    Args:
        output_dir: 输出目录
        per_device_train_batch_size: 每个设备的训练批次大小
        per_device_eval_batch_size: 每个设备的评估批次大小
        eval_strategy: 评估策略（steps/epoch）
        eval_steps: 评估步数间隔
        logging_steps: 日志记录步数间隔
        gradient_accumulation_steps: 梯度累积步数
        num_train_epochs: 训练轮数
        weight_decay: 权重衰减
        warmup_steps: 预热步数
        optim: 优化器类型
        lr_scheduler_type: 学习率调度器类型
        learning_rate: 学习率
        save_steps: 保存 checkpoints 的步数间隔
        save_total_limit: 保留的 checkpoints 最大数量
        bf16: 是否使用 BF16 混合精度
        fp16: 是否使用 FP16 混合精度
        seed: 随机种子
        dataloader_num_workers: 数据加载器工作进程数
        remove_unused_columns: 是否移除未使用的列
        report_to: 报告目标（none/swanlab 等）
        
    Returns:
        TrainingArguments 对象
    """
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        learning_rate=learning_rate,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=bf16,
        fp16=fp16,
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=remove_unused_columns,
        report_to=report_to,
    )
    
    logger.info(f"Training arguments created:\n{args}")
    return args


def create_trainer(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    args: TrainingArguments,
    data_collator: transformers.DataCollatorForLanguageModeling,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    callbacks: Optional[List[transformers.TrainerCallback]] = None,
) -> transformers.Trainer:
    """创建 Trainer 对象
    
    Args:
        model: 模型实例
        tokenizer: Tokenizer
        args: 训练参数
        data_collator: 数据整理器
        train_dataset: 训练数据集
        eval_dataset: 评估数据集（可选）
        callbacks: 回调函数列表
        
    Returns:
        Trainer 对象
    """
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    
    logger.info("Trainer created successfully")
    return trainer


class TrainerModule:
    """训练器模块：封装完整的训练流程"""
    
    def __init__(self):
        self.args: Optional[TrainingArguments] = None
        self.trainer: Optional[transformers.Trainer] = None
    
    def setup(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        data_collator: transformers.DataCollatorForLanguageModeling,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[transformers.TrainerCallback]] = None,
        **training_kwargs
    ) -> "TrainerModule":
        """设置训练器
        
        Args:
            model: 模型实例
            tokenizer: Tokenizer
            data_collator: 数据整理器
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            callbacks: 回调函数列表
            **training_kwargs: 传递给 TrainingArguments 的参数
            
        Returns:
            self
        """
        # 创建训练参数
        self.args = create_training_args(**training_kwargs)
        
        # 创建训练器
        self.trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            args=self.args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )
        
        return self
    
    def train(self) -> dict:
        """开始训练
        
        Returns:
            训练输出
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        
        logger.info("Starting training...")
        return self.trainer.train()
    
    def save_model(self, output_dir: str):
        """保存模型
        
        Args:
            output_dir: 保存目录
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
    
    def evaluate(self) -> dict:
        """评估模型
        
        Returns:
            评估结果
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        
        logger.info("Evaluating model...")
        return self.trainer.evaluate()
