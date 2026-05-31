"""
Qwen3-0.6B 预训练主入口
"""
import argparse
import logging
import os
from typing import Optional

# 设置环境变量（可选，用于优化）
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from src.config import get_config, Config
from src.data import DataModule
from src.model import ModelModule
from src.trainer import TrainerModule
from src.utils import setup_logging, print_model_info, run_generation_test


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Pretraining Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data (overrides config)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (overrides config)",
    )
    
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="Context length (overrides config)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per device (overrides config)",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path",
    )
    
    parser.add_argument(
        "--disable_swanlab",
        action="store_true",
        help="Disable SwanLab logging",
    )
    
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None):
    """主函数"""
    # 解析参数
    if args is None:
        args = parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(level=log_level, log_file=args.log_file)
    logger.info("Starting Qwen3-0.6B Pretraining")
    
    # 加载配置
    logger.info(f"Loading configuration from {args.config}")
    config = get_config(args.config)
    
    # 命令行参数覆盖配置
    if args.data_path:
        config.data.data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.model_name:
        config.model.model_name = args.model_name
    if args.context_length:
        config.data.context_length = args.context_length
        config.model.max_position_embeddings = args.context_length
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
        config.training.per_device_eval_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    if args.disable_swanlab:
        config.swanlab.enabled = False
    
    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.model.cache_dir, exist_ok=True)
    
    # ========== 数据准备 ==========
    logger.info("=" * 50)
    logger.info("Preparing Data")
    logger.info("=" * 50)
    
    data_module = DataModule(
        data_path=config.data.data_path,
        model_name=config.model.model_name,
        context_length=config.data.context_length,
        test_size=config.data.test_size,
        seed=config.data.seed,
        cache_dir=config.model.cache_dir,
        num_proc=config.data.num_proc,
        use_modelscope=True,
    )
    data_module.prepare()
    
    # 更新词表大小
    config.model.vocab_size = data_module.get_vocab_size()
    logger.info(f"Vocab size: {config.model.vocab_size}")
    
    # ========== 模型构建 ==========
    logger.info("=" * 50)
    logger.info("Building Model")
    logger.info("=" * 50)
    
    model_module = ModelModule(
        model_name=config.model.model_name,
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        intermediate_size=config.model.intermediate_size,
        num_attention_heads=config.model.num_attention_heads,
        num_hidden_layers=config.model.num_hidden_layers,
        max_position_embeddings=config.model.max_position_embeddings,
        bos_token_id=config.model.bos_token_id,
        eos_token_id=config.model.eos_token_id,
        cache_dir=config.model.cache_dir,
        use_modelscope=True,
        model_type="qwen2",  # Qwen3 使用 Qwen2 架构
    )
    model_module.build()
    
    # 打印模型信息
    print_model_info(model_module.model, model_module.config)
    
    # ========== 训练设置 ==========
    logger.info("=" * 50)
    logger.info("Setting Up Training")
    logger.info("=" * 50)
    
    # 准备回调函数
    callbacks = []
    if config.swanlab.enabled:
        try:
            import swanlab
            from swanlab.integration.transformers import SwanLabCallback
            
            swanlab.init(
                project=config.swanlab.project_name,
                experiment_name=config.swanlab.experiment_name,
                description=config.swanlab.description,
                config=config.to_dict(),
            )
            callbacks.append(SwanLabCallback())
            logger.info("SwanLab initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SwanLab: {e}")
            config.swanlab.enabled = False
    
    # 创建训练器
    trainer_module = TrainerModule()
    trainer_module.setup(
        model=model_module.model,
        tokenizer=data_module.tokenizer,
        data_collator=data_module.data_collator,
        train_dataset=data_module.get_train_dataset(),
        eval_dataset=data_module.get_eval_dataset(),
        callbacks=callbacks if callbacks else None,
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        learning_rate=config.training.learning_rate,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        seed=config.training.seed,
        dataloader_num_workers=config.training.dataloader_num_workers,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=config.training.report_to if not config.swanlab.enabled else "swanlab",
    )
    
    # ========== 开始训练 ==========
    logger.info("=" * 50)
    logger.info("Starting Training")
    logger.info("=" * 50)
    
    train_output = trainer_module.train()
    logger.info(f"Training completed: {train_output}")
    
    # ========== 保存模型 ==========
    logger.info("=" * 50)
    logger.info("Saving Model")
    logger.info("=" * 50)
    
    model_save_path = os.path.join(config.training.output_dir, "Weight")
    trainer_module.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # ========== 生成测试 ==========
    logger.info("=" * 50)
    logger.info("Running Generation Test")
    logger.info("=" * 50)
    
    run_generation_test(
        model=model_module.model,
        tokenizer=data_module.tokenizer,
        prompts=["人工智能", "牛顿", "北京市", "亚洲历史"],
        swanlab_enabled=config.swanlab.enabled,
    )
    
    # 结束 SwanLab
    if config.swanlab.enabled:
        try:
            import swanlab
            swanlab.finish()
        except Exception as e:
            logger.warning(f"Failed to finish SwanLab: {e}")
    
    logger.info("=" * 50)
    logger.info("Pretraining Completed Successfully!")
    logger.info("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())
