"""
工具函数模块
提供日志、生成测试等辅助功能
"""
import logging
import sys
from typing import List, Optional, Dict, Any
import transformers


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """设置日志系统
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        
    Returns:
        Logger 对象
    """
    logger = logging.getLogger("qwen3_pretrain")
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def generate_text(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    do_sample: bool = True,
    num_return_sequences: int = 1
) -> List[str]:
    """使用模型生成文本
    
    Args:
        model: 模型实例
        tokenizer: Tokenizer
        prompt: 输入提示词
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: Top-p 采样参数
        do_sample: 是否采样
        num_return_sequences: 返回序列数量
        
    Returns:
        生成的文本列表
    """
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    results = pipe(prompt)
    return [result[0]["generated_text"] for result in results]


def run_generation_test(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: Optional[List[str]] = None,
    swanlab_enabled: bool = False
) -> Dict[str, Any]:
    """运行生成测试
    
    Args:
        model: 模型实例
        tokenizer: Tokenizer
        prompts: 测试提示词列表
        swanlab_enabled: 是否记录到 SwanLab
        
    Returns:
        生成结果字典
    """
    if prompts is None:
        prompts = ["人工智能", "牛顿", "北京市", "亚洲历史"]
    
    results = {}
    swanlab_texts = []
    
    for prompt in prompts:
        try:
            generated = generate_text(model, tokenizer, prompt, num_return_sequences=1)
            results[prompt] = generated[0]
            
            if swanlab_enabled:
                import swanlab
                swanlab_texts.append(swanlab.Text(generated[0]))
            
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated[0][:200]}...")
            print("-" * 50)
        except Exception as e:
            print(f"Error generating for '{prompt}': {e}")
            results[prompt] = f"Error: {str(e)}"
    
    # 记录到 SwanLab
    if swanlab_enabled and swanlab_texts:
        try:
            import swanlab
            swanlab.log({"Generate": swanlab_texts})
        except Exception as e:
            print(f"Failed to log to SwanLab: {e}")
    
    return results


def print_model_info(
    model: transformers.PreTrainedModel,
    config: transformers.PretrainedConfig
):
    """打印模型信息"""
    model_size = sum(t.numel() for t in model.parameters())
    print("\n" + "=" * 50)
    print("Model Information")
    print("=" * 50)
    print(f"Model Config:\n{config}")
    print(f"\nModel Size: {model_size / 1000**2:.1f}M parameters")
    print(f"Model Size: {model_size / 1000**3:.2f}B parameters")
    print("=" * 50 + "\n")
