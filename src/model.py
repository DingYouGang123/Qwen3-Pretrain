"""
模型定义模块
负责模型的创建和配置
"""
import logging
from typing import Optional
import transformers

logger = logging.getLogger(__name__)


def create_model_config(
    model_name: str,
    vocab_size: int,
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    num_attention_heads: int = 16,
    num_hidden_layers: int = 24,
    max_position_embeddings: int = 512,
    bos_token_id: int = 151643,
    eos_token_id: int = 151643,
    cache_dir: str = "./cache",
    use_modelscope: bool = True
) -> transformers.PretrainedConfig:
    """创建模型配置
    
    Args:
        model_name: 模型名称（如 Qwen/Qwen3-0.6B）
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        intermediate_size: 中间层维度
        num_attention_heads: 注意力头数
        num_hidden_layers: 隐藏层数量
        max_position_embeddings: 最大位置嵌入
        bos_token_id: BOS token ID
        eos_token_id: EOS token ID
        cache_dir: 缓存目录
        use_modelscope: 是否使用 ModelScope 下载
        
    Returns:
        模型配置对象
    """
    logger.info(f"Creating model config from {model_name}")
    
    try:
        if use_modelscope:
            # 使用 ModelScope 下载配置
            import modelscope
            modelscope.AutoConfig.from_pretrained(model_name).save_pretrained(cache_dir)
            config = transformers.AutoConfig.from_pretrained(cache_dir)
        else:
            config = transformers.AutoConfig.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Failed to load with modelscope, trying direct load: {e}")
        config = transformers.AutoConfig.from_pretrained(model_name)
    
    # 更新配置参数
    config.vocab_size = vocab_size
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_attention_heads = num_attention_heads
    config.num_hidden_layers = num_hidden_layers
    config.max_position_embeddings = max_position_embeddings
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id
    
    logger.info(f"Model config created:\n{config}")
    return config


def create_model(
    config: transformers.PretrainedConfig,
    model_type: str = "qwen2"
) -> transformers.PreTrainedModel:
    """创建模型实例
    
    Args:
        config: 模型配置
        model_type: 模型类型（qwen2, qwen3 等）
        
    Returns:
        模型实例
    """
    logger.info(f"Creating {model_type} model from config")
    
    # Qwen3 目前可以使用 Qwen2 的架构
    if model_type in ["qwen2", "qwen3"]:
        model = transformers.Qwen2ForCausalLM(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_size = sum(t.numel() for t in model.parameters())
    logger.info(f"Model created. Size: {model_size / 1000**2:.1f}M parameters")
    
    return model


def load_pretrained_model(
    model_path: str,
    use_modelscope: bool = True
) -> transformers.PreTrainedModel:
    """加载预训练模型
    
    Args:
        model_path: 模型路径
        use_modelscope: 是否使用 ModelScope
        
    Returns:
        预训练模型
    """
    logger.info(f"Loading pretrained model from {model_path}")
    
    try:
        if use_modelscope:
            import modelscope
            modelscope.AutoModelForCausalLM.from_pretrained(model_path).save_pretrained("./cache")
            model = transformers.AutoModelForCausalLM.from_pretrained("./cache")
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"Failed to load with modelscope, trying direct load: {e}")
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    
    return model


class ModelModule:
    """模型模块：封装完整的模型创建流程"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        vocab_size: int = 151936,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 512,
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        cache_dir: str = "./cache",
        use_modelscope: bool = True,
        model_type: str = "qwen2"
    ):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.cache_dir = cache_dir
        self.use_modelscope = use_modelscope
        self.model_type = model_type
        
        self.config: Optional[transformers.PretrainedConfig] = None
        self.model: Optional[transformers.PreTrainedModel] = None
    
    def build(self) -> "ModelModule":
        """构建模型"""
        # 创建配置
        self.config = create_model_config(
            model_name=self.model_name,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            cache_dir=self.cache_dir,
            use_modelscope=self.use_modelscope
        )
        
        # 创建模型
        self.model = create_model(self.config, self.model_type)
        
        logger.info("Model building completed!")
        return self
    
    def get_model_size(self) -> int:
        """获取模型参数量"""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        return sum(t.numel() for t in self.model.parameters())
    
    def get_model_size_mb(self) -> float:
        """获取模型参数量（MB）"""
        return self.get_model_size() / 1000**2
