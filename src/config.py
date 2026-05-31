"""
配置管理模块
提供统一的配置加载和管理功能
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    """数据配置"""
    data_path: str = "/data/WIKI_CN/wikipedia-zh-cn-20260501.json"
    test_size: float = 0.1
    seed: int = 42
    context_length: int = 512
    num_proc: int = 4


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen3-0.6B"
    cache_dir: str = "./cache"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    max_position_embeddings: int = 512
    bos_token_id: int = 151643
    eos_token_id: int = 151643


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./checkpoints"
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 50
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    weight_decay: float = 0.1
    warmup_steps: int = 200
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    learning_rate: float = 5e-4
    save_steps: int = 500
    save_total_limit: int = 10
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "none"


@dataclass
class SwanLabConfig:
    """SwanLab 实验追踪配置"""
    enabled: bool = True
    project_name: str = "WikiLLM"
    experiment_name: Optional[str] = None
    description: str = "Qwen3-0.6B Pretraining"


@dataclass
class Config:
    """总配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """从字典加载配置"""
        config = cls()
        
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if 'swanlab' in config_dict:
            for key, value in config_dict['swanlab'].items():
                if hasattr(config.swanlab, key):
                    setattr(config.swanlab, key, value)
        
        return config

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            'data': {k: v for k, v in self.data.__dict__.items() if not k.startswith('_')},
            'model': {k: v for k, v in self.model.__dict__.items() if not k.startswith('_')},
            'training': {k: v for k, v in self.training.__dict__.items() if not k.startswith('_')},
            'swanlab': {k: v for k, v in self.swanlab.__dict__.items() if not k.startswith('_')},
        }

    def save(self, path: str):
        """保存配置到 YAML 文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)


def get_config(config_path: Optional[str] = None) -> Config:
    """获取配置对象
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        
    Returns:
        Config 对象
    """
    if config_path is None:
        return Config()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return Config.from_yaml(config_path)
