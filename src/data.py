"""
数据加载与预处理模块
负责数据集的加载、分词和预处理
"""
import logging
from typing import Dict, Any, Optional
import datasets
import transformers

logger = logging.getLogger(__name__)


def load_dataset(data_path: str, test_size: float = 0.1, seed: int = 42) -> datasets.DatasetDict:
    """加载数据集并划分训练集和测试集
    
    Args:
        data_path: 数据文件路径 (JSON 格式)
        test_size: 测试集比例
        seed: 随机种子
        
    Returns:
        DatasetDict 包含 train 和 test 数据集
    """
    logger.info(f"Loading dataset from {data_path}")
    
    raw_datasets = datasets.load_dataset("json", data_files=data_path)
    
    # 如果只有 train 分割，进行训练/测试划分
    if "train" in raw_datasets:
        raw_datasets = raw_datasets["train"].train_test_split(
            test_size=test_size, 
            seed=seed
        )
    
    logger.info(f"Dataset info:\n{raw_datasets}")
    return raw_datasets


def load_tokenizer(
    model_name: str, 
    cache_dir: str = "./cache",
    use_modelscope: bool = True
) -> transformers.PreTrainedTokenizer:
    """加载 Tokenizer
    
    Args:
        model_name: 模型名称或本地路径
        cache_dir: 缓存目录
        use_modelscope: 是否使用 ModelScope 下载（适用于国内网络）
        
    Returns:
        Tokenizer 对象
    """
    logger.info(f"Loading tokenizer from {model_name}")
    
    try:
        if use_modelscope:
            # 使用 ModelScope 下载
            import modelscope
            modelscope.AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)
            tokenizer = transformers.AutoTokenizer.from_pretrained(cache_dir)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Failed to load with modelscope, trying direct load: {e}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer


def tokenize_function(
    element: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    context_length: int
) -> Dict[str, Any]:
    """对文本进行分词处理
    
    Args:
        element: 输入文本元素
        tokenizer: Tokenizer 对象
        context_length: 上下文长度
        
    Returns:
        分词后的输入 IDs
    """
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    
    return {"input_ids": input_batch}


def preprocess_dataset(
    raw_datasets: datasets.DatasetDict,
    tokenizer: transformers.PreTrainedTokenizer,
    context_length: int = 512,
    num_proc: int = 4
) -> datasets.DatasetDict:
    """预处理数据集：分词和格式化
    
    Args:
        raw_datasets: 原始数据集
        tokenizer: Tokenizer 对象
        context_length: 上下文长度
        num_proc: 并行处理进程数
        
    Returns:
        预处理后的数据集
    """
    logger.info(f"Preprocessing dataset with context_length={context_length}")
    
    tokenize_fn = lambda x: tokenize_function(x, tokenizer, context_length)
    
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=num_proc,
        desc="Tokenizing datasets",
    )
    
    logger.info(f"Tokenized dataset info:\n{tokenized_datasets}")
    return tokenized_datasets


def create_data_collator(
    tokenizer: transformers.PreTrainedTokenizer,
    mlm: bool = False
) -> transformers.DataCollatorForLanguageModeling:
    """创建数据整理器
    
    Args:
        tokenizer: Tokenizer 对象
        mlm: 是否使用 MLM（掩码语言建模），预训练通常为 False
        
    Returns:
        DataCollator 对象
    """
    return transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm
    )


class DataModule:
    """数据模块：封装完整的数据处理流程"""
    
    def __init__(
        self,
        data_path: str,
        model_name: str,
        context_length: int = 512,
        test_size: float = 0.1,
        seed: int = 42,
        cache_dir: str = "./cache",
        num_proc: int = 4,
        use_modelscope: bool = True
    ):
        self.data_path = data_path
        self.model_name = model_name
        self.context_length = context_length
        self.test_size = test_size
        self.seed = seed
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.use_modelscope = use_modelscope
        
        self.raw_datasets: Optional[datasets.DatasetDict] = None
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None
        self.tokenized_datasets: Optional[datasets.DatasetDict] = None
        self.data_collator: Optional[transformers.DataCollatorForLanguageModeling] = None
    
    def prepare(self):
        """准备所有数据组件"""
        # 加载数据集
        self.raw_datasets = load_dataset(
            self.data_path,
            self.test_size,
            self.seed
        )
        
        # 加载 tokenizer
        self.tokenizer = load_tokenizer(
            self.model_name,
            self.cache_dir,
            self.use_modelscope
        )
        
        # 预处理数据集
        self.tokenized_datasets = preprocess_dataset(
            self.raw_datasets,
            self.tokenizer,
            self.context_length,
            self.num_proc
        )
        
        # 创建数据整理器
        self.data_collator = create_data_collator(self.tokenizer, mlm=False)
        
        logger.info("Data preparation completed!")
        return self
    
    def get_train_dataset(self) -> datasets.Dataset:
        """获取训练数据集"""
        if self.tokenized_datasets is None:
            raise RuntimeError("Data not prepared. Call prepare() first.")
        return self.tokenized_datasets["train"]
    
    def get_eval_dataset(self) -> datasets.Dataset:
        """获取评估数据集"""
        if self.tokenized_datasets is None:
            raise RuntimeError("Data not prepared. Call prepare() first.")
        return self.tokenized_datasets["test"]
    
    def get_vocab_size(self) -> int:
        """获取词表大小"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call prepare() first.")
        return len(self.tokenizer)
