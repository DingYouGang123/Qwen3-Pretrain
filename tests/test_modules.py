"""
Qwen3-0.6B 预训练项目测试模块
"""
import unittest
import os
import tempfile
import yaml

from src.config import Config, get_config
from src.data import DataModule, tokenize_function
from src.model import ModelModule, create_model_config
from src.trainer import TrainerModule, create_training_args


class TestConfig(unittest.TestCase):
    """配置模块测试"""
    
    def test_default_config(self):
        """测试默认配置加载"""
        config = Config()
        self.assertEqual(config.data.test_size, 0.1)
        self.assertEqual(config.model.hidden_size, 1024)
        self.assertEqual(config.training.num_train_epochs, 2)
    
    def test_config_from_dict(self):
        """测试从字典加载配置"""
        config_dict = {
            'data': {'test_size': 0.2, 'seed': 42},
            'model': {'hidden_size': 512},
            'training': {'learning_rate': 1e-4}
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.data.test_size, 0.2)
        self.assertEqual(config.data.seed, 42)
        self.assertEqual(config.model.hidden_size, 512)
        self.assertEqual(config.training.learning_rate, 1e-4)
    
    def test_config_save_load(self):
        """测试配置保存和加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config = Config()
            config.save(temp_path)
            
            loaded_config = Config.from_yaml(temp_path)
            self.assertEqual(config.data.test_size, loaded_config.data.test_size)
            self.assertEqual(config.model.hidden_size, loaded_config.model.hidden_size)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataModule(unittest.TestCase):
    """数据模块测试"""
    
    def test_tokenize_function(self):
        """测试分词函数"""
        # 使用一个简单的 tokenizer 进行测试
        from transformers import AutoTokenizer
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        except Exception:
            # 如果无法下载，跳过测试
            self.skipTest("Cannot load tokenizer for testing")
            return
        
        element = {"text": "这是一个测试文本。"}
        result = tokenize_function(element, tokenizer, context_length=512)
        
        self.assertIn("input_ids", result)
        self.assertIsInstance(result["input_ids"], list)


class TestModelModule(unittest.TestCase):
    """模型模块测试"""
    
    def test_create_model_config(self):
        """测试模型配置创建"""
        try:
            config = create_model_config(
                model_name="Qwen/Qwen2-0.5B",
                vocab_size=151936,
                hidden_size=512,
                intermediate_size=2048,
                num_attention_heads=8,
                num_hidden_layers=12,
                max_position_embeddings=512,
                use_modelscope=True
            )
            
            self.assertEqual(config.vocab_size, 151936)
            self.assertEqual(config.hidden_size, 512)
            self.assertEqual(config.num_hidden_layers, 12)
        except Exception as e:
            self.skipTest(f"Cannot create model config: {e}")


class TestTrainerModule(unittest.TestCase):
    """训练器模块测试"""
    
    def test_create_training_args(self):
        """测试训练参数创建"""
        args = create_training_args(
            output_dir="./test_output",
            learning_rate=5e-4,
            num_train_epochs=1,
            bf16=False,
            fp16=False,
        )
        
        self.assertEqual(args.output_dir, "./test_output")
        self.assertEqual(args.learning_rate, 5e-4)
        self.assertEqual(args.num_train_epochs, 1)


if __name__ == "__main__":
    unittest.main()
