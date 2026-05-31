# Qwen3 Pretrain Project

工业级 Qwen3-0.6B 大模型预训练项目

## 项目结构

```
/workspace/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── data.py             # 数据加载与预处理
│   ├── model.py            # 模型定义
│   ├── trainer.py          # 训练器封装
│   └── utils.py            # 工具函数
├── configs/                # 配置文件目录
│   ├── default.yaml        # 默认配置
│   └── qwen3_0.6b.yaml     # Qwen3-0.6B 专用配置
├── scripts/                # 脚本目录
│   ├── train.sh            # 训练启动脚本
│   └── download_data.sh    # 数据下载脚本
├── data/                   # 数据目录
├── logs/                   # 日志目录
├── checkpoints/            # 模型检查点目录
├── tests/                  # 测试目录
├── requirements.txt        # 依赖列表
├── setup.py                # 安装脚本
├── README.md               # 项目说明
└── train.py                # 训练入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 准备数据

```bash
bash scripts/download_data.sh
```

### 3. 修改配置

编辑 `configs/qwen3_0.6b.yaml` 文件，调整数据路径、训练参数等。

### 4. 开始训练
训练前请确保数据路径正确
```bash
python train.py --config configs/qwen3_0.6b.yaml
```

或使用多卡训练：

```bash
bash scripts/train.sh
```

## 特性

- ✅ 支持 Qwen3-0.6B 模型架构
- ✅ 完整的配置管理系统
- ✅ 模块化代码设计
- ✅ SwanLab 实验追踪集成
- ✅ ModelScope 模型下载支持
- ✅ 分布式训练支持
- ✅ 混合精度训练 (BF16/FP16)
- ✅ 梯度累积
- ✅ 学习率调度器
- ✅ 模型保存与生成测试

## 配置说明

主要配置项在 `configs/qwen3_0.6b.yaml` 中：

- `data_path`: 训练数据路径
- `model_name`: 基础模型名称
- `context_length`: 上下文长度
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_train_epochs`: 训练轮数
- `output_dir`: 输出目录

## 许可证

MIT License
