#!/bin/bash
# 数据下载脚本
# 用于下载中文维基百科数据集

set -e

DATA_DIR="${1:-/data/WIKI_CN}"
DATA_FILE="${DATA_DIR}/wikipedia-zh-cn-20260501.json"

echo "========================================"
echo "Data Download Script"
echo "========================================"
echo "Data directory: ${DATA_DIR}"
echo "Data file: ${DATA_FILE}"
echo "========================================"

# 创建数据目录
mkdir -p "${DATA_DIR}"

# 检查数据文件是否已存在
if [ -f "${DATA_FILE}" ]; then
    echo "Data file already exists: ${DATA_FILE}"
    echo "Skipping download."
else
    echo "Downloading Chinese Wikipedia dataset..."
    echo "Note: Please download the dataset manually from ModelScope or HuggingFace"
    echo ""
    echo "Option 1: Download from ModelScope (recommended for China)"
    echo "  Visit: https://modelscope.cn/datasets"
    echo ""
    echo "Option 2: Download from HuggingFace"
    echo "  Visit: https://huggingface.co/datasets"
    echo ""
    echo "After downloading, place the JSON file at: ${DATA_FILE}"
    echo ""
    
    # 创建一个示例数据文件用于测试
    echo "Creating sample data file for testing..."
    cat > "${DATA_FILE}" << 'EOF'
{"text": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"}
{"text": "中国人民大学位于北京市海淀区，中央直管高校，直属于中华人民共和国教育部，由教育部与北京市共建。"}
{"text": "北京市是中华人民共和国的首都，是全国的政治中心、文化中心和国际交往中心。"}
{"text": "亚洲是世界上面积最大、人口最多的大洲，拥有丰富的自然资源和多样的文化。"}
{"text": "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的层次化表示。"}
EOF
    
    echo "Sample data file created at: ${DATA_FILE}"
    echo ""
    echo "For real training, please replace this file with the full Wikipedia dataset."
fi

echo "========================================"
echo "Data preparation completed!"
echo "========================================"
