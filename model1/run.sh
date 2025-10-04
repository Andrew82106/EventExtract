#!/bin/bash
# 快速运行脚本

echo "================================================"
echo "简单询问式构图算法 - 快速启动脚本"
echo "================================================"

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.7+"
    exit 1
fi

# 进入model1目录
cd "$(dirname "$0")"

# 安装依赖（如果需要）
echo ""
echo "检查依赖..."
pip install -r requirements.txt -q

# 运行主程序
echo ""
echo "开始运行算法..."
echo ""

# 检查是否提供了攻击类型参数
if [ -z "$1" ]; then
    echo "使用默认攻击类型: suicide_ied"
    python main.py
else
    echo "使用指定攻击类型: $1"
    python main.py --attack_type "$1"
fi

echo ""
echo "================================================"
echo "完成!"
echo "================================================"

