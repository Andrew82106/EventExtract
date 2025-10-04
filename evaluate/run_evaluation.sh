#!/bin/bash
# 运行评测脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到evaluate目录
cd "$SCRIPT_DIR"

# 运行评测
echo "开始运行事件骨架图评测..."
python evaluator.py

echo ""
echo "评测完成！"
echo "结果已保存到 result/ 目录下"

