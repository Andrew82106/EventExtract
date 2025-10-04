#!/bin/bash
# 运行基于意义段的迭代构图算法

# 设置默认攻击类型
ATTACK_TYPE="${1:-suicide_ied}"

echo "开始运行基于意义段的迭代构图算法 (Model2)"
echo "攻击类型: $ATTACK_TYPE"
echo "================================"

# 运行主程序
python main.py --attack_type "$ATTACK_TYPE"

echo "================================"
echo "运行完成"

