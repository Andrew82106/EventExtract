# -*- coding: utf-8 -*-
"""
使用示例
展示如何使用简单询问式构图算法
"""
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main

# 示例1: 使用默认配置运行
print("示例1: 使用默认配置处理 suicide_ied 攻击类型")
print("-" * 60)
main(attack_type='suicide_ied')

# 示例2: 处理不同的攻击类型
# print("\n示例2: 处理 backpack_ied 攻击类型")
# print("-" * 60)
# main(attack_type='backpack_ied')

# 示例3: 使用自定义API密钥
# print("\n示例3: 使用自定义API密钥")
# print("-" * 60)
# main(attack_type='suicide_ied', api_key='your_custom_api_key')

