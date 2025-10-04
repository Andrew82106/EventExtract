#!/bin/bash
# 重新合并图并评测的运行脚本
# 注意：该脚本已更新为使用 Python 函数接口

echo "================================"
echo "重新合并图并评测工具"
echo "================================"
echo ""

# 显示帮助信息
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "用法:"
    echo "  bash run_remerge.sh              # 重新合并所有模型并评测（默认）"
    echo ""
    echo "更多使用方式，请直接在 Python 代码中调用函数："
    echo ""
    echo "  from evaluate.remerge_and_evaluate import remerge_only, remerge_and_evaluate"
    echo ""
    echo "  # 只重新合并，不评测"
    echo "  remerge_only()                              # 所有模型"
    echo "  remerge_only('model1')                      # 指定模型"
    echo "  remerge_only('model1', 'suicide_ied')       # 指定模型和攻击类型"
    echo ""
    echo "  # 重新合并并评测"
    echo "  remerge_and_evaluate()                      # 所有模型"
    echo "  remerge_and_evaluate('model1')              # 指定模型"
    echo "  remerge_and_evaluate('model1', 'suicide_ied') # 指定模型和攻击类型"
    echo ""
    exit 0
fi

# 运行默认操作：重新合并所有模型并评测
echo "执行默认操作: 重新合并所有模型并评测"
python remerge_and_evaluate.py

echo ""
echo "================================"
echo "完成!"
echo "================================"
