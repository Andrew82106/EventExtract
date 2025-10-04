# -*- coding: utf-8 -*-
"""
图验证工具
用于验证已生成的图文件中的事件类型是否符合本体定义
"""
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader
from event_validator import EventTypeValidator


def validate_graph_file(graph_file_path, data_loader):
    """
    验证单个图文件
    
    Args:
        graph_file_path (str): 图文件路径
        data_loader (DataLoader): 数据加载器
    """
    print(f"\n{'='*80}")
    print(f"验证图文件: {os.path.basename(graph_file_path)}")
    print('='*80)
    
    # 读取图文件
    try:
        with open(graph_file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取图文件 - {str(e)}")
        return None
    
    # 创建验证器
    event_types = data_loader.get_event_types()
    validator = EventTypeValidator(event_types)
    
    # 验证图数据
    validated_graph, validation_report = validator.validate_graph_data(
        graph_data,
        auto_fix=False  # 只验证，不修复
    )
    
    # 打印报告
    print(f"\n验证结果:")
    print(f"  总节点数: {validation_report['total_nodes']}")
    print(f"  有效节点: {validation_report['valid_nodes']}")
    print(f"  无效节点: {validation_report['invalid_nodes']}")
    
    if validation_report['total_nodes'] > 0:
        valid_rate = (validation_report['valid_nodes'] / 
                     validation_report['total_nodes'] * 100)
        print(f"  验证通过率: {valid_rate:.2f}%")
    
    # 显示详细问题
    if validation_report['node_issues']:
        print(f"\n发现 {len(validation_report['node_issues'])} 个问题节点:")
        print("-" * 80)
        
        for i, issue_info in enumerate(validation_report['node_issues'], 1):
            print(f"\n问题 {i}: 节点 '{issue_info['node_id']}'")
            for issue in issue_info['issues']:
                print(f"  {issue}")
    
    return validation_report


def validate_all_graphs(graphs_dir, data_loader):
    """
    验证目录中的所有图文件
    
    Args:
        graphs_dir (str): 图文件目录
        data_loader (DataLoader): 数据加载器
    """
    if not os.path.exists(graphs_dir):
        print(f"错误: 目录不存在 - {graphs_dir}")
        return
    
    # 收集所有JSON文件
    graph_files = []
    for file in os.listdir(graphs_dir):
        if file.endswith('.json'):
            graph_files.append(os.path.join(graphs_dir, file))
    
    if not graph_files:
        print(f"警告: 在 {graphs_dir} 中未找到任何JSON文件")
        return
    
    print(f"\n找到 {len(graph_files)} 个图文件")
    
    # 验证每个文件
    all_reports = []
    for graph_file in sorted(graph_files):
        report = validate_graph_file(graph_file, data_loader)
        if report:
            all_reports.append({
                'file': os.path.basename(graph_file),
                'report': report
            })
    
    # 打印汇总报告
    print(f"\n{'='*80}")
    print("汇总报告")
    print('='*80)
    
    total_nodes = sum(r['report']['total_nodes'] for r in all_reports)
    total_valid = sum(r['report']['valid_nodes'] for r in all_reports)
    total_invalid = sum(r['report']['invalid_nodes'] for r in all_reports)
    
    print(f"\n总计:")
    print(f"  验证文件数: {len(all_reports)}")
    print(f"  总节点数: {total_nodes}")
    print(f"  有效节点数: {total_valid}")
    print(f"  无效节点数: {total_invalid}")
    
    if total_nodes > 0:
        overall_rate = (total_valid / total_nodes * 100)
        print(f"  整体验证通过率: {overall_rate:.2f}%")
    
    print(f"\n各文件验证通过率:")
    for item in all_reports:
        total = item['report']['total_nodes']
        valid = item['report']['valid_nodes']
        rate = (valid / total * 100) if total > 0 else 0
        status = "✓" if rate == 100 else "✗"
        print(f"  {status} {item['file']}: {rate:.1f}% ({valid}/{total})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证事件图中的事件类型')
    parser.add_argument('--file', type=str, help='验证单个图文件')
    parser.add_argument('--dir', type=str, help='验证目录中的所有图文件')
    
    args = parser.parse_args()
    
    # 初始化数据加载器
    print("正在加载事件本体...")
    data_loader = DataLoader()
    
    if args.file:
        # 验证单个文件
        validate_graph_file(args.file, data_loader)
    elif args.dir:
        # 验证目录中的所有文件
        validate_all_graphs(args.dir, data_loader)
    else:
        # 默认验证result/graphs目录
        from config import GRAPHS_PATH
        print(f"\n使用默认目录: {GRAPHS_PATH}")
        validate_all_graphs(GRAPHS_PATH, data_loader)


if __name__ == '__main__':
    main()

