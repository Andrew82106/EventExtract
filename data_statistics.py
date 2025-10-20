#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集和结果统计脚本
用于统计项目中实际使用的数据情况和生成的结果
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 数据路径
DATASET_ROOT = ROOT_DIR / 'dataset'
EXTRACTED_TEXTS_PATH = DATASET_ROOT / 'processedData' / 'extracted_texts'
EXTRACTED_DATA_PATH = DATASET_ROOT / 'processedData' / 'extracted_data'
RESULT_ROOT = ROOT_DIR / 'result'


def count_files_in_directory(directory):
    """统计目录中的文件数量"""
    if not directory.exists():
        return 0
    
    count = 0
    for item in directory.rglob('*'):
        if item.is_file():
            count += 1
    return count


def count_text_files(attack_type_dir):
    """统计文本文件数量"""
    if not attack_type_dir.exists():
        return 0
    
    txt_files = list(attack_type_dir.rglob('*.txt'))
    return len(txt_files)


def count_graph_files(result_dir):
    """统计生成的图文件数量"""
    if not result_dir.exists():
        return {'individual': 0, 'merged': 0}
    
    individual = 0
    merged = 0
    
    for file in result_dir.iterdir():
        if file.is_file() and file.suffix == '.json':
            if file.name.startswith('merged_graph_'):
                merged += 1
            elif file.name.startswith('graph_'):
                individual += 1
    
    return {'individual': individual, 'merged': merged}


def load_ground_truth_statistics():
    """加载 ground truth 统计信息"""
    stats_file = EXTRACTED_DATA_PATH / 'statistics.json'
    if not stats_file.exists():
        return None
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_attack_type(attack_type):
    """分析单个攻击类型的数据"""
    result = {
        'attack_type': attack_type,
        'text_files': 0,
        'ground_truth': {},
        'model1_results': {},
        'model2_results': {}
    }
    
    # 统计文本文件
    text_dir = EXTRACTED_TEXTS_PATH / attack_type
    result['text_files'] = count_text_files(text_dir)
    
    # 统计 Model1 结果
    model1_dir = RESULT_ROOT / 'model1' / attack_type
    result['model1_results'] = count_graph_files(model1_dir)
    
    # 统计 Model2 结果
    model2_dir = RESULT_ROOT / 'model2' / attack_type
    result['model2_results'] = count_graph_files(model2_dir)
    
    return result


def main():
    """主函数"""
    print("=" * 80)
    print("数据集和结果统计报告")
    print("=" * 80)
    print()
    
    # 获取所有攻击类型
    attack_types = []
    if EXTRACTED_TEXTS_PATH.exists():
        for item in EXTRACTED_TEXTS_PATH.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                attack_types.append(item.name)
    
    attack_types.sort()
    
    # 加载 ground truth 统计
    gt_stats = load_ground_truth_statistics()
    
    # 统计每个攻击类型
    all_results = []
    for attack_type in attack_types:
        result = analyze_attack_type(attack_type)
        
        # 添加 ground truth 信息
        if gt_stats:
            for split in ['train', 'dev', 'test']:
                if attack_type in gt_stats.get(split, {}):
                    result['ground_truth'][split] = gt_stats[split][attack_type]
        
        all_results.append(result)
    
    # 打印详细统计
    print("攻击类型详细统计:")
    print("-" * 80)
    
    for result in all_results:
        print(f"\n【{result['attack_type']}】")
        print(f"  文本文件数: {result['text_files']}")
        
        # Ground Truth
        if result['ground_truth']:
            print(f"  Ground Truth:")
            total_instances = 0
            for split, stats in result['ground_truth'].items():
                instances = stats.get('num_instances', 0)
                total_instances += instances
                avg_events = stats.get('avg_events', 0)
                print(f"    - {split}: {instances} 个实例, 平均 {avg_events:.1f} 个事件")
            print(f"    总计: {total_instances} 个实例")
        
        # Model1 结果
        m1 = result['model1_results']
        if m1['individual'] > 0 or m1['merged'] > 0:
            print(f"  Model1 结果:")
            print(f"    - 单个图: {m1['individual']} 个")
            print(f"    - 融合图: {m1['merged']} 个")
        else:
            print(f"  Model1 结果: 无")
        
        # Model2 结果
        m2 = result['model2_results']
        if m2['individual'] > 0 or m2['merged'] > 0:
            print(f"  Model2 结果:")
            print(f"    - 单个图: {m2['individual']} 个")
            print(f"    - 融合图: {m2['merged']} 个")
        else:
            print(f"  Model2 结果: 无")
    
    # 打印总体统计
    print("\n" + "=" * 80)
    print("总体统计:")
    print("-" * 80)
    
    total_texts = sum(r['text_files'] for r in all_results)
    total_m1_graphs = sum(r['model1_results']['individual'] for r in all_results)
    total_m1_merged = sum(r['model1_results']['merged'] for r in all_results)
    total_m2_graphs = sum(r['model2_results']['individual'] for r in all_results)
    total_m2_merged = sum(r['model2_results']['merged'] for r in all_results)
    
    print(f"攻击类型总数: {len(attack_types)}")
    print(f"文本文件总数: {total_texts}")
    print(f"Model1 生成的单个图: {total_m1_graphs}")
    print(f"Model1 生成的融合图: {total_m1_merged}")
    print(f"Model2 生成的单个图: {total_m2_graphs}")
    print(f"Model2 生成的融合图: {total_m2_merged}")
    
    # 统计 ground truth 总数
    if gt_stats:
        print("\nGround Truth 总计:")
        for split in ['train', 'dev', 'test']:
            total = sum(
                gt_stats.get(split, {}).get(at, {}).get('num_instances', 0)
                for at in attack_types
            )
            print(f"  - {split}: {total} 个实例")
    
    # 打印表格（用于论文）
    print("\n" + "=" * 80)
    print("论文表格格式（Table 1: 数据集核心统计信息）:")
    print("-" * 80)
    print()
    
    # 映射到论文中的名称
    paper_mapping = {
        'suicide_ied': ('Suicide-IED', '自杀式简易爆炸装置'),
        'wiki_mass_car_bombings': ('Car-IED', '车载简易爆炸装置'),
        'wiki_ied_bombings': ('General-IED', '通用简易爆炸装置'),
    }
    
    if gt_stats:
        print("| 事件类型 | 训练集 | 验证集 | 测试集 | 总计 | 平均事件数 | 平均实体数 | 平均时序关系数 | 文本文件数 |")
        print("|---------|-------|-------|-------|------|----------|----------|--------------|----------|")
        
        for attack_type, (en_name, cn_name) in paper_mapping.items():
            if attack_type in attack_types:
                train_data = gt_stats.get('train', {}).get(attack_type, {})
                dev_data = gt_stats.get('dev', {}).get(attack_type, {})
                test_data = gt_stats.get('test', {}).get(attack_type, {})
                
                train_num = train_data.get('num_instances', 0)
                dev_num = dev_data.get('num_instances', 0)
                test_num = test_data.get('num_instances', 0)
                total_num = train_num + dev_num + test_num
                
                # 计算加权平均
                total_events = (
                    train_data.get('avg_events', 0) * train_num +
                    dev_data.get('avg_events', 0) * dev_num +
                    test_data.get('avg_events', 0) * test_num
                ) / total_num if total_num > 0 else 0
                
                total_entities = (
                    train_data.get('avg_entities', 0) * train_num +
                    dev_data.get('avg_entities', 0) * dev_num +
                    test_data.get('avg_entities', 0) * test_num
                ) / total_num if total_num > 0 else 0
                
                total_relations = (
                    train_data.get('avg_temporal_relations', 0) * train_num +
                    dev_data.get('avg_temporal_relations', 0) * dev_num +
                    test_data.get('avg_temporal_relations', 0) * test_num
                ) / total_num if total_num > 0 else 0
                
                # 获取文本文件数
                text_count = next(
                    (r['text_files'] for r in all_results if r['attack_type'] == attack_type),
                    0
                )
                
                print(f"| {en_name}<br>({cn_name}) | {train_num} | {dev_num} | {test_num} | {total_num} | {total_events:.2f} | {total_entities:.2f} | {total_relations:.2f} | {text_count} |")
        
        # 计算总计
        total_train = sum(
            gt_stats.get('train', {}).get(at, {}).get('num_instances', 0)
            for at in paper_mapping.keys()
        )
        total_dev = sum(
            gt_stats.get('dev', {}).get(at, {}).get('num_instances', 0)
            for at in paper_mapping.keys()
        )
        total_test = sum(
            gt_stats.get('test', {}).get(at, {}).get('num_instances', 0)
            for at in paper_mapping.keys()
        )
        total_all = total_train + total_dev + total_test
        
        # 计算加权平均（所有三种类型）
        total_avg_events = 0
        total_avg_entities = 0
        total_avg_relations = 0
        
        for attack_type in paper_mapping.keys():
            for split in ['train', 'dev', 'test']:
                data = gt_stats.get(split, {}).get(attack_type, {})
                num = data.get('num_instances', 0)
                total_avg_events += data.get('avg_events', 0) * num
                total_avg_entities += data.get('avg_entities', 0) * num
                total_avg_relations += data.get('avg_temporal_relations', 0) * num
        
        if total_all > 0:
            total_avg_events /= total_all
            total_avg_entities /= total_all
            total_avg_relations /= total_all
        
        total_text_files = sum(
            r['text_files'] for r in all_results
            if r['attack_type'] in paper_mapping.keys()
        )
        
        print(f"| **总计** | {total_train} | {total_dev} | {total_test} | {total_all} | {total_avg_events:.2f} | {total_avg_entities:.2f} | {total_avg_relations:.2f} | {total_text_files} |")
    
    print("\n" + "=" * 80)
    print("统计完成！")
    print("=" * 80)
    
    # 保存到文件
    output_file = ROOT_DIR / 'data_statistics_report.txt'
    print(f"\n报告已保存到: {output_file}")


if __name__ == '__main__':
    main()

