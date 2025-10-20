#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的数据集和结果统计脚本
包括缓存结果、正式结果、评测结果等
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
CACHE_ROOT = RESULT_ROOT / 'cache'


def count_files_recursive(directory, pattern='*.json'):
    """递归统计指定模式的文件数量"""
    if not directory.exists():
        return 0
    return len(list(directory.rglob(pattern)))


def count_text_files(attack_type_dir):
    """统计文本文件数量"""
    if not attack_type_dir.exists():
        return 0
    return len(list(attack_type_dir.rglob('*.txt')))


def analyze_result_directory(result_dir):
    """分析结果目录"""
    result = {
        'individual_graphs': 0,
        'merged_graphs': 0,
        'by_attack_type': defaultdict(lambda: {'individual': 0, 'merged': 0})
    }
    
    if not result_dir.exists():
        return result
    
    # 遍历攻击类型目录
    for attack_dir in result_dir.iterdir():
        if attack_dir.is_dir():
            attack_type = attack_dir.name
            for file in attack_dir.iterdir():
                if file.is_file() and file.suffix == '.json':
                    if file.name.startswith('merged_graph_'):
                        result['merged_graphs'] += 1
                        result['by_attack_type'][attack_type]['merged'] += 1
                    elif file.name.startswith('graph_'):
                        result['individual_graphs'] += 1
                        result['by_attack_type'][attack_type]['individual'] += 1
    
    return result


def analyze_cache_results():
    """分析缓存的运行结果"""
    cache_run_result = CACHE_ROOT / 'runResult'
    result = {
        'model1': {},
        'model2': {}
    }
    
    if not cache_run_result.exists():
        return result
    
    # 分析 Model1
    model1_dir = cache_run_result / 'model1'
    if model1_dir.exists():
        for model_version_dir in model1_dir.iterdir():
            if model_version_dir.is_dir():
                model_version = model_version_dir.name
                result['model1'][model_version] = {}
                
                for attack_dir in model_version_dir.iterdir():
                    if attack_dir.is_dir():
                        attack_type = attack_dir.name
                        count = count_files_recursive(attack_dir, '*.json')
                        result['model1'][model_version][attack_type] = count
    
    # 分析 Model2
    model2_dir = cache_run_result / 'model2'
    if model2_dir.exists():
        for model_version_dir in model2_dir.iterdir():
            if model_version_dir.is_dir():
                model_version = model_version_dir.name
                result['model2'][model_version] = {}
                
                for attack_dir in model_version_dir.iterdir():
                    if attack_dir.is_dir():
                        attack_type = attack_dir.name
                        count = count_files_recursive(attack_dir, '*.json')
                        result['model2'][model_version][attack_type] = count
    
    return result


def load_ground_truth_statistics():
    """加载 ground truth 统计信息"""
    stats_file = EXTRACTED_DATA_PATH / 'statistics.json'
    if not stats_file.exists():
        return None
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_evaluation_results():
    """分析评测结果"""
    result = {
        'unified': [],
        'layered': []
    }
    
    # 统一评测结果
    unified_dir = RESULT_ROOT / 'unified_evaluation'
    if unified_dir.exists():
        result['unified'] = sorted([
            f.name for f in unified_dir.iterdir()
            if f.is_file() and f.suffix == '.json'
        ])
    
    # 递进式评测结果
    layered_dir = RESULT_ROOT / 'layered_evaluation'
    if layered_dir.exists():
        result['layered'] = sorted([
            f.name for f in layered_dir.iterdir()
            if f.is_file() and f.suffix == '.json'
        ])
    
    return result


def main():
    """主函数"""
    print("=" * 100)
    print("完整数据集和结果统计报告")
    print("=" * 100)
    print()
    
    # 1. 数据集统计
    print("【1. 数据集统计】")
    print("-" * 100)
    
    attack_types = []
    if EXTRACTED_TEXTS_PATH.exists():
        for item in EXTRACTED_TEXTS_PATH.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                attack_types.append(item.name)
    attack_types.sort()
    
    gt_stats = load_ground_truth_statistics()
    
    total_text_files = 0
    for attack_type in attack_types:
        text_dir = EXTRACTED_TEXTS_PATH / attack_type
        text_count = count_text_files(text_dir)
        total_text_files += text_count
        
        gt_info = ""
        if gt_stats:
            total_instances = 0
            for split in ['train', 'dev', 'test']:
                instances = gt_stats.get(split, {}).get(attack_type, {}).get('num_instances', 0)
                total_instances += instances
            if total_instances > 0:
                gt_info = f" (Ground Truth: {total_instances} 个标注实例)"
        
        print(f"  {attack_type}: {text_count} 个文本文件{gt_info}")
    
    print(f"\n  总计: {len(attack_types)} 种攻击类型, {total_text_files} 个文本文件")
    
    # 2. 缓存的运行结果统计
    print("\n【2. 缓存的运行结果统计 (cache/runResult)】")
    print("-" * 100)
    
    cache_results = analyze_cache_results()
    
    print("\nModel1 缓存结果:")
    for model_version, attacks in cache_results['model1'].items():
        print(f"  {model_version}:")
        for attack_type, count in attacks.items():
            print(f"    - {attack_type}: {count} 个图文件")
    
    if not cache_results['model1']:
        print("  无缓存结果")
    
    print("\nModel2 缓存结果:")
    for model_version, attacks in cache_results['model2'].items():
        print(f"  {model_version}:")
        for attack_type, count in attacks.items():
            print(f"    - {attack_type}: {count} 个图文件")
    
    if not cache_results['model2']:
        print("  无缓存结果")
    
    # 3. 正式结果统计
    print("\n【3. 正式结果统计 (result/model*)】")
    print("-" * 100)
    
    model1_results = analyze_result_directory(RESULT_ROOT / 'model1')
    model2_results = analyze_result_directory(RESULT_ROOT / 'model2')
    
    print("\nModel1 正式结果:")
    if model1_results['individual_graphs'] > 0 or model1_results['merged_graphs'] > 0:
        print(f"  总计: {model1_results['individual_graphs']} 个单图, {model1_results['merged_graphs']} 个融合图")
        for attack_type, counts in model1_results['by_attack_type'].items():
            print(f"    - {attack_type}: {counts['individual']} 个单图, {counts['merged']} 个融合图")
    else:
        print("  无正式结果")
    
    print("\nModel2 正式结果:")
    if model2_results['individual_graphs'] > 0 or model2_results['merged_graphs'] > 0:
        print(f"  总计: {model2_results['individual_graphs']} 个单图, {model2_results['merged_graphs']} 个融合图")
        for attack_type, counts in model2_results['by_attack_type'].items():
            print(f"    - {attack_type}: {counts['individual']} 个单图, {counts['merged']} 个融合图")
    else:
        print("  无正式结果")
    
    # 4. 评测结果统计
    print("\n【4. 评测结果统计】")
    print("-" * 100)
    
    eval_results = analyze_evaluation_results()
    
    print(f"\n统一评测结果: {len(eval_results['unified'])} 个文件")
    for filename in eval_results['unified'][:5]:  # 只显示前5个
        print(f"  - {filename}")
    if len(eval_results['unified']) > 5:
        print(f"  ... 还有 {len(eval_results['unified']) - 5} 个文件")
    
    print(f"\n递进式评测结果: {len(eval_results['layered'])} 个文件")
    for filename in eval_results['layered'][:5]:  # 只显示前5个
        print(f"  - {filename}")
    if len(eval_results['layered']) > 5:
        print(f"  ... 还有 {len(eval_results['layered']) - 5} 个文件")
    
    # 5. 论文表格
    print("\n" + "=" * 100)
    print("【5. 论文表格格式 (Table 1: 数据集核心统计信息)】")
    print("-" * 100)
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
                text_dir = EXTRACTED_TEXTS_PATH / attack_type
                text_count = count_text_files(text_dir)
                
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
            count_text_files(EXTRACTED_TEXTS_PATH / at)
            for at in paper_mapping.keys()
            if (EXTRACTED_TEXTS_PATH / at).exists()
        )
        
        print(f"| **总计** | {total_train} | {total_dev} | {total_test} | {total_all} | {total_avg_events:.2f} | {total_avg_entities:.2f} | {total_avg_relations:.2f} | {total_text_files} |")
    
    # 6. 实验运行摘要
    print("\n" + "=" * 100)
    print("【6. 实验运行摘要】")
    print("-" * 100)
    print()
    
    # 统计所有运行过的模型和攻击类型组合
    experiments = defaultdict(set)
    
    # 从缓存结果中统计
    for model_version, attacks in cache_results['model1'].items():
        for attack_type in attacks.keys():
            experiments['model1'].add((attack_type, model_version))
    
    for model_version, attacks in cache_results['model2'].items():
        for attack_type in attacks.keys():
            experiments['model2'].add((attack_type, model_version))
    
    # 从正式结果中统计
    for attack_type in model1_results['by_attack_type'].keys():
        experiments['model1'].add((attack_type, 'unknown'))
    
    for attack_type in model2_results['by_attack_type'].keys():
        experiments['model2'].add((attack_type, 'unknown'))
    
    print("已运行的实验:")
    print("\nModel1:")
    if experiments['model1']:
        attack_model_map = defaultdict(set)
        for attack, model in experiments['model1']:
            attack_model_map[attack].add(model)
        for attack in sorted(attack_model_map.keys()):
            models = sorted(attack_model_map[attack])
            print(f"  - {attack}: {', '.join(models)}")
    else:
        print("  无运行记录")
    
    print("\nModel2:")
    if experiments['model2']:
        attack_model_map = defaultdict(set)
        for attack, model in experiments['model2']:
            attack_model_map[attack].add(model)
        for attack in sorted(attack_model_map.keys()):
            models = sorted(attack_model_map[attack])
            print(f"  - {attack}: {', '.join(models)}")
    else:
        print("  无运行记录")
    
    print("\n" + "=" * 100)
    print("统计完成！")
    print("=" * 100)


if __name__ == '__main__':
    main()

