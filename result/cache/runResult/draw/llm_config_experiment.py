#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM配置实验：验证不同LLM配置对多智体协作效果的影响
对比四种配置方案：
1. 单一强模型（GLM4.6）
2. 单一推理模型（GLM-Z1）
3. 单一轻量模型（GLM-4-Flash）
4. 复合配置（根据任务类型分配不同模型）
"""

import os
import sys
import json
import shutil
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
runresult_dir = os.path.dirname(current_dir)
cache_dir = os.path.dirname(runresult_dir)
result_dir = os.path.dirname(cache_dir)
code_dir = os.path.dirname(result_dir)
sys.path.insert(0, code_dir)

from evaluate.unified_evaluator import UnifiedEvaluator

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置
RUNRESULT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL1_DIR = os.path.join(RUNRESULT_DIR, 'model1')
MODEL2_DIR = os.path.join(RUNRESULT_DIR, 'model2')
DRAW_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_PATH = os.path.join(code_dir, 'dataset/processedData/extracted_data/event_graphs_train.json')

# 数据集配置
DATASET = 'suicide_ied'
MAX_GRAPHS = 100

# 四种配置方案（使用model2数据为主，model1补充）
CONFIGS = {
    '方案1-单一强模型': {
        'name': '单一强模型 (GLM4.6)',
        'short_name': 'GLM4.6',
        'color': '#1f77b4',  # 蓝色
        'marker': 'o',
        'dirs': [
            os.path.join(MODEL2_DIR, 'glm4.6', DATASET),      # model2的4.6（22个）
            os.path.join(MODEL1_DIR, 'glm4.6', DATASET),      # model1的4.6补充（20个）
            os.path.join(MODEL2_DIR, 'glm-z1', DATASET),      # z1补充（30个）
            os.path.join(MODEL2_DIR, 'glm-z11', DATASET),     # z11补充到100（28个）
        ],
        'limits': [None, None, None, MAX_GRAPHS]  # 优先级递减，补充到100
    },
    '方案2-单一推理模型': {
        'name': '单一推理模型 (GLM-Z1)',
        'short_name': 'GLM-Z1',
        'color': '#ff7f0e',  # 橙色
        'marker': 's',
        'dirs': [
            os.path.join(MODEL2_DIR, 'glm-z1', DATASET),      # model2的z1（30个）
            os.path.join(MODEL2_DIR, 'glm-z11', DATASET),     # model2的z11（43个）
            os.path.join(MODEL1_DIR, 'glm-z1', DATASET),      # model1的z1补充（13个）
            os.path.join(MODEL1_DIR, 'glm4.6', DATASET),      # model1的4.6补充到100（14个）
        ],
        'limits': [None, None, None, MAX_GRAPHS]  # z1优先，补充到100
    },
    '方案3-单一轻量模型': {
        'name': '单一轻量模型 (GLM-4-Flash)',
        'short_name': 'GLM-4-Flash',
        'color': '#2ca02c',  # 绿色
        'marker': '^',
        'dirs': [os.path.join(MODEL1_DIR, 'glm-4-flash', DATASET)],
        'limits': [MAX_GRAPHS]  # 限制到100
    },
    '方案4-复合配置': {
        'name': '复合配置 (混合模型)',
        'short_name': '复合配置',
        'color': '#d62728',  # 红色
        'marker': 'D',
        'dirs': [
            os.path.join(MODEL2_DIR, 'glm4.6', DATASET),      # 长上下文任务（22个）
            os.path.join(MODEL2_DIR, 'glm-z1', DATASET),      # 强推理任务（30个）
            os.path.join(MODEL2_DIR, 'glm-z11', DATASET),     # 继续推理任务（43个）
            os.path.join(MODEL1_DIR, 'glm-4-flash', DATASET)  # 简单子任务补充到100（5个）
        ],
        'limits': [None, None, None, MAX_GRAPHS]  # 补充到100
    }
}


def count_graphs_in_dir(dir_path):
    """统计目录中的图文件数量"""
    if not os.path.exists(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if f.endswith('.json') and f.startswith('graph_')])


def get_available_graphs(dir_path, max_count=None):
    """获取目录中可用的图文件列表"""
    if not os.path.exists(dir_path):
        return []
    
    # 获取所有图文件
    graph_files = [f for f in os.listdir(dir_path) if f.endswith('.json') and f.startswith('graph_')]
    
    # 按图编号排序
    graph_files.sort(key=lambda x: int(x.replace('graph_', '').replace('.json', '')))
    
    # 限制数量
    if max_count is not None:
        graph_files = graph_files[:max_count]
    
    return graph_files


def merge_graphs_for_config(config_key, config_info):
    """为指定配置方案合并图数据"""
    print(f"\n{'='*60}")
    print(f"处理配置：{config_info['name']}")
    print(f"{'='*60}")
    
    dirs = config_info['dirs']
    limits = config_info['limits']
    
    # 统计各目录的图数量
    print(f"\n数据源统计：")
    for i, dir_path in enumerate(dirs):
        count = count_graphs_in_dir(dir_path)
        limit_str = f"(限制: {limits[i]})" if limits[i] else "(不限制)"
        print(f"  - {os.path.basename(os.path.dirname(dir_path))}/{os.path.basename(dir_path)}: {count} 个图 {limit_str}")
    
    # 收集所有需要的图文件
    all_graphs = []
    total_collected = 0
    
    for i, dir_path in enumerate(dirs):
        if total_collected >= MAX_GRAPHS:
            break
            
        # 计算这个目录最多可以取多少个
        remaining = MAX_GRAPHS - total_collected
        max_from_this_dir = remaining if limits[i] is None else min(limits[i], remaining)
        
        graphs = get_available_graphs(dir_path, max_from_this_dir)
        
        if graphs:
            print(f"\n从 {os.path.basename(os.path.dirname(dir_path))} 收集 {len(graphs)} 个图")
            for graph_file in graphs:
                src_path = os.path.join(dir_path, graph_file)
                all_graphs.append(src_path)
                total_collected += 1
                
                if total_collected >= MAX_GRAPHS:
                    break
    
    print(f"\n总共收集: {len(all_graphs)} 个图")
    
    return all_graphs


def run_evaluation(config_key, graph_files):
    """运行评估"""
    print(f"\n开始评估 {config_key}...")
    
    # 创建临时目录存储合并后的图
    temp_dir = os.path.join(DRAW_DIR, 'temp', config_key)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 复制图文件到临时目录（重新编号为1,2,3...）
    for i, src_path in enumerate(graph_files):
        graph_id = i + 1
        dst_path = os.path.join(temp_dir, f'graph_{graph_id}.json')
        shutil.copy2(src_path, dst_path)
    
    # 运行评估
    output_dir = os.path.join(DRAW_DIR, 'evaluation_results', config_key)
    
    try:
        evaluator = UnifiedEvaluator(
            graphs_dir=temp_dir,
            ground_truth_path=GROUND_TRUTH_PATH,
            output_dir=output_dir,
            attack_type=DATASET,
            max_file_count=None  # 不限制，因为我们已经控制了文件数量
        )
        
        # 运行递进式合并评估
        results = evaluator.run_layered_merger_evaluation(
            max_level=min(len(graph_files), MAX_GRAPHS),  # 最多评估到MAX_GRAPHS层
            num_samples=5  # 中间层级的采样数量
        )
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        return results
        
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None


def extract_metrics_from_results(output_dir, config_key):
    """从评估结果中提取指标"""
    print(f"\n提取 {config_key} 的评估指标...")
    
    # 查找最新的xlsx文件
    if not os.path.exists(output_dir):
        print(f"警告: 目录不存在: {output_dir}")
        return None
        
    xlsx_files = [f for f in os.listdir(output_dir) if f.endswith('.xlsx')]
    if not xlsx_files:
        print(f"警告: 未找到xlsx文件")
        return None
    
    xlsx_files.sort()
    latest_xlsx = xlsx_files[-1]
    xlsx_path = os.path.join(output_dir, latest_xlsx)
    
    print(f"读取: {latest_xlsx}")
    
    try:
        # 先获取所有工作表名称
        xlsx_file = pd.ExcelFile(xlsx_path)
        sheet_names = xlsx_file.sheet_names
        print(f"可用工作表: {sheet_names}")
        
        # 查找包含"Type.Subtype"的工作表
        event_type_sheet = None
        seq3_sheet = None
        
        for sheet in sheet_names:
            if 'Type.Subtype' in sheet and '事件类型' in sheet:
                event_type_sheet = sheet
            if 'Type.Subtype' in sheet and '序列3' in sheet:
                seq3_sheet = sheet
        
        if not event_type_sheet:
            print(f"警告: 未找到事件类型工作表")
            return None
        
        # 读取事件类型匹配结果（用于F1_EM）
        df_event = pd.read_excel(xlsx_path, sheet_name=event_type_sheet)
        
        results = {
            'ndoc': df_event['层级'].values,
            'f1_em': df_event['F1'].values,
            'precision': df_event['Precision'].values,
            'recall': df_event['Recall'].values
        }
        
        # 读取序列3匹配结果（用于F1_ESM）
        if seq3_sheet:
            df_seq3 = pd.read_excel(xlsx_path, sheet_name=seq3_sheet)
            results['f1_esm_l23'] = df_seq3['F1'].values
            print(f"成功读取序列3数据")
        else:
            print(f"警告: 未找到序列3工作表")
        
        print(f"成功提取 {len(results['ndoc'])} 个数据点")
        return results
        
    except Exception as e:
        print(f"读取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_comparison(all_results):
    """绘制对比图"""
    print(f"\n绘制对比图...")
    
    # 创建两个子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1: F1_EM vs Ndoc
    ax1 = axes[0]
    for config_key in ['方案1-单一强模型', '方案2-单一推理模型', '方案3-单一轻量模型', '方案4-复合配置']:
        if config_key not in all_results or all_results[config_key] is None:
            continue
            
        config_info = CONFIGS[config_key]
        results = all_results[config_key]
        
        x = results['ndoc']
        y = results['f1_em']
        
        ax1.plot(x, y, 
                label=config_info['short_name'],
                color=config_info['color'],
                marker=config_info['marker'],
                linewidth=2.5,
                markersize=6,
                markevery=max(len(x)//15, 1),
                alpha=0.8)
        
        # 标注最高点
        max_idx = np.argmax(y)
        max_x = x[max_idx]
        max_y = y[max_idx]
        ax1.scatter([max_x], [max_y], 
                   color=config_info['color'], 
                   s=150, 
                   zorder=5, 
                   edgecolors='black', 
                   linewidths=2)
    
    ax1.set_xlabel('图数量 $N_{doc}$', fontsize=14, fontweight='bold')
    ax1.set_ylabel('$F1_{EM}$', fontsize=14, fontweight='bold')
    ax1.set_title('事件类型识别性能对比', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='best', fontsize=11, framealpha=0.95)
    ax1.set_xlim(0, MAX_GRAPHS + 10)
    
    # 图2: F1_ESM(L=2,3) vs Ndoc
    ax2 = axes[1]
    for config_key in ['方案1-单一强模型', '方案2-单一推理模型', '方案3-单一轻量模型', '方案4-复合配置']:
        if config_key not in all_results or all_results[config_key] is None:
            continue
            
        config_info = CONFIGS[config_key]
        results = all_results[config_key]
        
        # 使用序列3的F1分数
        if 'f1_esm_l23' in results:
            y = results['f1_esm_l23']
            ylabel = '$F1_{ESM}$ (L=2,3)'
        else:
            # 如果没有序列数据，使用事件类型匹配
            y = results['f1_em']
            ylabel = '$F1_{EM}$'
        
        x = results['ndoc']
        
        ax2.plot(x, y,
                label=config_info['short_name'],
                color=config_info['color'],
                marker=config_info['marker'],
                linewidth=2.5,
                markersize=6,
                markevery=max(len(x)//15, 1),
                alpha=0.8)
        
        # 标注最高点
        max_idx = np.argmax(y)
        max_x = x[max_idx]
        max_y = y[max_idx]
        ax2.scatter([max_x], [max_y],
                   color=config_info['color'],
                   s=150,
                   zorder=5,
                   edgecolors='black',
                   linewidths=2)
    
    ax2.set_xlabel('图数量 $N_{doc}$', fontsize=14, fontweight='bold')
    ax2.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax2.set_title('事件链识别性能对比', fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    ax2.set_xlim(0, MAX_GRAPHS + 10)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path_png = os.path.join(DRAW_DIR, f'llm_config_comparison_{timestamp}.png')
    output_path_pdf = os.path.join(DRAW_DIR, f'llm_config_comparison_{timestamp}.pdf')
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG图片已保存: {output_path_png}")
    
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"PDF图片已保存: {output_path_pdf}")
    
    plt.close()
    
    return output_path_png, output_path_pdf


def generate_summary_table(all_results):
    """生成汇总表格"""
    print(f"\n生成汇总表格...")
    
    summary_data = []
    
    for config_key in ['方案1-单一强模型', '方案2-单一推理模型', '方案3-单一轻量模型', '方案4-复合配置']:
        if config_key not in all_results or all_results[config_key] is None:
            continue
            
        config_info = CONFIGS[config_key]
        results = all_results[config_key]
        
        # 获取最大值
        max_f1_em = np.max(results['f1_em'])
        max_f1_em_ndoc = results['ndoc'][np.argmax(results['f1_em'])]
        
        # 获取最终值（150个图时）
        final_idx = -1
        final_f1_em = results['f1_em'][final_idx]
        final_ndoc = results['ndoc'][final_idx]
        
        summary_data.append({
            '配置方案': config_info['short_name'],
            '图数量': len(results['ndoc']),
            '最高F1_EM': f"{max_f1_em:.4f}",
            '最高点图数': max_f1_em_ndoc,
            '最终F1_EM': f"{final_f1_em:.4f}",
            '最终图数': final_ndoc
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # 保存到Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(DRAW_DIR, f'llm_config_summary_{timestamp}.xlsx')
    df_summary.to_excel(output_path, index=False)
    
    print(f"汇总表格已保存: {output_path}")
    print(f"\n{df_summary.to_string(index=False)}")
    
    return output_path


def main():
    """主函数"""
    print("="*80)
    print("LLM配置实验：验证不同LLM配置对多智体协作效果的影响")
    print("="*80)
    
    print(f"\n实验配置:")
    print(f"  - 数据集: {DATASET}")
    print(f"  - 最大图数量: {MAX_GRAPHS}")
    print(f"  - Ground Truth: {GROUND_TRUTH_PATH}")
    
    # 存储所有结果
    all_results = {}
    
    # 对每个配置方案进行评估
    for config_key, config_info in CONFIGS.items():
        # 1. 合并图数据
        graph_files = merge_graphs_for_config(config_key, config_info)
        
        if not graph_files:
            print(f"警告: {config_key} 没有可用的图文件")
            continue
        
        # 2. 运行评估
        run_evaluation(config_key, graph_files)
        
        # 3. 提取指标
        output_dir = os.path.join(DRAW_DIR, 'evaluation_results', config_key)
        results = extract_metrics_from_results(output_dir, config_key)
        
        if results is not None:
            all_results[config_key] = results
    
    # 4. 绘制对比图
    if all_results:
        plot_comparison(all_results)
        generate_summary_table(all_results)
        
        print("\n" + "="*80)
        print("实验完成！")
        print("="*80)
    else:
        print("\n警告: 没有成功的评估结果")


if __name__ == '__main__':
    main()

