#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制LLM配置实验对比图
从已生成的XLSX文件中提取数据并绘制
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置
DRAW_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_DIR = os.path.join(os.path.dirname(DRAW_DIR), 'xlsx')

# XLSX文件映射（最新生成的文件 - 限制100个图）
XLSX_FILES = {
    '方案1-单一强模型': 'layered_merger_suicide_ied_unknown_20251021_122122.xlsx',
    '方案2-单一推理模型': 'layered_merger_suicide_ied_unknown_20251021_122215.xlsx',
    '方案3-单一轻量模型': 'layered_merger_suicide_ied_unknown_20251021_122311.xlsx',
    '方案4-复合配置': 'layered_merger_suicide_ied_unknown_20251021_122408.xlsx',
}

# 配置信息
CONFIGS = {
    '方案1-单一强模型': {
        'name': '单一强模型 (GLM4.6)',
        'short_name': 'GLM4.6',
        'color': 'black',
        'marker': 'o',
        'linestyle': '-'
    },
    '方案2-单一推理模型': {
        'name': '单一推理模型 (GLM-Z1)',
        'short_name': 'GLM-Z1',
        'color': 'black',
        'marker': 's',
        'linestyle': '--'
    },
    '方案3-单一轻量模型': {
        'name': '单一轻量模型 (GLM-4-Flash)',
        'short_name': 'GLM-4-Flash',
        'color': 'black',
        'marker': '^',
        'linestyle': '-.'
    },
    '方案4-复合配置': {
        'name': '复合配置 (混合模型)',
        'short_name': '复合配置',
        'color': 'black',
        'marker': 'D',
        'linestyle': ':'
    }
}


def extract_metrics(xlsx_file):
    """从XLSX文件中提取指标"""
    xlsx_path = os.path.join(XLSX_DIR, xlsx_file)
    
    if not os.path.exists(xlsx_path):
        print(f"警告: 文件不存在: {xlsx_path}")
        return None
    
    try:
        # 获取所有工作表
        xlsx = pd.ExcelFile(xlsx_path)
        sheet_names = xlsx.sheet_names
        
        # 查找Type.Subtype相关的工作表
        event_type_sheet = None
        seq2_sheet = None
        seq3_sheet = None
        
        for sheet in sheet_names:
            if 'Type.Subtype' in sheet and '事件类型' in sheet:
                event_type_sheet = sheet
            if 'Type.Subtype' in sheet and '序列2' in sheet:
                seq2_sheet = sheet
            if 'Type.Subtype' in sheet and '序列3' in sheet:
                seq3_sheet = sheet
        
        if not event_type_sheet:
            print(f"警告: 未找到事件类型工作表")
            return None
        
        # 读取数据
        df_event = pd.read_excel(xlsx_path, sheet_name=event_type_sheet)
        
        results = {
            'ndoc': df_event['层级'].values,
            'f1_em': df_event['F1'].values,
            'precision': df_event['Precision'].values,
            'recall': df_event['Recall'].values
        }
        
        # 读取序列2数据
        if seq2_sheet:
            df_seq2 = pd.read_excel(xlsx_path, sheet_name=seq2_sheet)
            results['f1_esm_l2'] = df_seq2['F1'].values
        
        # 读取序列3数据
        if seq3_sheet:
            df_seq3 = pd.read_excel(xlsx_path, sheet_name=seq3_sheet)
            results['f1_esm_l3'] = df_seq3['F1'].values
        
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
    
    # 创建2x2布局的四个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # 展平为一维数组方便索引
    
    # 图1: F1_EM vs Ndoc（事件类型识别）- 左上
    ax1 = axes[0]
    for config_key in ['方案1-单一强模型', '方案2-单一推理模型', '方案3-单一轻量模型', '方案4-复合配置']:
        if config_key not in all_results or all_results[config_key] is None:
            continue
            
        config_info = CONFIGS[config_key]
        results = all_results[config_key]
        
        x = results['ndoc']
        y_raw = results['f1_em']
        
        # 平滑处理
        y = pd.Series(y_raw).rolling(window=10, center=True, min_periods=1).mean().values
        
        # 绘制平滑曲线
        ax1.plot(x, y, 
                label=config_info['short_name'],
                color=config_info['color'],
                linestyle=config_info['linestyle'],
                linewidth=2.0,
                alpha=1.0)
        
        # 稀疏标记
        ax1.plot(x, y,
                color=config_info['color'],
                linestyle='',
                marker=config_info['marker'],
                markersize=6,
                markevery=10,
                alpha=1.0)
    
    ax1.set_xlabel('图数量 $N_{doc}$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$F1_{EM}$', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 事件类型识别', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    # 图2: F1_ESM (L=2) - 右上
    ax2 = axes[1]
    for config_key in ['方案1-单一强模型', '方案2-单一推理模型', '方案3-单一轻量模型', '方案4-复合配置']:
        if config_key not in all_results or all_results[config_key] is None:
            continue
            
        config_info = CONFIGS[config_key]
        results = all_results[config_key]
        
        x = results['ndoc']
        
        if 'f1_esm_l2' in results:
            y_raw = results['f1_esm_l2']
            # 平滑处理
            y = pd.Series(y_raw).rolling(window=10, center=True, min_periods=1).mean().values
            
            # 绘制平滑曲线
            ax2.plot(x, y,
                    label=config_info['short_name'],
                    color=config_info['color'],
                    linestyle=config_info['linestyle'],
                    linewidth=2.0,
                    alpha=1.0)
                    
            # 稀疏标记
            ax2.plot(x, y,
                    color=config_info['color'],
                    linestyle='',
                    marker=config_info['marker'],
                    markersize=6,
                    markevery=10,
                    alpha=1.0)
    
    ax2.set_xlabel('图数量 $N_{doc}$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$F1_{ESM}$ (L=2)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 事件链识别 (L=2)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    # 图3: F1_ESM (L=3) - 左下
    ax3 = axes[2]
    for config_key in ['方案1-单一强模型', '方案2-单一推理模型', '方案3-单一轻量模型', '方案4-复合配置']:
        if config_key not in all_results or all_results[config_key] is None:
            continue
            
        config_info = CONFIGS[config_key]
        results = all_results[config_key]
        
        x = results['ndoc']
        
        if 'f1_esm_l3' in results:
            y_raw = results['f1_esm_l3']
            # 平滑处理
            y = pd.Series(y_raw).rolling(window=10, center=True, min_periods=1).mean().values
            
            # 绘制平滑曲线
            ax3.plot(x, y,
                    label=config_info['short_name'],
                    color=config_info['color'],
                    linestyle=config_info['linestyle'],
                    linewidth=2.0,
                    alpha=1.0)
                    
            # 稀疏标记
            ax3.plot(x, y,
                    color=config_info['color'],
                    linestyle='',
                    marker=config_info['marker'],
                    markersize=6,
                    markevery=10,
                    alpha=1.0)
    
    ax3.set_xlabel('图数量 $N_{doc}$', fontsize=12, fontweight='bold')
    ax3.set_ylabel('$F1_{ESM}$ (L=3)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) 事件链识别 (L=3)', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    # 图4: 性能提升率（边际效益）- 右下
    ax4 = axes[3]
    for config_key in ['方案1-单一强模型', '方案2-单一推理模型', '方案3-单一轻量模型', '方案4-复合配置']:
        if config_key not in all_results or all_results[config_key] is None:
            continue
            
        config_info = CONFIGS[config_key]
        results = all_results[config_key]
        
        x = results['ndoc']
        y = results['f1_em']
        
        # 计算性能增长率（使用移动窗口平滑）
        # 这里本身已经是导数，可以不用再平滑，或者轻微平滑
        window_size = 5
        growth_rates = []
        x_growth = []
        
        for i in range(window_size, len(x)):
            delta_f1 = y[i] - y[i-window_size]
            delta_ndoc = x[i] - x[i-window_size]
            if delta_ndoc > 0:
                growth_rate = delta_f1 / delta_ndoc
                growth_rates.append(growth_rate)
                x_growth.append(x[i])
        
        if growth_rates:
            # 对增长率也进行轻微平滑
            y_growth = pd.Series(growth_rates).rolling(window=5, center=True, min_periods=1).mean().values
            
            # 绘制平滑曲线
            ax4.plot(x_growth, y_growth,
                    label=config_info['short_name'],
                    color=config_info['color'],
                    linestyle=config_info['linestyle'],
                    linewidth=2.0,
                    alpha=1.0)
            
            # 稀疏标记
            ax4.plot(x_growth, y_growth,
                    color=config_info['color'],
                    linestyle='',
                    marker=config_info['marker'],
                    markersize=6,
                    markevery=10,
                    alpha=1.0)
    
    ax4.set_xlabel('图数量 $N_{doc}$', fontsize=12, fontweight='bold')
    ax4.set_ylabel('$\\Delta F1_{EM} / \\Delta N_{doc}$', fontsize=12, fontweight='bold')
    ax4.set_title('(d) 性能边际效益', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图片（不带时间戳，直接覆盖）
    output_path_png = os.path.join(DRAW_DIR, 'llm_config_comparison.png')
    output_path_pdf = os.path.join(DRAW_DIR, 'llm_config_comparison.pdf')
    
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
        
        # 获取最终值
        final_idx = -1
        final_f1_em = results['f1_em'][final_idx]
        final_ndoc = results['ndoc'][final_idx]
        
        # ESM L=2指标
        if 'f1_esm_l2' in results:
            max_f1_esm_l2 = np.max(results['f1_esm_l2'])
            max_f1_esm_l2_ndoc = results['ndoc'][np.argmax(results['f1_esm_l2'])]
            final_f1_esm_l2 = results['f1_esm_l2'][final_idx]
        else:
            max_f1_esm_l2 = None
            max_f1_esm_l2_ndoc = None
            final_f1_esm_l2 = None
        
        # ESM L=3指标
        if 'f1_esm_l3' in results:
            max_f1_esm_l3 = np.max(results['f1_esm_l3'])
            max_f1_esm_l3_ndoc = results['ndoc'][np.argmax(results['f1_esm_l3'])]
            final_f1_esm_l3 = results['f1_esm_l3'][final_idx]
        else:
            max_f1_esm_l3 = None
            max_f1_esm_l3_ndoc = None
            final_f1_esm_l3 = None
        
        summary_data.append({
            '配置方案': config_info['short_name'],
            '总图数': len(results['ndoc']),
            '最高F1_EM': f"{max_f1_em:.4f}",
            '最高F1_EM图数': int(max_f1_em_ndoc),
            '最终F1_EM': f"{final_f1_em:.4f}",
            '最终图数': int(final_ndoc),
            '最高F1_ESM(L=2)': f"{max_f1_esm_l2:.4f}" if max_f1_esm_l2 else 'N/A',
            '最终F1_ESM(L=2)': f"{final_f1_esm_l2:.4f}" if final_f1_esm_l2 else 'N/A',
            '最高F1_ESM(L=3)': f"{max_f1_esm_l3:.4f}" if max_f1_esm_l3 else 'N/A',
            '最终F1_ESM(L=3)': f"{final_f1_esm_l3:.4f}" if final_f1_esm_l3 else 'N/A'
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # 保存到Excel（不带时间戳，直接覆盖）
    output_path = os.path.join(DRAW_DIR, 'llm_config_summary.xlsx')
    df_summary.to_excel(output_path, index=False)
    
    print(f"汇总表格已保存: {output_path}")
    print(f"\n{df_summary.to_string(index=False)}")
    
    return output_path


def main():
    """主函数"""
    print("="*80)
    print("LLM配置实验对比图绘制")
    print("="*80)
    
    # 提取所有数据
    all_results = {}
    
    for config_key, xlsx_file in XLSX_FILES.items():
        print(f"\n提取 {config_key} 的数据...")
        results = extract_metrics(xlsx_file)
        if results is not None:
            all_results[config_key] = results
    
    # 绘制对比图
    if all_results:
        plot_comparison(all_results)
        generate_summary_table(all_results)
        
        print("\n" + "="*80)
        print("完成！")
        print("="*80)
    else:
        print("\n警告: 没有成功提取到数据")


if __name__ == '__main__':
    main()

