#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
递进式图合并的边际递减效应可视化分析
用于论文图表生成
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据文件路径
DRAW_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_DIR = os.path.join(os.path.dirname(DRAW_DIR), 'xlsx')

datasets = {
    'Suicide-IED': 'layered_merger_suicide_ied_model1_combined_20251021_103009.xlsx',
    'Car-IED': 'layered_merger_wiki_mass_car_bombings_model1_combined_20251021_103110.xlsx',
    'General-IED': 'layered_merger_wiki_ied_bombings_model1_combined_20251021_103152.xlsx'
}

# 颜色和标记配置
colors = {
    'Suicide-IED': 'black',
    'Car-IED': 'black',
    'General-IED': 'black'
}

markers = {
    'Suicide-IED': 'o',
    'Car-IED': 's',
    'General-IED': '^'
}

linestyles = {
    'Suicide-IED': '-',
    'Car-IED': '--',
    'General-IED': ':'
}

# 创建图表
fig, ax = plt.subplots(figsize=(12, 7))

# 为每个数据集绘制曲线
peak_info = {}

for name, file in datasets.items():
    file_path = os.path.join(XLSX_DIR, file)
    df = pd.read_excel(file_path, sheet_name='Type.Subtype_序列3')
    
    # 获取数据
    x = df['层级'].values
    y_raw = df['F1'].values
    
    # 数据平滑处理 (移动平均，窗口大小=10)
    # center=True 保证平滑后的曲线不会向右滞后
    y = pd.Series(y_raw).rolling(window=10, center=True, min_periods=1).mean().values
    
    # 找出峰值点 (在平滑曲线上找)
    max_idx = np.argmax(y)
    peak_x = x[max_idx]
    peak_y = y[max_idx]
    
    # 全合并点 (使用原始数据的最后一点，保证终点准确)
    full_x = df.iloc[-1]['层级']
    full_y = df.iloc[-1]['F1']
    
    # 保存峰值信息用于调整标注位置
    peak_info[name] = {
        'x': peak_x,
        'y': peak_y,
        'full_x': full_x,
        'full_y': full_y
    }
    
    # 绘制曲线
    # 1. 绘制平滑后的主曲线
    ax.plot(x, y, color='black', linewidth=2.0, 
            label=name, 
            linestyle=linestyles[name],
            alpha=1.0)
            
    # 2. 稀疏绘制标记 (每30个点画一个，避免拥挤)
    # 并在图例中展示这些标记
    ax.plot(x, y, color='black', linewidth=0,
            marker=markers[name], markersize=6,
            markevery=30, alpha=1.0)
    
    # 标注峰值点
    ax.scatter([peak_x], [peak_y], color='black', s=150, 
               zorder=5, edgecolors='black', linewidths=2, facecolors='white')

# 调整标注位置避免重叠
# Suicide-IED: 在右侧偏下方，避免出边框
ax.annotate(f'峰值\n({peak_info["Suicide-IED"]["x"]}, {peak_info["Suicide-IED"]["y"]:.3f})', 
            xy=(peak_info['Suicide-IED']['x'], peak_info['Suicide-IED']['y']), 
            xytext=(peak_info['Suicide-IED']['x']+10, peak_info['Suicide-IED']['y']-0.015),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Car-IED: 在左上方
ax.annotate(f'峰值\n({peak_info["Car-IED"]["x"]}, {peak_info["Car-IED"]["y"]:.3f})', 
            xy=(peak_info['Car-IED']['x'], peak_info['Car-IED']['y']), 
            xytext=(peak_info['Car-IED']['x']-35, peak_info['Car-IED']['y']+0.015),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# General-IED: 在正上方
ax.annotate(f'峰值\n({peak_info["General-IED"]["x"]}, {peak_info["General-IED"]["y"]:.3f})', 
            xy=(peak_info['General-IED']['x'], peak_info['General-IED']['y']), 
            xytext=(peak_info['General-IED']['x']-10, peak_info['General-IED']['y']+0.025),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Car-IED的全合并点标注
car_full_x = peak_info['Car-IED']['full_x']
car_full_y = peak_info['Car-IED']['full_y']
ax.scatter([car_full_x], [car_full_y], color='black', s=150, 
           zorder=5, marker='X', edgecolors='black', linewidths=2, facecolors='white')
ax.annotate(f'全合并\n({car_full_x}, {car_full_y:.3f})', 
            xy=(car_full_x, car_full_y), 
            xytext=(car_full_x+10, car_full_y-0.025),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 添加下降区域的阴影（Car-IED）
# 使用平滑后的数据来画阴影，会更整洁
file_path = os.path.join(XLSX_DIR, datasets['Car-IED'])
df_car = pd.read_excel(file_path, sheet_name='Type.Subtype_序列3')
y_car_raw = df_car['F1'].values
# 同样进行平滑
y_car_smooth = pd.Series(y_car_raw).rolling(window=10, center=True, min_periods=1).mean().values
max_idx = np.argmax(y_car_smooth)
x_car = df_car['层级'].values
peak_y_car = y_car_smooth[max_idx]

ax.fill_between(x_car[max_idx:], y_car_smooth[max_idx:], 
                peak_y_car, alpha=0.15, color='gray',
                label='边际递减区域 (Car-IED)')

# 设置标签（去掉标题）
ax.set_xlabel('图数量 $N_{doc}$', fontsize=14, fontweight='bold')
ax.set_ylabel('$F1_{ESM}$ (L=3)', fontsize=14, fontweight='bold')

# 设置网格
ax.grid(True, linestyle='--', alpha=0.3, linewidth=1)
ax.set_axisbelow(True)

# 设置图例（去掉阴影，使用透明框）
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
          shadow=False, fancybox=False, edgecolor='gray', frameon=True)

# 设置坐标轴范围（扩大Y轴范围以容纳标注）
ax.set_xlim(0, 280)
ax.set_ylim(0, 0.23)

plt.tight_layout()

# 保存图片
output_dir = os.path.dirname(__file__)
output_path_png = os.path.join(output_dir, 'diminishing_returns_analysis.png')
output_path_pdf = os.path.join(output_dir, 'diminishing_returns_analysis.pdf')

plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"PNG图片已保存到: {output_path_png}")

plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"PDF版本已保存到: {output_path_pdf}")

plt.close()

print("\n图表生成完成！")
print("="*60)
print("使用说明：")
print("1. PNG格式适合预览和嵌入PPT")
print("2. PDF格式适合论文投稿（矢量图，无损缩放）")
print("3. 分辨率：300 DPI（符合学术期刊要求）")
print("="*60)

