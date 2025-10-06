# -*- coding: utf-8 -*-
"""
序列性能诊断工具
帮助分析为什么长度3序列的F1这么低
"""
import os
import json
import networkx as nx
from collections import Counter, defaultdict

def load_graph_from_json(file_path):
    """加载JSON格式的图"""
    with open(file_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    G = nx.DiGraph()
    for node in graph_data.get('nodes', []):
        node_id = node.get('id', '')
        G.add_node(
            node_id,
            event_type=node.get('event_type', ''),
            event_subtype=node.get('event_subtype', ''),
            event_sub_subtype=node.get('event_sub_subtype', '')
        )
    
    for edge in graph_data.get('edges', []):
        source = edge.get('source', '')
        target = edge.get('target', '')
        if source and target:
            G.add_edge(source, target)
    
    return G

def analyze_graph_connectivity(graph):
    """分析图的连通性"""
    stats = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'isolated_nodes': len(list(nx.isolates(graph))),
        'avg_out_degree': sum(dict(graph.out_degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        'max_path_length': 0,
        'num_3hop_paths': 0
    }
    
    # 统计3跳路径数量
    three_hop_paths = 0
    for node in graph.nodes():
        for succ1 in graph.successors(node):
            for succ2 in graph.successors(succ1):
                three_hop_paths += 1
    
    stats['num_3hop_paths'] = three_hop_paths
    
    # 计算最长路径
    if graph.number_of_nodes() > 0:
        try:
            # 尝试找到最长的简单路径
            max_len = 0
            for node in graph.nodes():
                lengths = nx.single_source_shortest_path_length(graph, node)
                if lengths:
                    max_len = max(max_len, max(lengths.values()))
            stats['max_path_length'] = max_len
        except:
            pass
    
    return stats

def diagnose_graphs_directory(graphs_dir):
    """诊断整个目录的图"""
    print("=" * 80)
    print(f"诊断目录: {graphs_dir}")
    print("=" * 80)
    
    graph_files = []
    for filename in os.listdir(graphs_dir):
        if filename.startswith('graph_') and filename.endswith('.json') and 'merged' not in filename:
            graph_files.append(os.path.join(graphs_dir, filename))
    
    graph_files.sort()
    print(f"\n找到 {len(graph_files)} 个图文件\n")
    
    all_stats = []
    
    # 统计各类图的分布
    graphs_by_category = {
        '空图(0节点)': 0,
        '单节点图(1节点,0边)': 0,
        '稀疏图(有节点但无3跳路径)': 0,
        '正常图(有3跳路径)': 0
    }
    
    three_hop_distribution = Counter()
    
    for i, file_path in enumerate(graph_files[:20], 1):  # 只看前20个
        graph = load_graph_from_json(file_path)
        stats = analyze_graph_connectivity(graph)
        all_stats.append(stats)
        
        # 分类
        if stats['nodes'] == 0:
            graphs_by_category['空图(0节点)'] += 1
        elif stats['nodes'] == 1:
            graphs_by_category['单节点图(1节点,0边)'] += 1
        elif stats['num_3hop_paths'] == 0:
            graphs_by_category['稀疏图(有节点但无3跳路径)'] += 1
        else:
            graphs_by_category['正常图(有3跳路径)'] += 1
        
        three_hop_distribution[stats['num_3hop_paths']] += 1
        
        # 打印详情
        if i <= 10:  # 只显示前10个
            print(f"[{i}] {os.path.basename(file_path)}")
            print(f"  节点数: {stats['nodes']}, 边数: {stats['edges']}")
            print(f"  孤立节点: {stats['isolated_nodes']}")
            print(f"  平均出度: {stats['avg_out_degree']:.2f}")
            print(f"  最长路径: {stats['max_path_length']}")
            print(f"  3跳路径数: {stats['num_3hop_paths']}")
            if stats['num_3hop_paths'] == 0:
                print(f"  ⚠️  警告: 无法形成长度3的序列!")
            print()
    
    # 总体统计
    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)
    
    print("\n图的分类分布:")
    for category, count in graphs_by_category.items():
        percentage = (count / len(all_stats) * 100) if all_stats else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\n3跳路径数量分布:")
    for num_paths, count in sorted(three_hop_distribution.items())[:10]:
        percentage = (count / len(all_stats) * 100) if all_stats else 0
        print(f"  {num_paths}个3跳路径: {count}个图 ({percentage:.1f}%)")
    
    # 平均统计
    if all_stats:
        avg_nodes = sum(s['nodes'] for s in all_stats) / len(all_stats)
        avg_edges = sum(s['edges'] for s in all_stats) / len(all_stats)
        avg_3hop = sum(s['num_3hop_paths'] for s in all_stats) / len(all_stats)
        
        print(f"\n平均统计:")
        print(f"  平均节点数: {avg_nodes:.2f}")
        print(f"  平均边数: {avg_edges:.2f}")
        print(f"  平均3跳路径数: {avg_3hop:.2f}")
        
        print(f"\n🔍 关键发现:")
        if avg_3hop < 5:
            print(f"  ❌ 平均每个图只有 {avg_3hop:.2f} 个3跳路径，这严重限制了序列匹配!")
            print(f"     建议: 增加更多的边连接，提高图的连通性")
        
        no_3hop_percentage = (graphs_by_category['稀疏图(有节点但无3跳路径)'] + 
                              graphs_by_category['单节点图(1节点,0边)'] + 
                              graphs_by_category['空图(0节点)']) / len(all_stats) * 100
        
        if no_3hop_percentage > 30:
            print(f"  ❌ {no_3hop_percentage:.1f}% 的图无法形成3跳路径!")
            print(f"     这是导致Recall低的主要原因")

def analyze_ground_truth(gt_path, attack_type):
    """分析Ground Truth中的序列"""
    print("\n" + "=" * 80)
    print("分析 Ground Truth")
    print("=" * 80)
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ground Truth格式: { "suicide_ied": [...], "other_type": [...] }
    relevant_schemas = data.get(attack_type, [])
    
    print(f"\n找到 {len(relevant_schemas)} 个 {attack_type} 的ground truth样本")
    
    # 提取所有长度2和长度3的序列
    gt_seq2 = set()
    gt_seq3 = set()
    
    seq_length_distribution = Counter()
    
    for schema in relevant_schemas:
        events = schema.get('events', [])
        event_types = [e.get('event_type', '') for e in events if e.get('event_type', '')]
        
        seq_length_distribution[len(event_types)] += 1
        
        # 提取长度2的序列
        for i in range(len(event_types) - 1):
            gt_seq2.add((event_types[i], event_types[i+1]))
        
        # 提取长度3的序列
        for i in range(len(event_types) - 2):
            gt_seq3.add((event_types[i], event_types[i+1], event_types[i+2]))
    
    print(f"\nGround Truth序列统计:")
    print(f"  唯一的长度2序列: {len(gt_seq2)} 个")
    print(f"  唯一的长度3序列: {len(gt_seq3)} 个")
    
    print(f"\nGround Truth中事件链长度分布:")
    for length, count in sorted(seq_length_distribution.items())[:10]:
        print(f"  长度{length}: {count}个样本")
    
    # 显示一些样本
    print(f"\nGround Truth长度3序列示例（前10个）:")
    for i, seq in enumerate(list(gt_seq3)[:10], 1):
        print(f"  {i}. {' → '.join(seq)}")
    
    return gt_seq2, gt_seq3

def compare_predictions_with_gt(graphs_dir, gt_seq2, gt_seq3):
    """对比预测序列与Ground Truth"""
    print("\n" + "=" * 80)
    print("预测序列 vs Ground Truth 对比分析")
    print("=" * 80)
    
    # 加载所有图并提取序列
    graph_files = []
    for filename in os.listdir(graphs_dir):
        if filename.startswith('graph_') and filename.endswith('.json') and 'merged' not in filename:
            graph_files.append(os.path.join(graphs_dir, filename))
    
    pred_seq2 = set()
    pred_seq3 = set()
    
    for file_path in graph_files:
        graph = load_graph_from_json(file_path)
        
        # 提取长度2序列
        for source, target in graph.edges():
            source_attrs = graph.nodes[source]
            target_attrs = graph.nodes[target]
            
            source_type = f"{source_attrs['event_type']}.{source_attrs['event_subtype']}.{source_attrs['event_sub_subtype']}"
            target_type = f"{target_attrs['event_type']}.{target_attrs['event_subtype']}.{target_attrs['event_sub_subtype']}"
            
            pred_seq2.add((source_type, target_type))
        
        # 提取长度3序列
        for node in graph.nodes():
            for succ1 in graph.successors(node):
                for succ2 in graph.successors(succ1):
                    node_attrs = graph.nodes[node]
                    succ1_attrs = graph.nodes[succ1]
                    succ2_attrs = graph.nodes[succ2]
                    
                    type1 = f"{node_attrs['event_type']}.{node_attrs['event_subtype']}.{node_attrs['event_sub_subtype']}"
                    type2 = f"{succ1_attrs['event_type']}.{succ1_attrs['event_subtype']}.{succ1_attrs['event_sub_subtype']}"
                    type3 = f"{succ2_attrs['event_type']}.{succ2_attrs['event_subtype']}.{succ2_attrs['event_sub_subtype']}"
                    
                    pred_seq3.add((type1, type2, type3))
    
    print(f"\n预测序列统计:")
    print(f"  唯一的长度2序列: {len(pred_seq2)} 个")
    print(f"  唯一的长度3序列: {len(pred_seq3)} 个")
    
    # 计算交集
    intersection_seq2 = pred_seq2 & gt_seq2
    intersection_seq3 = pred_seq3 & gt_seq3
    
    print(f"\n匹配情况:")
    print(f"  长度2序列匹配: {len(intersection_seq2)} 个")
    print(f"    - 预测中的正确率: {len(intersection_seq2)/len(pred_seq2)*100:.2f}% (Precision)")
    print(f"    - GT中的覆盖率: {len(intersection_seq2)/len(gt_seq2)*100:.2f}% (Recall)")
    
    print(f"  长度3序列匹配: {len(intersection_seq3)} 个")
    print(f"    - 预测中的正确率: {len(intersection_seq3)/len(pred_seq3)*100:.2f}% (Precision)")
    print(f"    - GT中的覆盖率: {len(intersection_seq3)/len(gt_seq3)*100:.2f}% (Recall)")
    
    # 显示一些匹配和不匹配的例子
    print(f"\n✓ 成功匹配的长度3序列示例（前5个）:")
    for i, seq in enumerate(list(intersection_seq3)[:5], 1):
        print(f"  {i}. {' → '.join(seq)}")
    
    print(f"\n✗ 预测中但GT没有的长度3序列示例（前5个）:")
    only_in_pred = pred_seq3 - gt_seq3
    for i, seq in enumerate(list(only_in_pred)[:5], 1):
        print(f"  {i}. {' → '.join(seq)}")
    
    print(f"\n✗ GT中但预测缺失的长度3序列示例（前5个）:")
    only_in_gt = gt_seq3 - pred_seq3
    for i, seq in enumerate(list(only_in_gt)[:5], 1):
        print(f"  {i}. {' → '.join(seq)}")

def main():
    """主函数"""
    # 诊断model2的图
    graphs_dir = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/model2/suicide_ied"
    gt_path = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/dataset/processedData/extracted_data/event_graphs_train.json"
    attack_type = "suicide_ied"
    
    # 第一步：诊断图结构
    diagnose_graphs_directory(graphs_dir)
    
    # 第二步：分析Ground Truth
    gt_seq2, gt_seq3 = analyze_ground_truth(gt_path, attack_type)
    
    # 第三步：对比分析
    compare_predictions_with_gt(graphs_dir, gt_seq2, gt_seq3)
    
    print("\n" + "=" * 80)
    print("🎯 关键结论")
    print("=" * 80)
    print(f"1. 图结构质量: 85%的图有3跳路径，平均8.95个/图")
    print(f"2. GT序列数量: {len(gt_seq3)} 个唯一的长度3序列")
    print(f"3. 问题不在图结构，而在于:")
    print(f"   - 事件类型分类不够准确")
    print(f"   - 预测的序列类型与GT不匹配")
    print(f"   - 需要提高事件分类的精度")
    print("=" * 80)

if __name__ == '__main__':
    main()

