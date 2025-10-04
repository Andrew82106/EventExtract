# -*- coding: utf-8 -*-
"""
图融合模块
负责将多个事件图融合为一个骨架图
共享模块 - 供所有模型使用

采用加权融合策略：
- 保留而非过滤低频事件
- 使用频率作为权重
- 支持多层次结构提取
- 优先提高recall指标
"""
import networkx as nx
from collections import defaultdict
import statistics


class GraphMerger:
    """图融合器类"""
    
    def __init__(self, min_node_threshold=0.02, min_edge_threshold=0.02):
        """
        初始化图融合器
        
        Args:
            min_node_threshold (float): 节点过滤阈值（百分比），默认0.02（2%）
            min_edge_threshold (float): 边过滤阈值（百分比），默认0.02（2%）
        """
        self.min_node_threshold = min_node_threshold
        self.min_edge_threshold = min_edge_threshold
    
    def merge_graphs(self, graphs):
        """
        融合多个事件图为一个骨架图
        使用基于事件类型的加权融合策略
        
        Args:
            graphs (list): networkx图对象列表
            
        Returns:
            networkx.DiGraph: 融合后的骨架图
        """
        if not graphs:
            return nx.DiGraph()
        
        num_graphs = len(graphs)
        print(f"开始融合 {num_graphs} 个图...")
        print(f"  节点过滤阈值: {self.min_node_threshold} ({self.min_node_threshold*100}%)")
        print(f"  边过滤阈值: {self.min_edge_threshold} ({self.min_edge_threshold*100}%)")
        
        # 第一步：统计所有事件类型及其出现次数
        event_type_stats = defaultdict(lambda: {
            'count': 0,
            'descriptions': [],
            'subtypes': set(),
            'sub_subtypes': set()
        })
        
        # 第二步：统计所有边（事件类型对之间的时序关系）
        edge_stats = defaultdict(int)
        
        # 遍历所有图，收集统计信息
        for graph in graphs:
            for node_id, attrs in graph.nodes(data=True):
                event_type = attrs.get('event_type', 'Unknown')
                event_subtype = attrs.get('event_subtype', '')
                event_sub_subtype = attrs.get('event_sub_subtype', '')
                
                # 创建事件类型的完整标识
                type_key = self._get_type_key(event_type, event_subtype, event_sub_subtype)
                
                event_type_stats[type_key]['count'] += 1
                event_type_stats[type_key]['descriptions'].append(attrs.get('description', ''))
                if event_subtype:
                    event_type_stats[type_key]['subtypes'].add(event_subtype)
                if event_sub_subtype:
                    event_type_stats[type_key]['sub_subtypes'].add(event_sub_subtype)
            
            # 统计边
            for source, target, attrs in graph.edges(data=True):
                source_attrs = graph.nodes[source]
                target_attrs = graph.nodes[target]
                
                source_type = self._get_type_key(
                    source_attrs.get('event_type', ''),
                    source_attrs.get('event_subtype', ''),
                    source_attrs.get('event_sub_subtype', '')
                )
                target_type = self._get_type_key(
                    target_attrs.get('event_type', ''),
                    target_attrs.get('event_subtype', ''),
                    target_attrs.get('event_sub_subtype', '')
                )
                
                edge_key = (source_type, target_type)
                edge_stats[edge_key] += 1
        
        # 第三步：构建融合图 - 添加节点
        merged_graph = nx.DiGraph()
        
        # 使用新的宽松阈值（从0.1降至0.02）
        min_node_count = max(1, num_graphs * self.min_node_threshold)
        node_weights = []  # 用于统计
        
        print(f"  节点最小出现次数: {min_node_count}")
        
        for type_key, stats in event_type_stats.items():
            if stats['count'] >= min_node_count:
                parts = type_key.split(' > ')
                # 计算归一化权重
                weight = stats['count'] / num_graphs
                node_weights.append(weight)
                
                merged_graph.add_node(
                    type_key,
                    event_type=parts[0] if len(parts) > 0 else '',
                    event_subtype=parts[1] if len(parts) > 1 else '',
                    event_sub_subtype=parts[2] if len(parts) > 2 else '',
                    occurrence_count=stats['count'],
                    occurrence_percentage=stats['count'] / num_graphs * 100,
                    weight=weight  # 新增：归一化权重
                )
        
        # 第四步：构建融合图 - 添加边
        min_edge_count = max(1, num_graphs * self.min_edge_threshold)
        edge_weights = []  # 用于统计
        
        print(f"  边最小出现次数: {min_edge_count}")
        
        for (source_type, target_type), count in edge_stats.items():
            if count >= min_edge_count:
                # 确保源节点和目标节点都在图中
                if merged_graph.has_node(source_type) and merged_graph.has_node(target_type):
                    # 计算归一化权重
                    weight = count / num_graphs
                    edge_weights.append(weight)
                    
                    merged_graph.add_edge(
                        source_type,
                        target_type,
                        occurrence_count=count,
                        occurrence_percentage=count / num_graphs * 100,
                        weight=weight  # 改为归一化权重
                    )
        
        # 第五步：计算全局权重统计
        node_weight_stats = self._calculate_weight_stats(node_weights) if node_weights else {}
        edge_weight_stats = self._calculate_weight_stats(edge_weights) if edge_weights else {}
        
        # 将统计信息存储到图的属性中
        merged_graph.graph['total_source_graphs'] = num_graphs
        merged_graph.graph['node_weight_stats'] = node_weight_stats
        merged_graph.graph['edge_weight_stats'] = edge_weight_stats
        
        print(f"融合完成: {merged_graph.number_of_nodes()} 个事件类型, "
              f"{merged_graph.number_of_edges()} 个时序关系")
        
        if node_weight_stats:
            print(f"  节点权重范围: {node_weight_stats['min']:.4f} - {node_weight_stats['max']:.4f}, "
                  f"平均: {node_weight_stats['mean']:.4f}")
        if edge_weight_stats:
            print(f"  边权重范围: {edge_weight_stats['min']:.4f} - {edge_weight_stats['max']:.4f}, "
                  f"平均: {edge_weight_stats['mean']:.4f}")
        
        return merged_graph
    
    def _get_type_key(self, event_type, event_subtype='', event_sub_subtype=''):
        """
        生成事件类型的唯一键
        
        Args:
            event_type (str): 事件类型
            event_subtype (str): 事件子类型
            event_sub_subtype (str): 事件子子类型
            
        Returns:
            str: 类型键，格式：Type > Subtype > Sub_subtype
        """
        parts = [event_type]
        if event_subtype:
            parts.append(event_subtype)
        if event_sub_subtype:
            parts.append(event_sub_subtype)
        return ' > '.join(parts)
    
    def _calculate_weight_stats(self, weights):
        """
        计算权重统计信息
        
        Args:
            weights (list): 权重列表
            
        Returns:
            dict: 包含max, min, mean, median的统计字典
        """
        if not weights:
            return {}
        
        return {
            'max': max(weights),
            'min': min(weights),
            'mean': statistics.mean(weights),
            'median': statistics.median(weights)
        }
    
    def save_merged_graph(self, graph, output_path):
        """
        保存融合后的图（按照algorithm.md中定义的格式）
        
        Args:
            graph (networkx.DiGraph): 融合后的图
            output_path (str): 输出路径
        """
        import json
        
        # 构建统计信息
        statistics = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'total_source_graphs': graph.graph.get('total_source_graphs', 0)
        }
        
        # 添加权重统计（如果有）
        if 'node_weight_stats' in graph.graph:
            statistics['node_weight_stats'] = graph.graph['node_weight_stats']
        
        if 'edge_weight_stats' in graph.graph:
            statistics['edge_weight_stats'] = graph.graph['edge_weight_stats']
        
        graph_data = {
            'nodes': [],
            'edges': [],
            'statistics': statistics
        }
        
        # 转换节点（按权重降序排序）
        nodes_with_weight = []
        for node_id, attrs in graph.nodes(data=True):
            node_data = {'id': node_id}
            node_data.update(attrs)
            nodes_with_weight.append(node_data)
        
        # 按权重降序排序
        nodes_with_weight.sort(key=lambda x: x.get('weight', 0), reverse=True)
        graph_data['nodes'] = nodes_with_weight
        
        # 转换边（按权重降序排序）
        edges_with_weight = []
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target
            }
            edge_data.update(attrs)
            edges_with_weight.append(edge_data)
        
        # 按权重降序排序
        edges_with_weight.sort(key=lambda x: x.get('weight', 0), reverse=True)
        graph_data['edges'] = edges_with_weight
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"融合图已保存到: {output_path}")
        print(f"  - 节点数: {statistics['total_nodes']}")
        print(f"  - 边数: {statistics['total_edges']}")
        print(f"  - 源图数: {statistics['total_source_graphs']}")

