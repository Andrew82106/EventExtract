# -*- coding: utf-8 -*-
"""
图融合模块
负责将多个事件图融合为一个骨架图
"""
import networkx as nx
from collections import defaultdict


class GraphMerger:
    """图融合器类"""
    
    def __init__(self):
        """初始化图融合器"""
        pass
    
    def merge_graphs(self, graphs):
        """
        融合多个事件图为一个骨架图
        使用基于事件类型的节点合并策略
        
        Args:
            graphs (list): networkx图对象列表
            
        Returns:
            networkx.DiGraph: 融合后的骨架图
        """
        if not graphs:
            return nx.DiGraph()
        
        print(f"开始融合 {len(graphs)} 个图...")
        
        # 统计所有事件类型及其出现次数
        event_type_stats = defaultdict(lambda: {
            'count': 0,
            'descriptions': [],
            'subtypes': set(),
            'sub_subtypes': set()
        })
        
        # 统计所有边（事件类型对之间的关系）
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
        
        # 构建融合图
        merged_graph = nx.DiGraph()
        
        # 添加节点（只保留出现次数超过阈值的事件类型）
        min_count = max(1, len(graphs) * 0.1)  # 至少在10%的图中出现
        for type_key, stats in event_type_stats.items():
            if stats['count'] >= min_count:
                parts = type_key.split(' > ')
                merged_graph.add_node(
                    type_key,
                    event_type=parts[0] if len(parts) > 0 else '',
                    event_subtype=parts[1] if len(parts) > 1 else '',
                    event_sub_subtype=parts[2] if len(parts) > 2 else '',
                    occurrence_count=stats['count'],
                    occurrence_percentage=stats['count'] / len(graphs) * 100
                )
        
        # 添加边（只保留出现次数超过阈值的边）
        min_edge_count = max(1, len(graphs) * 0.1)  # 至少在10%的图中出现
        for (source_type, target_type), count in edge_stats.items():
            if count >= min_edge_count:
                # 确保源节点和目标节点都在图中
                if merged_graph.has_node(source_type) and merged_graph.has_node(target_type):
                    merged_graph.add_edge(
                        source_type,
                        target_type,
                        weight=count,
                        occurrence_percentage=count / len(graphs) * 100
                    )
        
        print(f"融合完成: {merged_graph.number_of_nodes()} 个事件类型, "
              f"{merged_graph.number_of_edges()} 个时序关系")
        
        return merged_graph
    
    def _get_type_key(self, event_type, event_subtype='', event_sub_subtype=''):
        """
        生成事件类型的唯一键
        
        Args:
            event_type (str): 事件类型
            event_subtype (str): 事件子类型
            event_sub_subtype (str): 事件子子类型
            
        Returns:
            str: 类型键
        """
        parts = [event_type]
        if event_subtype:
            parts.append(event_subtype)
        if event_sub_subtype:
            parts.append(event_sub_subtype)
        return ' > '.join(parts)
    
    def save_merged_graph(self, graph, output_path):
        """
        保存融合后的图
        
        Args:
            graph (networkx.DiGraph): 融合后的图
            output_path (str): 输出路径
        """
        import json
        
        graph_data = {
            'nodes': [],
            'edges': [],
            'statistics': {
                'total_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges()
            }
        }
        
        # 转换节点
        for node_id, attrs in graph.nodes(data=True):
            node_data = {'id': node_id}
            node_data.update(attrs)
            graph_data['nodes'].append(node_data)
        
        # 转换边
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target
            }
            edge_data.update(attrs)
            graph_data['edges'].append(edge_data)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"融合图已保存到: {output_path}")

