# -*- coding: utf-8 -*-
"""
统一评测器
整合递进式图合并评测和分层权重评测，支持三层事件类型评测
"""
import os
import sys
import json
import networkx as nx
from collections import defaultdict
from datetime import datetime
import random
import pandas as pd

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger
from mergeGraph import GraphMerger


class UnifiedEvaluator:
    """统一评测器：支持递进式合并评测、分层权重评测、三层事件类型评测"""
    
    def __init__(self, graphs_dir, ground_truth_path, output_dir, attack_type, max_file_count=None):
        """
        初始化评测器
        
        Args:
            graphs_dir (str or list): 图文件夹路径或路径列表
            ground_truth_path (str): ground truth数据路径
            output_dir (str): 输出目录路径
            attack_type (str): 攻击类型
            max_file_count (int, list, or None): 每个目录的最大文件数量限制
                - None: 不限制
                - int: 所有目录使用相同的限制
                - list: 每个目录对应一个限制值，长度必须与 graphs_dir 相同
        """
        # 标准化 graphs_dir 为列表格式
        if isinstance(graphs_dir, str):
            self.graphs_dir_list = [graphs_dir]
        elif isinstance(graphs_dir, list):
            self.graphs_dir_list = graphs_dir
        else:
            raise ValueError("graphs_dir 必须是字符串或字符串列表")
        
        # 标准化 max_file_count 为列表格式
        if max_file_count is None:
            # 不限制
            self.max_file_count_list = [None] * len(self.graphs_dir_list)
        elif isinstance(max_file_count, int):
            # 所有目录使用相同限制
            self.max_file_count_list = [max_file_count] * len(self.graphs_dir_list)
        elif isinstance(max_file_count, list):
            # 每个目录单独限制
            if len(max_file_count) != len(self.graphs_dir_list):
                raise ValueError(f"max_file_count 列表长度({len(max_file_count)})必须与 graphs_dir 列表长度({len(self.graphs_dir_list)})相同")
            self.max_file_count_list = max_file_count
        else:
            raise ValueError("max_file_count 必须是 None、int 或 list")
        
        self.graphs_dir = graphs_dir
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir
        self.attack_type = attack_type
        self.logger = Logger()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 分层权重配置
        self.layer_config = {
            'core': {'min_weight': 0.5, 'name': '核心层'},
            'important': {'min_weight': 0.2, 'max_weight': 0.5, 'name': '重要层'},
            'supplementary': {'min_weight': 0.02, 'max_weight': 0.2, 'name': '补充层'}
        }
        
        # 加载数据
        self.graphs = []
        self.ground_truth = None
        
        self._load_graphs()
        self._load_ground_truth()
    
    def _load_graphs(self):
        """加载所有图文件"""
        if len(self.graphs_dir_list) == 1:
            self.logger.printLog(f"从 {self.graphs_dir_list[0]} 加载图文件...")
        else:
            self.logger.printLog(f"从 {len(self.graphs_dir_list)} 个目录加载图文件...")
            for i, dir_path in enumerate(self.graphs_dir_list):
                limit_info = f" (限制最多 {self.max_file_count_list[i]} 个文件)" if self.max_file_count_list[i] else ""
                self.logger.printLog(f"  - {dir_path}{limit_info}")
        
        graph_files = []
        
        # 遍历所有目录
        for dir_idx, graphs_dir in enumerate(self.graphs_dir_list):
            if not os.path.exists(graphs_dir):
                self.logger.printLog(f"警告: 目录不存在: {graphs_dir}")
                continue
            
            # 收集当前目录的所有图文件
            dir_files = []
            for filename in os.listdir(graphs_dir):
                if filename.endswith('.json') and 'merged' not in filename:
                    file_path = os.path.join(graphs_dir, filename)
                    dir_files.append(file_path)
            
            # 按文件名排序
            dir_files.sort()
            
            # 应用最大文件数量限制
            max_count = self.max_file_count_list[dir_idx]
            if max_count is not None and len(dir_files) > max_count:
                self.logger.printLog(f"  ! 目录 {os.path.basename(graphs_dir)} 有 {len(dir_files)} 个文件，限制为 {max_count} 个")
                # 随机采样（保持随机种子以便复现）
                random.seed(42)
                dir_files = random.sample(dir_files, max_count)
                dir_files.sort()  # 重新排序以保持一致性
                self.logger.printLog(f"    已随机选择 {max_count} 个文件")
            
            # 添加到总列表
            graph_files.extend(dir_files)
        
        # 不再进行全局排序，保持各目录文件的相对顺序
        
        for file_path in graph_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                # 转换为networkx图
                graph = self._json_to_networkx(graph_data)
                self.graphs.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'graph': graph,
                    'data': graph_data  # 保留原始数据用于权重过滤
                })
            except Exception as e:
                self.logger.printLog(f"加载图文件失败 {file_path}: {str(e)}")
        
        self.logger.printLog(f"成功加载 {len(self.graphs)} 个图文件")
    
    def _json_to_networkx(self, graph_data):
        """将JSON格式的图转换为networkx图"""
        G = nx.DiGraph()
        
        # 添加节点
        for node in graph_data.get('nodes', []):
            node_id = node.get('id', '')
            G.add_node(
                node_id,
                event_type=node.get('event_type', ''),
                event_subtype=node.get('event_subtype', ''),
                event_sub_subtype=node.get('event_sub_subtype', ''),
                description=node.get('description', ''),
                support_text=node.get('support_text', ''),
                occurrence_count=node.get('occurrence_count', 1),
                occurrence_percentage=node.get('occurrence_percentage', 100.0),
                weight=node.get('weight', 1.0)
            )
        
        # 添加边
        for edge in graph_data.get('edges', []):
            source = edge.get('source', '')
            target = edge.get('target', '')
            if G.has_node(source) and G.has_node(target):
                G.add_edge(
                    source,
                    target,
                    relation=edge.get('relation', 'before'),
                    support_text=edge.get('support_text', ''),
                    weight=edge.get('weight', 1),
                    occurrence_count=edge.get('occurrence_count', 1),
                    occurrence_percentage=edge.get('occurrence_percentage', 100.0)
                )
        
        return G
    
    def _load_ground_truth(self):
        """加载ground truth数据"""
        self.logger.printLog(f"加载ground truth: {self.ground_truth_path}")
        
        try:
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            if self.attack_type in gt_data:
                self.ground_truth = gt_data[self.attack_type]
                self.logger.printLog(f"成功加载 {self.attack_type} 的ground truth ({len(self.ground_truth)}个样本)")
            else:
                self.logger.printLog(f"警告: ground truth中未找到 {self.attack_type}")
                self.ground_truth = []
        except Exception as e:
            self.logger.printLog(f"加载ground truth失败: {str(e)}")
            self.ground_truth = []
    
    def _extract_event_types_from_graph(self, graph, hierarchy_level=3):
        """
        从图中提取事件类型集合
        
        Args:
            graph: NetworkX图对象
            hierarchy_level: 层级 (1=只Type, 2=Type.Subtype, 3=Type.Subtype.Sub_subtype)
        
        Returns:
            set: 事件类型集合
        """
        event_types = set()
        
        for node_id, attrs in graph.nodes(data=True):
            event_type = attrs.get('event_type', '')
            event_subtype = attrs.get('event_subtype', '')
            event_sub_subtype = attrs.get('event_sub_subtype', '')
            
            # 根据层级构建事件类型字符串
            if hierarchy_level == 1:
                # 只考虑Type
                if event_type:
                    event_types.add(event_type)
            elif hierarchy_level == 2:
                # 考虑Type.Subtype
                if event_type:
                    parts = [event_type]
                    if event_subtype:
                        parts.append(event_subtype)
                    event_types.add('.'.join(parts))
            else:  # hierarchy_level == 3
                # 考虑Type.Subtype.Sub_subtype
                if event_type:
                    parts = [event_type]
                    if event_subtype:
                        parts.append(event_subtype)
                    if event_sub_subtype:
                        parts.append(event_sub_subtype)
                    event_types.add('.'.join(parts))
        
        return event_types
    
    def _extract_event_types_from_gt(self, hierarchy_level=3):
        """
        从ground truth中提取事件类型集合
        
        Args:
            hierarchy_level: 层级 (1=只Type, 2=Type.Subtype, 3=Type.Subtype.Sub_subtype)
        
        Returns:
            set: 事件类型集合
        """
        event_types = set()
        
        for schema in self.ground_truth:
            for event in schema.get('events', []):
                event_type_full = event.get('event_type', '')
                if not event_type_full:
                    continue
                
                # 解析完整的事件类型
                parts = event_type_full.split('.')
                
                if hierarchy_level == 1:
                    # 只取第一层
                    if len(parts) >= 1:
                        event_types.add(parts[0])
                elif hierarchy_level == 2:
                    # 取前两层
                    if len(parts) >= 2:
                        event_types.add('.'.join(parts[:2]))
                    elif len(parts) == 1:
                        event_types.add(parts[0])
                else:  # hierarchy_level == 3
                    # 取完整的三层
                    event_types.add(event_type_full)
        
        return event_types
    
    def _extract_sequences_from_graph(self, graph, seq_length=2, hierarchy_level=3):
        """从图中提取事件序列"""
        sequences = set()
        
        def get_event_type_str(attrs, level):
            """根据层级获取事件类型字符串"""
            event_type = attrs.get('event_type', '')
            event_subtype = attrs.get('event_subtype', '')
            event_sub_subtype = attrs.get('event_sub_subtype', '')
            
            if level == 1:
                return event_type
            elif level == 2:
                parts = [event_type]
                if event_subtype:
                    parts.append(event_subtype)
                return '.'.join(parts)
            else:  # level == 3
                parts = [event_type]
                if event_subtype:
                    parts.append(event_subtype)
                if event_sub_subtype:
                    parts.append(event_sub_subtype)
                return '.'.join(parts)
        
        if seq_length == 2:
            # 长度为2的序列：直接从边提取
            for source, target in graph.edges():
                source_attrs = graph.nodes[source]
                target_attrs = graph.nodes[target]
                
                source_type = get_event_type_str(source_attrs, hierarchy_level)
                target_type = get_event_type_str(target_attrs, hierarchy_level)
                
                sequences.add((source_type, target_type))
        
        elif seq_length == 3:
            # 长度为3的序列：查找路径
            for node in graph.nodes():
                for successor1 in graph.successors(node):
                    for successor2 in graph.successors(successor1):
                        node_attrs = graph.nodes[node]
                        succ1_attrs = graph.nodes[successor1]
                        succ2_attrs = graph.nodes[successor2]
                        
                        type1 = get_event_type_str(node_attrs, hierarchy_level)
                        type2 = get_event_type_str(succ1_attrs, hierarchy_level)
                        type3 = get_event_type_str(succ2_attrs, hierarchy_level)
                        
                        sequences.add((type1, type2, type3))
        
        return sequences
    
    def _extract_sequences_from_gt(self, seq_length=2, hierarchy_level=3):
        """从ground truth中提取事件序列"""
        sequences = set()
        
        for schema in self.ground_truth:
            events = schema.get('events', [])
            event_types_full = [e.get('event_type', '') for e in events if e.get('event_type', '')]
            
            # 根据层级处理事件类型
            event_types = []
            for event_type_full in event_types_full:
                parts = event_type_full.split('.')
                if hierarchy_level == 1:
                    event_types.append(parts[0] if len(parts) >= 1 else '')
                elif hierarchy_level == 2:
                    event_types.append('.'.join(parts[:2]) if len(parts) >= 2 else parts[0] if len(parts) == 1 else '')
                else:  # hierarchy_level == 3
                    event_types.append(event_type_full)
            
            # 提取指定长度的序列
            for i in range(len(event_types) - seq_length + 1):
                seq = tuple(event_types[i:i+seq_length])
                if all(seq):  # 确保序列中没有空字符串
                    sequences.add(seq)
        
        return sequences
    
    def _calculate_f1_precision_recall(self, predicted_set, ground_truth_set):
        """计算F1、Precision和Recall"""
        if not ground_truth_set:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 
                    'predicted_count': len(predicted_set), 'gt_count': 0, 'intersection_count': 0}
        
        if not predicted_set:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'predicted_count': 0, 'gt_count': len(ground_truth_set), 'intersection_count': 0}
        
        # 计算交集
        intersection = predicted_set & ground_truth_set
        
        # 计算Precision和Recall
        precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0.0
        
        # 计算F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predicted_count': len(predicted_set),
            'gt_count': len(ground_truth_set),
            'intersection_count': len(intersection)
        }
    
    def evaluate_single_graph(self, graph):
        """
        评测单个图（支持三层事件类型评测）
        
        Returns:
            dict: 包含三个层级的评测结果
        """
        metrics = {}
        
        # 三个层级的事件类型评测
        for level in [1, 2, 3]:
            level_name = f'level{level}'
            
            # 提取事件类型
            pred_types = self._extract_event_types_from_graph(graph, hierarchy_level=level)
            gt_types = self._extract_event_types_from_gt(hierarchy_level=level)
            type_metrics = self._calculate_f1_precision_recall(pred_types, gt_types)
            
            # 提取长度2的序列
            pred_seq2 = self._extract_sequences_from_graph(graph, seq_length=2, hierarchy_level=level)
            gt_seq2 = self._extract_sequences_from_gt(seq_length=2, hierarchy_level=level)
            seq2_metrics = self._calculate_f1_precision_recall(pred_seq2, gt_seq2)
            
            # 提取长度3的序列
            pred_seq3 = self._extract_sequences_from_graph(graph, seq_length=3, hierarchy_level=level)
            gt_seq3 = self._extract_sequences_from_gt(seq_length=3, hierarchy_level=level)
            seq3_metrics = self._calculate_f1_precision_recall(pred_seq3, gt_seq3)
            
            metrics[level_name] = {
                'event_types': type_metrics,
                'sequences_len2': seq2_metrics,
                'sequences_len3': seq3_metrics
            }
        
        return metrics
    
    def _filter_graph_by_weight(self, graph_data, min_weight=None, max_weight=None):
        """根据权重过滤图"""
        if not graph_data or 'nodes' not in graph_data or 'edges' not in graph_data:
            return graph_data
        
        # 过滤节点
        filtered_nodes = []
        valid_node_ids = set()
        
        for node in graph_data['nodes']:
            node_weight = node.get('weight', 0)
            
            if min_weight is not None and node_weight < min_weight:
                continue
            if max_weight is not None and node_weight >= max_weight:
                continue
            
            filtered_nodes.append(node)
            valid_node_ids.add(node.get('id'))
        
        # 过滤边
        filtered_edges = []
        for edge in graph_data['edges']:
            edge_weight = edge.get('weight', 0)
            source = edge.get('source')
            target = edge.get('target')
            
            if source not in valid_node_ids or target not in valid_node_ids:
                continue
            
            if min_weight is not None and edge_weight < min_weight:
                continue
            if max_weight is not None and edge_weight >= max_weight:
                continue
            
            filtered_edges.append(edge)
        
        return {
            'nodes': filtered_nodes,
            'edges': filtered_edges
        }
    
    def run_layered_merger_evaluation(self, max_level=None, num_samples=5):
        """
        运行递进式图合并评测
        
        Args:
            max_level: 最大测试层级
            num_samples: 中间层级的采样数量
        
        Returns:
            dict: 评测结果
        """
        self.logger.printLog("\n" + "="*80)
        self.logger.printLog("开始递进式图合并评测（三层事件类型）")
        self.logger.printLog("="*80)
        
        if not self.graphs:
            self.logger.printLog("错误: 没有加载到任何图")
            return None
        
        if not self.ground_truth:
            self.logger.printLog("错误: 没有加载到ground truth")
            return None
        
        total_graphs = len(self.graphs)
        self.logger.printLog(f"总图数: {total_graphs}")
        self.logger.printLog(f"攻击类型: {self.attack_type}")
        
        # 确定最大层级
        if max_level is None:
            max_level = total_graphs
        else:
            max_level = min(max_level, total_graphs)
        
        self.logger.printLog(f"测试层级: 1 到 {max_level}")
        
        results = {
            'attack_type': self.attack_type,
            'total_graphs': total_graphs,
            'max_level': max_level,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'levels': {}
        }
        
        graph_merger = GraphMerger(min_node_threshold=0.0, min_edge_threshold=0.0)
        
        # 逐层测试
        for level in range(1, max_level + 1):
            self.logger.printLog(f"\n{'='*60}")
            self.logger.printLog(f"层级 {level}: 测试 {level} 个图的合并效果")
            self.logger.printLog(f"{'='*60}")
            
            level_results = self._evaluate_level(level, graph_merger, num_samples)
            results['levels'][level] = level_results
            
            # 打印当前层级的平均结果
            self._print_level_summary(level, level_results)
        
        # 保存结果
        self._save_results(results, 'layered_merger')
        
        # 打印总结
        self._print_final_summary(results)
        
        return results
    
    def _evaluate_level(self, level, graph_merger, num_samples):
        """评测指定层级"""
        total_graphs = len(self.graphs)
        
        if level == 1:
            # 层级1: 测试每个单独的图
            self.logger.printLog(f"  测试 {total_graphs} 个单独的图...")
            
            all_metrics = []
            for i, graph_info in enumerate(self.graphs, 1):
                metrics = self.evaluate_single_graph(graph_info['graph'])
                metrics['graph_name'] = graph_info['filename']
                all_metrics.append(metrics)
                
                if i % 10 == 0 or i == total_graphs:
                    self.logger.printLog(f"    已测试 {i}/{total_graphs} 个图")
            
            avg_metrics = self._calculate_average_metrics(all_metrics)
            
            return {
                'level': level,
                'num_combinations': total_graphs,
                'metrics': all_metrics,
                'average': avg_metrics
            }
        
        elif level == total_graphs:
            # 最高层级: 合并所有图
            self.logger.printLog(f"  合并所有 {total_graphs} 个图...")
            
            all_graph_objects = [g['graph'] for g in self.graphs]
            merged_graph = graph_merger.merge_graphs(all_graph_objects)
            
            metrics = self.evaluate_single_graph(merged_graph)
            metrics['graph_name'] = f'all_{total_graphs}_graphs_merged'
            
            return {
                'level': level,
                'num_combinations': 1,
                'metrics': [metrics],
                'average': self._metrics_to_dict(metrics)
            }
        
        else:
            # 中间层级: 随机采样
            self.logger.printLog(f"  随机采样 {num_samples} 个组合...")
            
            random.seed(42)
            all_metrics = []
            
            for i in range(num_samples):
                combo_indices = tuple(sorted(random.sample(range(total_graphs), level)))
                graphs_to_merge = [self.graphs[idx]['graph'] for idx in combo_indices]
                merged_graph = graph_merger.merge_graphs(graphs_to_merge)
                
                metrics = self.evaluate_single_graph(merged_graph)
                metrics['graph_name'] = f"combo_{i+1}"
                all_metrics.append(metrics)
                
                if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                    self.logger.printLog(f"    已测试 {i + 1}/{num_samples} 个组合")
            
            avg_metrics = self._calculate_average_metrics(all_metrics)
            
            return {
                'level': level,
                'num_combinations': num_samples,
                'metrics': all_metrics,
                'average': avg_metrics
            }
    
    def _metrics_to_dict(self, metrics):
        """将单个metrics对象转换为dict"""
        return {
            f'level{i}': metrics[f'level{i}']
            for i in [1, 2, 3]
        }
    
    def _calculate_average_metrics(self, all_metrics):
        """计算平均指标"""
        if not all_metrics:
            return {}
        
        # 初始化累加器
        sums = {}
        for level in [1, 2, 3]:
            level_name = f'level{level}'
            sums[level_name] = {
                'event_types': defaultdict(float),
                'sequences_len2': defaultdict(float),
                'sequences_len3': defaultdict(float)
            }
        
        # 累加所有指标
        for metrics in all_metrics:
            for level in [1, 2, 3]:
                level_name = f'level{level}'
                for category in ['event_types', 'sequences_len2', 'sequences_len3']:
                    for key, value in metrics[level_name][category].items():
                        sums[level_name][category][key] += value
        
        # 计算平均
        n = len(all_metrics)
        avg = {}
        for level in [1, 2, 3]:
            level_name = f'level{level}'
            avg[level_name] = {}
            for category in ['event_types', 'sequences_len2', 'sequences_len3']:
                avg[level_name][category] = {
                    key: value / n for key, value in sums[level_name][category].items()
                }
        
        return avg
    
    def _print_level_summary(self, level, level_results):
        """打印层级评测摘要"""
        avg = level_results['average']
        
        self.logger.printLog(f"\n  层级 {level} 平均结果:")
        
        for i in [1, 2, 3]:
            level_name = f'level{i}'
            hierarchy_desc = ['只Type', 'Type.Subtype', 'Type.Subtype.Sub_subtype'][i-1]
            
            self.logger.printLog(f"    [{hierarchy_desc}]")
            self.logger.printLog(f"      事件类型: F1={avg[level_name]['event_types']['f1']:.4f}, "
                               f"P={avg[level_name]['event_types']['precision']:.4f}, "
                               f"R={avg[level_name]['event_types']['recall']:.4f}")
            self.logger.printLog(f"      序列(长度2): F1={avg[level_name]['sequences_len2']['f1']:.4f}, "
                               f"P={avg[level_name]['sequences_len2']['precision']:.4f}, "
                               f"R={avg[level_name]['sequences_len2']['recall']:.4f}")
            self.logger.printLog(f"      序列(长度3): F1={avg[level_name]['sequences_len3']['f1']:.4f}, "
                               f"P={avg[level_name]['sequences_len3']['precision']:.4f}, "
                               f"R={avg[level_name]['sequences_len3']['recall']:.4f}")
    
    def _print_final_summary(self, results):
        """打印最终总结"""
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog("递进式评测总结")
        self.logger.printLog(f"{'='*80}")
        
        for hierarchy_level in [1, 2, 3]:
            level_name = f'level{hierarchy_level}'
            hierarchy_desc = ['只Type', 'Type.Subtype', 'Type.Subtype.Sub_subtype'][hierarchy_level-1]
            
            # 事件类型匹配
            self.logger.printLog(f"\n[{hierarchy_desc}] 事件类型匹配F1:")
            self.logger.printLog(f"{'层级':<8} {'组合数':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
            self.logger.printLog("-" * 60)
            
            for level in sorted(results['levels'].keys()):
                level_data = results['levels'][level]
                avg = level_data['average']
                num_combos = level_data['num_combinations']
                
                self.logger.printLog(
                    f"{level:<8} {num_combos:<10} "
                    f"{avg[level_name]['event_types']['f1']:<10.4f} "
                    f"{avg[level_name]['event_types']['precision']:<12.4f} "
                    f"{avg[level_name]['event_types']['recall']:<10.4f}"
                )
            
            # 序列长度2匹配
            self.logger.printLog(f"\n[{hierarchy_desc}] 序列匹配F1 (长度2):")
            self.logger.printLog(f"{'层级':<8} {'组合数':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
            self.logger.printLog("-" * 60)
            
            for level in sorted(results['levels'].keys()):
                level_data = results['levels'][level]
                avg = level_data['average']
                num_combos = level_data['num_combinations']
                
                self.logger.printLog(
                    f"{level:<8} {num_combos:<10} "
                    f"{avg[level_name]['sequences_len2']['f1']:<10.4f} "
                    f"{avg[level_name]['sequences_len2']['precision']:<12.4f} "
                    f"{avg[level_name]['sequences_len2']['recall']:<10.4f}"
                )
            
            # 序列长度3匹配
            self.logger.printLog(f"\n[{hierarchy_desc}] 序列匹配F1 (长度3):")
            self.logger.printLog(f"{'层级':<8} {'组合数':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
            self.logger.printLog("-" * 60)
            
            for level in sorted(results['levels'].keys()):
                level_data = results['levels'][level]
                avg = level_data['average']
                num_combos = level_data['num_combinations']
                
                self.logger.printLog(
                    f"{level:<8} {num_combos:<10} "
                    f"{avg[level_name]['sequences_len3']['f1']:<10.4f} "
                    f"{avg[level_name]['sequences_len3']['precision']:<12.4f} "
                    f"{avg[level_name]['sequences_len3']['recall']:<10.4f}"
                )
    
    def _save_results(self, results, evaluation_type='layered_merger'):
        """保存评测结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 检查所有目录，确定模型类型
        model_type = "unknown"
        all_dirs_str = " ".join(self.graphs_dir_list)
        if "model1" in all_dirs_str:
            model_type = "model1"
        elif "model2" in all_dirs_str:
            model_type = "model2"
        elif "model3" in all_dirs_str:
            model_type = "model3"
        
        # 如果是多个目录，添加 "combined" 标记
        if len(self.graphs_dir_list) > 1:
            model_type = f"{model_type}_combined"
        
        output_file = os.path.join(
            self.output_dir,
            f"{evaluation_type}_{self.attack_type}_{model_type}_{timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.printLog(f"\n评测结果已保存到: {output_file}")
        
        # 同时导出xlsx文件
        self._export_to_xlsx(results, evaluation_type, model_type, timestamp)
    
    def _export_to_xlsx(self, results, evaluation_type, model_type, timestamp):
        """导出评测结果为xlsx文件"""
        # 创建xlsx输出目录（在项目根目录下）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xlsx_dir = os.path.join(project_root, 'result', 'cache', 'runResult', 'xlsx')
        os.makedirs(xlsx_dir, exist_ok=True)
        
        # 创建Excel文件路径
        xlsx_file = os.path.join(
            xlsx_dir,
            f"{evaluation_type}_{self.attack_type}_{model_type}_{timestamp}.xlsx"
        )
        
        # 创建Excel写入器
        with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
            
            # ========== 第一个工作表：全合并汇总 ==========
            # 找到最高层级（所有图合并）的结果
            max_level = max(results['levels'].keys())
            all_merged_data = results['levels'][max_level]['average']
            
            # 创建汇总数据
            summary_data = []
            for hierarchy_level in [1, 2, 3]:
                level_name = f'level{hierarchy_level}'
                hierarchy_desc = ['只Type', 'Type.Subtype', 'Type.Subtype.Sub_subtype'][hierarchy_level-1]
                
                # 事件类型指标
                summary_data.append({
                    '层级': hierarchy_desc,
                    '指标类型': '事件类型',
                    'F1': all_merged_data[level_name]['event_types']['f1'],
                    'Precision': all_merged_data[level_name]['event_types']['precision'],
                    'Recall': all_merged_data[level_name]['event_types']['recall'],
                    '预测数量': all_merged_data[level_name]['event_types'].get('predicted_count', 0),
                    'GT数量': all_merged_data[level_name]['event_types'].get('gt_count', 0),
                    '交集数量': all_merged_data[level_name]['event_types'].get('intersection_count', 0)
                })
                
                # 序列2指标
                summary_data.append({
                    '层级': hierarchy_desc,
                    '指标类型': '序列(长度2)',
                    'F1': all_merged_data[level_name]['sequences_len2']['f1'],
                    'Precision': all_merged_data[level_name]['sequences_len2']['precision'],
                    'Recall': all_merged_data[level_name]['sequences_len2']['recall'],
                    '预测数量': all_merged_data[level_name]['sequences_len2'].get('predicted_count', 0),
                    'GT数量': all_merged_data[level_name]['sequences_len2'].get('gt_count', 0),
                    '交集数量': all_merged_data[level_name]['sequences_len2'].get('intersection_count', 0)
                })
                
                # 序列3指标
                summary_data.append({
                    '层级': hierarchy_desc,
                    '指标类型': '序列(长度3)',
                    'F1': all_merged_data[level_name]['sequences_len3']['f1'],
                    'Precision': all_merged_data[level_name]['sequences_len3']['precision'],
                    'Recall': all_merged_data[level_name]['sequences_len3']['recall'],
                    '预测数量': all_merged_data[level_name]['sequences_len3'].get('predicted_count', 0),
                    'GT数量': all_merged_data[level_name]['sequences_len3'].get('gt_count', 0),
                    '交集数量': all_merged_data[level_name]['sequences_len3'].get('intersection_count', 0)
                })
            
            # 写入汇总表（第一个工作表）
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='全合并汇总', index=False)
            
            # ========== 第二个工作表：最优图数量组合 ==========
            optimal_data = []
            
            for hierarchy_level in [1, 2, 3]:
                level_name = f'level{hierarchy_level}'
                hierarchy_desc = ['只Type', 'Type.Subtype', 'Type.Subtype.Sub_subtype'][hierarchy_level-1]
                
                # 收集所有层级的数据（图数量 -> 指标值）
                level_data_map = {}
                for level in sorted(results['levels'].keys()):
                    level_data = results['levels'][level]
                    avg = level_data['average']
                    
                    level_data_map[level] = {
                        'EM': avg[level_name]['event_types']['f1'],
                        'ESM2': avg[level_name]['sequences_len2']['f1'],
                        'ESM3': avg[level_name]['sequences_len3']['f1']
                    }
                
                # 找出EM最大值对应的图数量
                em_max_level = max(level_data_map.keys(), key=lambda x: level_data_map[x]['EM'])
                em_max_data = level_data_map[em_max_level]
                optimal_data.append({
                    '层级': hierarchy_desc,
                    '最大指标': 'EM',
                    '图数量': em_max_level,
                    'EM (F1)': em_max_data['EM'],
                    'ESM2 (F1)': em_max_data['ESM2'],
                    'ESM3 (F1)': em_max_data['ESM3']
                })
                
                # 找出ESM2最大值对应的图数量
                esm2_max_level = max(level_data_map.keys(), key=lambda x: level_data_map[x]['ESM2'])
                esm2_max_data = level_data_map[esm2_max_level]
                optimal_data.append({
                    '层级': hierarchy_desc,
                    '最大指标': 'ESM2',
                    '图数量': esm2_max_level,
                    'EM (F1)': esm2_max_data['EM'],
                    'ESM2 (F1)': esm2_max_data['ESM2'],
                    'ESM3 (F1)': esm2_max_data['ESM3']
                })
                
                # 找出ESM3最大值对应的图数量
                esm3_max_level = max(level_data_map.keys(), key=lambda x: level_data_map[x]['ESM3'])
                esm3_max_data = level_data_map[esm3_max_level]
                optimal_data.append({
                    '层级': hierarchy_desc,
                    '最大指标': 'ESM3',
                    '图数量': esm3_max_level,
                    'EM (F1)': esm3_max_data['EM'],
                    'ESM2 (F1)': esm3_max_data['ESM2'],
                    'ESM3 (F1)': esm3_max_data['ESM3']
                })
            
            # 写入最优组合表（第二个工作表）
            df_optimal = pd.DataFrame(optimal_data)
            df_optimal.to_excel(writer, sheet_name='最优图数量组合', index=False)
            
            # ========== 其余工作表：各层级详细数据 ==========
            # 为每个层级和指标类型创建工作表
            for hierarchy_level in [1, 2, 3]:
                level_name = f'level{hierarchy_level}'
                hierarchy_desc = ['只Type', 'Type.Subtype', 'Type.Subtype.Sub_subtype'][hierarchy_level-1]
                
                # 事件类型匹配结果
                event_types_data = []
                for level in sorted(results['levels'].keys()):
                    level_data = results['levels'][level]
                    avg = level_data['average']
                    num_combos = level_data['num_combinations']
                    
                    event_types_data.append({
                        '层级': level,
                        '组合数': num_combos,
                        'F1': avg[level_name]['event_types']['f1'],
                        'Precision': avg[level_name]['event_types']['precision'],
                        'Recall': avg[level_name]['event_types']['recall']
                    })
                
                # 序列匹配结果（长度2）
                sequences_len2_data = []
                for level in sorted(results['levels'].keys()):
                    level_data = results['levels'][level]
                    avg = level_data['average']
                    num_combos = level_data['num_combinations']
                    
                    sequences_len2_data.append({
                        '层级': level,
                        '组合数': num_combos,
                        'F1': avg[level_name]['sequences_len2']['f1'],
                        'Precision': avg[level_name]['sequences_len2']['precision'],
                        'Recall': avg[level_name]['sequences_len2']['recall']
                    })
                
                # 序列匹配结果（长度3）
                sequences_len3_data = []
                for level in sorted(results['levels'].keys()):
                    level_data = results['levels'][level]
                    avg = level_data['average']
                    num_combos = level_data['num_combinations']
                    
                    sequences_len3_data.append({
                        '层级': level,
                        '组合数': num_combos,
                        'F1': avg[level_name]['sequences_len3']['f1'],
                        'Precision': avg[level_name]['sequences_len3']['precision'],
                        'Recall': avg[level_name]['sequences_len3']['recall']
                    })
                
                # 写入工作表
                sheet_name = f'{hierarchy_desc}'
                if len(sheet_name) > 31:  # Excel工作表名称限制
                    sheet_name = sheet_name[:31]
                
                # 事件类型匹配
                df_event_types = pd.DataFrame(event_types_data)
                df_event_types.to_excel(writer, sheet_name=f'{sheet_name}_事件类型', index=False)
                
                # 序列匹配（长度2）
                df_seq2 = pd.DataFrame(sequences_len2_data)
                df_seq2.to_excel(writer, sheet_name=f'{sheet_name}_序列2', index=False)
                
                # 序列匹配（长度3）
                df_seq3 = pd.DataFrame(sequences_len3_data)
                df_seq3.to_excel(writer, sheet_name=f'{sheet_name}_序列3', index=False)
        
        self.logger.printLog(f"Excel文件已保存到: {xlsx_file}")


def main(choice):
    """主函数 - 参数硬编码"""
    
    # ==================== 配置参数 ====================
    
    # 图文件目录（可以是单个路径或路径列表）
    # GRAPHS_DIR = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/model2/suicide_ied"
    if choice == 1:
        GRAPHS_DIR = [
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model1/glm-4-flash/suicide_ied",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model2/glm4.6/suicide_ied",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model2/glm-z1/suicide_ied",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model2/glm-z11/suicide_ied",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model1/glm-z1/suicide_ied",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model1/glm4.6/suicide_ied",
        ]
        MAX_FILE_COUNT = [
            1000,
            1000,
            1000,
            1000,
            1000,
            1000
        ]
        # 攻击类型
        ATTACK_TYPE = "suicide_ied"
    elif choice == 2:
        GRAPHS_DIR = [
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model2/glm4.6/wiki_mass_car_bombings",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model2/glm-z1/wiki_mass_car_bombings",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model1/glm-z1/wiki_mass_car_bombings",
        ]
        MAX_FILE_COUNT = [
            1000,
            1000,
            1000
        ]
        # 攻击类型
        ATTACK_TYPE = "wiki_mass_car_bombings"
    elif choice == 3:
        GRAPHS_DIR = [
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model1/glm-z1/wiki_ied_bombings",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model2/glm-z1/wiki_ied_bombings",
            "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/runResult/model2/glm4.6/wiki_ied_bombings",
        ]
        MAX_FILE_COUNT = [
            1000,
            1000,
            1000
        ]
        # 攻击类型
        ATTACK_TYPE = "wiki_ied_bombings"
    else:
        raise ValueError(f"Invalid choice: {choice}")

    assert len(GRAPHS_DIR) == len(MAX_FILE_COUNT)
    
    # Ground Truth路径
    GROUND_TRUTH_PATH = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/dataset/processedData/extracted_data/event_graphs_train.json"
    # GROUND_TRUTH_PATH = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/dataset/processedData/extracted_data/event_graphs_test.json"
    # GROUND_TRUTH_PATH = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/dataset/processedData/extracted_data/event_graphs_dev.json"
    
    # 输出目录
    OUTPUT_DIR = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/unified_evaluation"
    
    
    # 递进式评测参数
    MAX_LEVEL = None  # None表示测试到全合并，或指定具体数字如20
    NUM_SAMPLES = 3  # 中间层级的采样数量
    
    # ==================================================
    
    # 创建评测器
    evaluator = UnifiedEvaluator(
        graphs_dir=GRAPHS_DIR,
        ground_truth_path=GROUND_TRUTH_PATH,
        output_dir=OUTPUT_DIR,
        attack_type=ATTACK_TYPE,
        max_file_count=MAX_FILE_COUNT
    )
    
    # 运行递进式图合并评测（包含三层事件类型评测）
    results = evaluator.run_layered_merger_evaluation(
        max_level=MAX_LEVEL,
        num_samples=NUM_SAMPLES
    )
    
    print("\n" + "="*80)
    print("评测完成!")
    print("="*80)


if __name__ == '__main__':
    main(1)
    main(2)
    main(3)

