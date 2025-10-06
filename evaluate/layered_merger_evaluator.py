# -*- coding: utf-8 -*-
"""
递进式图合并评测器
测试不同层级图合并的效果：单图 -> 2级图 -> 3级图 -> ... -> 全合并图
"""
import os
import sys
import json
import networkx as nx
from collections import defaultdict
from datetime import datetime
import random

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger
from mergeGraph import GraphMerger


class LayeredMergerEvaluator:
    """递进式图合并评测器"""
    
    def __init__(self, graphs_dir, ground_truth_path, output_dir):
        """
        初始化评测器
        
        Args:
            graphs_dir (str or list): 图文件夹路径（包含多个graph_X.json文件）
                                      可以是单个字符串路径，也可以是多个路径组成的列表
            ground_truth_path (str): ground truth数据路径
            output_dir (str): 输出目录路径
        """
        # 标准化 graphs_dir 为列表格式
        if isinstance(graphs_dir, str):
            self.graphs_dir_list = [graphs_dir]
        elif isinstance(graphs_dir, list):
            self.graphs_dir_list = graphs_dir
        else:
            raise ValueError("graphs_dir 必须是字符串或字符串列表")
        
        self.graphs_dir = graphs_dir  # 保留原始输入，用于兼容性
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir
        self.logger = Logger()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self.graphs = []
        self.ground_truth = None
        self.attack_type = self._extract_attack_type_from_path(self.graphs_dir_list[0])
        
        self._load_graphs()
        self._load_ground_truth()
    
    def _extract_attack_type_from_path(self, path):
        """从路径中提取攻击类型"""
        # 假设路径格式为 .../model2/suicide_ied/
        parts = path.rstrip('/').split('/')
        return parts[-1] if parts else 'unknown'
    
    def _load_graphs(self):
        """加载所有图文件（支持从多个目录加载）"""
        if len(self.graphs_dir_list) == 1:
            self.logger.printLog(f"从 {self.graphs_dir_list[0]} 加载图文件...")
        else:
            self.logger.printLog(f"从 {len(self.graphs_dir_list)} 个目录加载图文件...")
            for dir_path in self.graphs_dir_list:
                self.logger.printLog(f"  - {dir_path}")
        
        graph_files = []
        
        # 遍历所有目录
        for graphs_dir in self.graphs_dir_list:
            if not os.path.exists(graphs_dir):
                self.logger.printLog(f"警告: 目录不存在: {graphs_dir}")
                continue
                
            for filename in os.listdir(graphs_dir):
                if filename.startswith('graph_') and filename.endswith('.json') and 'merged' not in filename:
                    file_path = os.path.join(graphs_dir, filename)
                    graph_files.append(file_path)
        
        # 按文件名排序
        graph_files.sort()
        
        for file_path in graph_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                # 转换为networkx图
                graph = self._json_to_networkx(graph_data)
                self.graphs.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'graph': graph
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
                occurrence_percentage=node.get('occurrence_percentage', 100.0)
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
                self.logger.printLog(f"成功加载 {self.attack_type} 的ground truth")
            else:
                self.logger.printLog(f"警告: ground truth中未找到 {self.attack_type}")
                self.ground_truth = []
        except Exception as e:
            self.logger.printLog(f"加载ground truth失败: {str(e)}")
            self.ground_truth = []
    
    def _extract_event_types_from_graph(self, graph):
        """从图中提取事件类型集合"""
        event_types = set()
        
        for node_id, attrs in graph.nodes(data=True):
            event_type = attrs.get('event_type', '')
            event_subtype = attrs.get('event_subtype', '')
            event_sub_subtype = attrs.get('event_sub_subtype', '')
            
            # 构建完整的事件类型
            full_type = f"{event_type}.{event_subtype}.{event_sub_subtype}"
            event_types.add(full_type)
        
        return event_types
    
    def _extract_event_types_from_gt(self):
        """从ground truth中提取事件类型集合"""
        event_types = set()
        
        for schema in self.ground_truth:
            for event in schema.get('events', []):
                event_type = event.get('event_type', '')
                if event_type:
                    event_types.add(event_type)
        
        return event_types
    
    def _extract_sequences_from_graph(self, graph, seq_length=2):
        """从图中提取事件序列"""
        sequences = set()
        
        if seq_length == 2:
            # 长度为2的序列：直接从边提取
            for source, target in graph.edges():
                source_attrs = graph.nodes[source]
                target_attrs = graph.nodes[target]
                
                source_type = f"{source_attrs['event_type']}.{source_attrs['event_subtype']}.{source_attrs['event_sub_subtype']}"
                target_type = f"{target_attrs['event_type']}.{target_attrs['event_subtype']}.{target_attrs['event_sub_subtype']}"
                
                sequences.add((source_type, target_type))
        
        elif seq_length == 3:
            # 长度为3的序列：查找路径
            for node in graph.nodes():
                for successor1 in graph.successors(node):
                    for successor2 in graph.successors(successor1):
                        node_attrs = graph.nodes[node]
                        succ1_attrs = graph.nodes[successor1]
                        succ2_attrs = graph.nodes[successor2]
                        
                        type1 = f"{node_attrs['event_type']}.{node_attrs['event_subtype']}.{node_attrs['event_sub_subtype']}"
                        type2 = f"{succ1_attrs['event_type']}.{succ1_attrs['event_subtype']}.{succ1_attrs['event_sub_subtype']}"
                        type3 = f"{succ2_attrs['event_type']}.{succ2_attrs['event_subtype']}.{succ2_attrs['event_sub_subtype']}"
                        
                        sequences.add((type1, type2, type3))
        
        return sequences
    
    def _extract_sequences_from_gt(self, seq_length=2):
        """从ground truth中提取事件序列"""
        sequences = set()
        
        for schema in self.ground_truth:
            events = schema.get('events', [])
            event_types = [e.get('event_type', '') for e in events if e.get('event_type', '')]
            
            # 提取指定长度的序列
            for i in range(len(event_types) - seq_length + 1):
                seq = tuple(event_types[i:i+seq_length])
                sequences.add(seq)
        
        return sequences
    
    def _calculate_f1_precision_recall(self, predicted_set, ground_truth_set):
        """
        计算F1、Precision和Recall
        
        Args:
            predicted_set (set): 预测集合
            ground_truth_set (set): 真实集合
            
        Returns:
            dict: {'f1': float, 'precision': float, 'recall': float}
        """
        if not ground_truth_set:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        if not predicted_set:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # 计算交集
        intersection = predicted_set & ground_truth_set
        
        # 计算Precision
        precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
        
        # 计算Recall
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
        """评测单个图"""
        # 提取事件类型
        predicted_types = self._extract_event_types_from_graph(graph)
        gt_types = self._extract_event_types_from_gt()
        
        # 计算事件类型指标
        type_metrics = self._calculate_f1_precision_recall(predicted_types, gt_types)
        
        # 提取长度为2的序列
        predicted_seq2 = self._extract_sequences_from_graph(graph, seq_length=2)
        gt_seq2 = self._extract_sequences_from_gt(seq_length=2)
        seq2_metrics = self._calculate_f1_precision_recall(predicted_seq2, gt_seq2)
        
        # 提取长度为3的序列
        predicted_seq3 = self._extract_sequences_from_graph(graph, seq_length=3)
        gt_seq3 = self._extract_sequences_from_gt(seq_length=3)
        seq3_metrics = self._calculate_f1_precision_recall(predicted_seq3, gt_seq3)
        
        return {
            'event_types': type_metrics,
            'sequences_len2': seq2_metrics,
            'sequences_len3': seq3_metrics
        }
    
    def run_layered_evaluation(self, max_level=None):
        """
        运行递进式评测
        
        Args:
            max_level (int): 最大测试层级，None表示测试到全合并
            
        Returns:
            dict: 评测结果
        """
        self.logger.printLog("\n" + "="*80)
        self.logger.printLog("开始递进式图合并评测")
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
            
            level_results = self._evaluate_level(level, graph_merger)
            results['levels'][level] = level_results
            
            # 打印当前层级的平均结果
            self._print_level_summary(level, level_results)
        
        # 保存结果
        self._save_results(results)
        
        # 打印总结
        self._print_final_summary(results)
        
        return results
    
    def _evaluate_level(self, level, graph_merger):
        """
        评测指定层级的图合并效果
        
        Args:
            level (int): 层级（合并的图数量）
            graph_merger (GraphMerger): 图合并器
            
        Returns:
            dict: 该层级的评测结果
        """
        total_graphs = len(self.graphs)
        
        if level == 1:
            # 层级1: 测试每个单独的图
            self.logger.printLog(f"  测试 {total_graphs} 个单独的图...")
            
            all_metrics = []
            for i, graph_info in enumerate(self.graphs, 1):
                metrics = self.evaluate_single_graph(graph_info['graph'])
                metrics['graph_name'] = graph_info['filename']
                all_metrics.append(metrics)
                
                if i % 5 == 0 or i == total_graphs:
                    self.logger.printLog(f"    已测试 {i}/{total_graphs} 个图")
            
            # 计算平均指标
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
            # 中间层级: 直接随机采样
            max_samples = 20  # 最多采样20个组合
            
            self.logger.printLog(f"  随机采样 {max_samples} 个组合...")
            
            random.seed(42)  # 固定随机种子以保证可重复性
            all_metrics = []
            
            for i in range(max_samples):
                # 直接随机选择level个图的索引
                combo_indices = tuple(sorted(random.sample(range(total_graphs), level)))
                
                # 获取要合并的图
                graphs_to_merge = [self.graphs[idx]['graph'] for idx in combo_indices]
                
                # 合并图
                merged_graph = graph_merger.merge_graphs(graphs_to_merge)
                
                # 评测
                metrics = self.evaluate_single_graph(merged_graph)
                metrics['graph_name'] = f"graphs_{'_'.join(map(str, combo_indices))}"
                all_metrics.append(metrics)
                
                if (i + 1) % 10 == 0 or (i + 1) == max_samples:
                    self.logger.printLog(f"    已测试 {i + 1}/{max_samples} 个组合")
            
            # 计算平均指标
            avg_metrics = self._calculate_average_metrics(all_metrics)
            
            return {
                'level': level,
                'num_combinations': max_samples,
                'metrics': all_metrics,
                'average': avg_metrics
            }
    
    def _metrics_to_dict(self, metrics):
        """将单个metrics对象转换为dict（用于平均计算）"""
        return {
            'event_types': metrics['event_types'],
            'sequences_len2': metrics['sequences_len2'],
            'sequences_len3': metrics['sequences_len3']
        }
    
    def _calculate_average_metrics(self, all_metrics):
        """计算平均指标"""
        if not all_metrics:
            return {}
        
        # 初始化累加器
        sums = {
            'event_types': defaultdict(float),
            'sequences_len2': defaultdict(float),
            'sequences_len3': defaultdict(float)
        }
        
        # 累加所有指标
        for metrics in all_metrics:
            for category in ['event_types', 'sequences_len2', 'sequences_len3']:
                for key, value in metrics[category].items():
                    sums[category][key] += value
        
        # 计算平均
        n = len(all_metrics)
        avg = {}
        for category in ['event_types', 'sequences_len2', 'sequences_len3']:
            avg[category] = {
                key: value / n for key, value in sums[category].items()
            }
        
        return avg
    
    def _print_level_summary(self, level, level_results):
        """打印层级评测摘要"""
        avg = level_results['average']
        
        self.logger.printLog(f"\n  层级 {level} 平均结果:")
        self.logger.printLog(f"    事件类型匹配:")
        self.logger.printLog(f"      F1:        {avg['event_types']['f1']:.4f}")
        self.logger.printLog(f"      Precision: {avg['event_types']['precision']:.4f}")
        self.logger.printLog(f"      Recall:    {avg['event_types']['recall']:.4f}")
        
        self.logger.printLog(f"    序列匹配(长度2):")
        self.logger.printLog(f"      F1:        {avg['sequences_len2']['f1']:.4f}")
        self.logger.printLog(f"      Precision: {avg['sequences_len2']['precision']:.4f}")
        self.logger.printLog(f"      Recall:    {avg['sequences_len2']['recall']:.4f}")
        
        self.logger.printLog(f"    序列匹配(长度3):")
        self.logger.printLog(f"      F1:        {avg['sequences_len3']['f1']:.4f}")
        self.logger.printLog(f"      Precision: {avg['sequences_len3']['precision']:.4f}")
        self.logger.printLog(f"      Recall:    {avg['sequences_len3']['recall']:.4f}")
    
    def _print_final_summary(self, results):
        """打印最终总结"""
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog("递进式评测总结")
        self.logger.printLog(f"{'='*80}")
        
        # 表1：事件类型匹配指标
        self.logger.printLog(f"\n各层级F1分数对比 (事件类型):")
        self.logger.printLog(f"{'层级':<10} {'组合数':<15} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        self.logger.printLog("-" * 60)
        
        for level in sorted(results['levels'].keys()):
            level_data = results['levels'][level]
            avg = level_data['average']
            num_combos = level_data['num_combinations']
            
            self.logger.printLog(
                f"{level:<10} {num_combos:<15} "
                f"{avg['event_types']['f1']:<10.4f} "
                f"{avg['event_types']['precision']:<12.4f} "
                f"{avg['event_types']['recall']:<10.4f}"
            )
        
        # 表2：事件序列匹配指标（长度2）
        self.logger.printLog(f"\n各层级F1分数对比 (事件序列-长度2):")
        self.logger.printLog(f"{'层级':<10} {'组合数':<15} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        self.logger.printLog("-" * 60)
        
        for level in sorted(results['levels'].keys()):
            level_data = results['levels'][level]
            avg = level_data['average']
            num_combos = level_data['num_combinations']
            
            self.logger.printLog(
                f"{level:<10} {num_combos:<15} "
                f"{avg['sequences_len2']['f1']:<10.4f} "
                f"{avg['sequences_len2']['precision']:<12.4f} "
                f"{avg['sequences_len2']['recall']:<10.4f}"
            )
        
        # 表3：事件序列匹配指标（长度3）
        self.logger.printLog(f"\n各层级F1分数对比 (事件序列-长度3):")
        self.logger.printLog(f"{'层级':<10} {'组合数':<15} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        self.logger.printLog("-" * 60)
        
        for level in sorted(results['levels'].keys()):
            level_data = results['levels'][level]
            avg = level_data['average']
            num_combos = level_data['num_combinations']
            
            self.logger.printLog(
                f"{level:<10} {num_combos:<15} "
                f"{avg['sequences_len3']['f1']:<10.4f} "
                f"{avg['sequences_len3']['precision']:<12.4f} "
                f"{avg['sequences_len3']['recall']:<10.4f}"
            )
    
    def _save_results(self, results):
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
            f"layered_evaluation_{self.attack_type}_{model_type}_{timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.printLog(f"\n评测结果已保存到: {output_file}")


def main():
    """主函数"""
    # 硬编码配置
    GRAPHS_DIR = [
        "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/model2/suicide_ied",
        "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/suicide_ied"
    ]
    # GRAPHS_DIR = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/cache/suicide_ied"
    # GROUND_TRUTH_PATH = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/dataset/processedData/extracted_data/event_graphs_train.json"
    GROUND_TRUTH_PATH = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/dataset/processedData/extracted_data/event_graphs_test.json"
    OUTPUT_DIR = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/layered_evaluation"
    
    # 创建评测器
    evaluator = LayeredMergerEvaluator(
        graphs_dir=GRAPHS_DIR,
        ground_truth_path=GROUND_TRUTH_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # 运行评测（测试所有层级）
    results = evaluator.run_layered_evaluation()
    
    print("\n评测完成!")


if __name__ == '__main__':
    main()

