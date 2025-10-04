# -*- coding: utf-8 -*-
"""
事件骨架图评测器
按照algorithm.md中的评测方法评测生成的事件骨架图
"""
import os
import sys
import json
import networkx as nx
from collections import defaultdict
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger


class GraphEvaluator:
    """事件骨架图评测器"""
    
    def __init__(self, ground_truth_path, result_root_path):
        """
        初始化评测器
        
        Args:
            ground_truth_path: ground truth数据路径
            result_root_path: 结果根目录路径
        """
        self.ground_truth_path = ground_truth_path
        self.result_root_path = result_root_path
        self.logger = Logger()
        self.ground_truth_data = {}
        
        # 加载ground truth数据
        self._load_ground_truth()
    
    def _load_ground_truth(self):
        """加载ground truth数据"""
        self.logger.printLog(f"加载ground truth数据: {self.ground_truth_path}")
        
        try:
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                self.ground_truth_data = json.load(f)
            
            self.logger.printLog(f"成功加载ground truth，包含攻击类型: {list(self.ground_truth_data.keys())}")
        except Exception as e:
            self.logger.printLog(f"加载ground truth失败: {str(e)}")
    
    def _extract_event_types_from_gt(self, attack_type):
        """
        从ground truth中提取事件类型集合
        
        Args:
            attack_type: 攻击类型
            
        Returns:
            set: 事件类型集合
        """
        event_types = set()
        
        if attack_type not in self.ground_truth_data:
            return event_types
        
        schemas = self.ground_truth_data[attack_type]
        for schema in schemas:
            for event in schema.get('events', []):
                event_type = event.get('event_type', '')
                if event_type:
                    event_types.add(event_type)
        
        return event_types
    
    def _extract_event_sequences_from_gt(self, attack_type, seq_length=2):
        """
        从ground truth中提取事件序列（长度为2或3）
        
        Args:
            attack_type: 攻击类型
            seq_length: 序列长度 (2或3)
            
        Returns:
            set: 事件序列集合，每个序列是一个元组
        """
        sequences = set()
        
        if attack_type not in self.ground_truth_data:
            self.logger.printLog(f"  警告: ground truth中未找到攻击类型 {attack_type}")
            return sequences
        
        schemas = self.ground_truth_data[attack_type]
        total_relations = 0
        total_events = 0
        
        for schema in schemas:
            # 根据temporal_relations构建事件序列
            temporal_relations = schema.get('temporal_relations', [])
            events = {e['event_id']: e['event_type'] for e in schema.get('events', [])}
            
            total_relations += len(temporal_relations)
            total_events += len(events)
            
            # 构建事件图
            G = nx.DiGraph()
            for relation in temporal_relations:
                before_event = relation.get('before')
                after_event = relation.get('after')
                if before_event in events and after_event in events:
                    # before事件 -> after事件，表示before在时序上早于after
                    G.add_edge(events[before_event], events[after_event])
            
            # 提取长度为seq_length的路径
            for node in G.nodes():
                # 使用DFS找到所有从当前节点出发的路径
                paths = self._find_paths_dfs(G, node, seq_length)
                for path in paths:
                    if len(path) == seq_length:
                        sequences.add(tuple(path))
        
        self.logger.printLog(f"  Ground Truth统计: {len(schemas)}个schema, {total_events}个事件, {total_relations}个时序关系")
        self.logger.printLog(f"  提取到{len(sequences)}个长度为{seq_length}的事件序列")
        
        return sequences
    
    def _find_paths_dfs(self, G, start_node, length):
        """
        使用DFS查找从起始节点出发的所有指定长度的路径
        
        Args:
            G: NetworkX图对象
            start_node: 起始节点
            length: 路径长度
            
        Returns:
            list: 路径列表
        """
        if length == 1:
            return [[start_node]]
        
        paths = []
        for neighbor in G.successors(start_node):
            sub_paths = self._find_paths_dfs(G, neighbor, length - 1)
            for sub_path in sub_paths:
                paths.append([start_node] + sub_path)
        
        return paths
    
    def _load_generated_graph(self, graph_path):
        """
        加载生成的图
        
        Args:
            graph_path: 图文件路径
            
        Returns:
            dict: 图数据字典，包含nodes和edges
        """
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            return graph_data
        except Exception as e:
            self.logger.printLog(f"加载图失败 {graph_path}: {str(e)}")
            return None
    
    def _extract_event_types_from_graph(self, graph_data):
        """
        从生成的图中提取事件类型集合
        
        Args:
            graph_data: 图数据字典
            
        Returns:
            set: 事件类型集合
        """
        event_types = set()
        
        if not graph_data or 'nodes' not in graph_data:
            return event_types
        
        for node in graph_data['nodes']:
            # 构建完整的事件类型（包含type, subtype, sub_subtype）
            event_type = node.get('event_type', '')
            event_subtype = node.get('event_subtype', '')
            event_sub_subtype = node.get('event_sub_subtype', '')
            
            # 构建事件类型字符串，匹配ground truth的格式
            if event_type:
                parts = [event_type]
                if event_subtype:
                    parts.append(event_subtype)
                if event_sub_subtype:
                    parts.append(event_sub_subtype)
                full_type = '.'.join(parts)
                event_types.add(full_type)
        
        return event_types
    
    def _extract_event_sequences_from_graph(self, graph_data, seq_length=2):
        """
        从生成的图中提取事件序列（长度为2或3）
        
        Args:
            graph_data: 图数据字典
            seq_length: 序列长度 (2或3)
            
        Returns:
            set: 事件序列集合，每个序列是一个元组
        """
        sequences = set()
        
        if not graph_data or 'nodes' not in graph_data or 'edges' not in graph_data:
            return sequences
        
        # 构建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点
        node_id_to_type = {}
        for node in graph_data['nodes']:
            node_id = node.get('id', '')
            event_type = node.get('event_type', '')
            event_subtype = node.get('event_subtype', '')
            event_sub_subtype = node.get('event_sub_subtype', '')
            
            # 构建完整的事件类型
            parts = [event_type]
            if event_subtype:
                parts.append(event_subtype)
            if event_sub_subtype:
                parts.append(event_sub_subtype)
            full_type = '.'.join(parts)
            
            node_id_to_type[node_id] = full_type
            G.add_node(full_type)
        
        # 添加边
        for edge in graph_data['edges']:
            source = edge.get('source', '')
            target = edge.get('target', '')
            if source in node_id_to_type and target in node_id_to_type:
                source_type = node_id_to_type[source]
                target_type = node_id_to_type[target]
                G.add_edge(source_type, target_type)
        
        # 提取长度为seq_length的路径
        for node in G.nodes():
            paths = self._find_paths_dfs(G, node, seq_length)
            for path in paths:
                if len(path) == seq_length:
                    sequences.add(tuple(path))
        
        return sequences
    
    def calculate_f1(self, set_a, set_b):
        """
        计算两个集合之间的F1值
        
        Args:
            set_a: 集合A
            set_b: 集合B
            
        Returns:
            tuple: (precision, recall, f1)
        """
        if not set_a and not set_b:
            return 1.0, 1.0, 1.0
        
        if not set_a or not set_b:
            return 0.0, 0.0, 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        if union == 0:
            return 0.0, 0.0, 0.0
        
        precision = intersection / len(set_a) if len(set_a) > 0 else 0.0
        recall = intersection / len(set_b) if len(set_b) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return precision, recall, f1
    
    def evaluate_merged_graph(self, merged_graph_path, attack_type):
        """
        评测单个融合图
        
        Args:
            merged_graph_path: 融合图文件路径
            attack_type: 攻击类型
            
        Returns:
            dict: 评测结果
        """
        result = {
            'graph_path': merged_graph_path,
            'attack_type': attack_type,
            'metrics': {}
        }
        
        # 加载生成的图
        graph_data = self._load_generated_graph(merged_graph_path)
        if not graph_data:
            result['error'] = '无法加载图数据'
            return result
        
        # 指标1：主题重合F1值
        gt_event_types = self._extract_event_types_from_gt(attack_type)
        gen_event_types = self._extract_event_types_from_graph(graph_data)
        
        topic_precision, topic_recall, topic_f1 = self.calculate_f1(gen_event_types, gt_event_types)
        
        result['metrics']['topic_overlap'] = {
            'precision': topic_precision,
            'recall': topic_recall,
            'f1': topic_f1,
            'generated_types': list(gen_event_types),
            'ground_truth_types': list(gt_event_types),
            'matched_types': list(gen_event_types & gt_event_types)
        }
        
        # 指标2：事件序列匹配F1值（长度为2）
        gt_seq2 = self._extract_event_sequences_from_gt(attack_type, seq_length=2)
        gen_seq2 = self._extract_event_sequences_from_graph(graph_data, seq_length=2)
        
        seq2_precision, seq2_recall, seq2_f1 = self.calculate_f1(gen_seq2, gt_seq2)
        
        result['metrics']['sequence_2_matching'] = {
            'precision': seq2_precision,
            'recall': seq2_recall,
            'f1': seq2_f1,
            'generated_sequences': [list(s) for s in gen_seq2],
            'ground_truth_sequences': [list(s) for s in gt_seq2],
            'matched_sequences': [list(s) for s in (gen_seq2 & gt_seq2)]
        }
        
        # 指标2：事件序列匹配F1值（长度为3）
        gt_seq3 = self._extract_event_sequences_from_gt(attack_type, seq_length=3)
        gen_seq3 = self._extract_event_sequences_from_graph(graph_data, seq_length=3)
        
        seq3_precision, seq3_recall, seq3_f1 = self.calculate_f1(gen_seq3, gt_seq3)
        
        result['metrics']['sequence_3_matching'] = {
            'precision': seq3_precision,
            'recall': seq3_recall,
            'f1': seq3_f1,
            'generated_sequences': [list(s) for s in gen_seq3],
            'ground_truth_sequences': [list(s) for s in gt_seq3],
            'matched_sequences': [list(s) for s in (gen_seq3 & gt_seq3)]
        }
        
        return result
    
    def evaluate_all_models(self):
        """
        遍历result文件夹下的所有模型，对每个模型的每个攻击类型进行评测
        
        Returns:
            dict: 所有模型的评测结果
        """
        all_results = {}
        
        if not os.path.exists(self.result_root_path):
            self.logger.printLog(f"结果目录不存在: {self.result_root_path}")
            return all_results
        
        # 遍历所有模型文件夹
        for model_name in os.listdir(self.result_root_path):
            model_path = os.path.join(self.result_root_path, model_name)
            
            # 跳过非目录文件
            if not os.path.isdir(model_path):
                continue
            
            # 跳过隐藏文件夹
            if model_name.startswith('.'):
                continue
            
            self.logger.printLog(f"\n{'='*80}")
            self.logger.printLog(f"评测模型: {model_name}")
            self.logger.printLog(f"{'='*80}")
            
            model_results = {}
            
            # 遍历所有攻击类型文件夹
            for attack_type in os.listdir(model_path):
                attack_type_path = os.path.join(model_path, attack_type)
                
                # 跳过非目录文件
                if not os.path.isdir(attack_type_path):
                    continue
                
                self.logger.printLog(f"\n评测攻击类型: {attack_type}")
                
                # 查找融合图文件
                merged_graph_file = None
                for file in os.listdir(attack_type_path):
                    if file.startswith('merged_graph_') and file.endswith('.json'):
                        merged_graph_file = os.path.join(attack_type_path, file)
                        break
                
                if merged_graph_file:
                    self.logger.printLog(f"评测融合图: {os.path.basename(merged_graph_file)}")
                    result = self.evaluate_merged_graph(merged_graph_file, attack_type)
                    model_results[attack_type] = result
                    
                    # 打印评测结果
                    self._print_evaluation_result(result)
                else:
                    self.logger.printLog(f"警告: 未找到融合图文件")
                    model_results[attack_type] = {
                        'error': '未找到融合图文件'
                    }
            
            all_results[model_name] = model_results
        
        return all_results
    
    def _print_evaluation_result(self, result):
        """打印评测结果"""
        if 'error' in result:
            self.logger.printLog(f"  错误: {result['error']}")
            return
        
        metrics = result.get('metrics', {})
        
        # 打印主题重合F1值
        if 'topic_overlap' in metrics:
            topic = metrics['topic_overlap']
            self.logger.printLog(f"  [指标1] 主题重合F1值:")
            self.logger.printLog(f"    Precision: {topic['precision']:.4f}")
            self.logger.printLog(f"    Recall: {topic['recall']:.4f}")
            self.logger.printLog(f"    F1: {topic['f1']:.4f}")
            self.logger.printLog(f"    生成事件类型数: {len(topic['generated_types'])}")
            self.logger.printLog(f"    Ground Truth事件类型数: {len(topic['ground_truth_types'])}")
            self.logger.printLog(f"    匹配事件类型数: {len(topic['matched_types'])}")
        
        # 打印事件序列匹配F1值（长度为2）
        if 'sequence_2_matching' in metrics:
            seq2 = metrics['sequence_2_matching']
            self.logger.printLog(f"  [指标2] 事件序列匹配F1值 (长度=2):")
            self.logger.printLog(f"    Precision: {seq2['precision']:.4f}")
            self.logger.printLog(f"    Recall: {seq2['recall']:.4f}")
            self.logger.printLog(f"    F1: {seq2['f1']:.4f}")
            self.logger.printLog(f"    生成序列数: {len(seq2['generated_sequences'])}")
            self.logger.printLog(f"    Ground Truth序列数: {len(seq2['ground_truth_sequences'])}")
            self.logger.printLog(f"    匹配序列数: {len(seq2['matched_sequences'])}")
        
        # 打印事件序列匹配F1值（长度为3）
        if 'sequence_3_matching' in metrics:
            seq3 = metrics['sequence_3_matching']
            self.logger.printLog(f"  [指标2] 事件序列匹配F1值 (长度=3):")
            self.logger.printLog(f"    Precision: {seq3['precision']:.4f}")
            self.logger.printLog(f"    Recall: {seq3['recall']:.4f}")
            self.logger.printLog(f"    F1: {seq3['f1']:.4f}")
            self.logger.printLog(f"    生成序列数: {len(seq3['generated_sequences'])}")
            self.logger.printLog(f"    Ground Truth序列数: {len(seq3['ground_truth_sequences'])}")
            self.logger.printLog(f"    匹配序列数: {len(seq3['matched_sequences'])}")
    
    def save_results(self, results, output_path):
        """
        保存评测结果
        
        Args:
            results: 评测结果字典
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.printLog(f"\n评测结果已保存到: {output_path}")
        except Exception as e:
            self.logger.printLog(f"保存结果失败: {str(e)}")


def main():
    """主函数"""
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    ground_truth_path = os.path.join(
        project_root, 'dataset', 'processedData', 'extracted_data', 'event_graphs_train.json'
    )
    result_root_path = os.path.join(project_root, 'result')
    
    # 创建评测器
    evaluator = GraphEvaluator(ground_truth_path, result_root_path)
    
    # 执行评测
    evaluator.logger.printLog("="*80)
    evaluator.logger.printLog("开始评测所有模型")
    evaluator.logger.printLog("="*80)
    
    results = evaluator.evaluate_all_models()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(result_root_path, f"evaluation_results_{timestamp}.json")
    evaluator.save_results(results, output_path)
    
    evaluator.logger.printLog("\n" + "="*80)
    evaluator.logger.printLog("评测完成!")
    evaluator.logger.printLog("="*80)


if __name__ == '__main__':
    main()

