# -*- coding: utf-8 -*-
"""
重新合并图并评测
在现有的模型生成结果基础上，重新合并生成总图并进行评测
"""
import os
import sys
import json
import networkx as nx
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mergeGraph import GraphMerger
from utils import Logger
from evaluator import GraphEvaluator

# 硬编码的默认路径
DEFAULT_RESULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
DEFAULT_GROUND_TRUTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'dataset', 'processedData', 'extracted_data', 'event_graphs_test.json')


class RemergeTool:
    """重新合并工具类"""
    
    def __init__(self, result_root, ground_truth_path):
        """
        初始化工具
        
        Args:
            result_root: 结果根目录路径
            ground_truth_path: ground truth数据路径
        """
        self.result_root = result_root
        self.ground_truth_path = ground_truth_path
        self.logger = Logger()
        self.merger = GraphMerger()
    
    def load_graph_from_json(self, json_path):
        """
        从JSON文件加载图
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            networkx.DiGraph: 图对象
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            G = nx.DiGraph()
            
            # 添加节点
            for node in graph_data.get('nodes', []):
                node_id = node['id']
                attrs = {k: v for k, v in node.items() if k != 'id'}
                G.add_node(node_id, **attrs)
            
            # 添加边
            for edge in graph_data.get('edges', []):
                source = edge['source']
                target = edge['target']
                attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
                if G.has_node(source) and G.has_node(target):
                    G.add_edge(source, target, **attrs)
            
            return G
        except Exception as e:
            self.logger.printLog(f"加载图失败 {json_path}: {str(e)}")
            return None
    
    def load_graphs_from_directory(self, dir_path):
        """
        从目录中加载所有图文件（排除merged_graph文件）
        
        Args:
            dir_path: 目录路径
            
        Returns:
            list: 图对象列表
        """
        graphs = []
        
        if not os.path.exists(dir_path):
            self.logger.printLog(f"目录不存在: {dir_path}")
            return graphs
        
        # 获取所有以graph_开头且不是merged_graph的json文件
        files = []
        for filename in os.listdir(dir_path):
            if filename.startswith('graph_') and filename.endswith('.json') and not filename.startswith('merged_graph'):
                files.append(filename)
        
        # 按文件名排序
        files.sort(key=lambda x: int(x.replace('graph_', '').replace('.json', '')))
        
        self.logger.printLog(f"找到 {len(files)} 个图文件")
        
        for filename in files:
            filepath = os.path.join(dir_path, filename)
            graph = self.load_graph_from_json(filepath)
            if graph and graph.number_of_nodes() > 0:
                graphs.append(graph)
                self.logger.printLog(f"  - 加载: {filename} ({graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边)")
        
        return graphs
    
    def remerge_model_attack_type(self, model_name, attack_type):
        """
        重新合并指定模型和攻击类型的图
        
        Args:
            model_name: 模型名称 (如 model1, model2)
            attack_type: 攻击类型 (如 suicide_ied)
            
        Returns:
            bool: 是否成功
        """
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog(f"重新合并: {model_name} / {attack_type}")
        self.logger.printLog(f"{'='*80}")
        
        # 构建路径
        attack_dir = os.path.join(self.result_root, model_name, attack_type)
        
        if not os.path.exists(attack_dir):
            self.logger.printLog(f"错误: 目录不存在 {attack_dir}")
            return False
        
        # 加载所有单个图
        graphs = self.load_graphs_from_directory(attack_dir)
        
        if not graphs:
            self.logger.printLog(f"错误: 未找到可用的图文件")
            return False
        
        self.logger.printLog(f"成功加载 {len(graphs)} 个图")
        
        # 合并图
        self.logger.printLog(f"\n开始合并...")
        merged_graph = self.merger.merge_graphs(graphs)
        
        # 保存合并后的图
        merged_graph_path = os.path.join(attack_dir, f"merged_graph_{attack_type}.json")
        self.merger.save_merged_graph(merged_graph, merged_graph_path)
        
        self.logger.printLog(f"合并完成!")
        self.logger.printLog(f"- 输入图数量: {len(graphs)}")
        self.logger.printLog(f"- 融合图节点数: {merged_graph.number_of_nodes()}")
        self.logger.printLog(f"- 融合图边数: {merged_graph.number_of_edges()}")
        self.logger.printLog(f"- 保存路径: {merged_graph_path}")
        
        return True
    
    def remerge_all(self):
        """
        重新合并所有模型和攻击类型的图
        
        Returns:
            dict: 成功和失败的统计
        """
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog(f"开始重新合并所有模型结果")
        self.logger.printLog(f"结果目录: {self.result_root}")
        self.logger.printLog(f"{'='*80}")
        
        stats = {
            'success': [],
            'failed': []
        }
        
        # 遍历result目录下的所有模型文件夹
        if not os.path.exists(self.result_root):
            self.logger.printLog(f"错误: 结果目录不存在 {self.result_root}")
            return stats
        
        for model_name in os.listdir(self.result_root):
            model_path = os.path.join(self.result_root, model_name)
            
            # 跳过非目录和cache目录
            if not os.path.isdir(model_path) or model_name == 'cache':
                continue
            
            # 跳过非模型目录（不以model开头）
            if not model_name.startswith('model'):
                continue
            
            # 遍历该模型下的所有攻击类型文件夹
            for attack_type in os.listdir(model_path):
                attack_path = os.path.join(model_path, attack_type)
                
                if not os.path.isdir(attack_path):
                    continue
                
                # 重新合并
                success = self.remerge_model_attack_type(model_name, attack_type)
                
                if success:
                    stats['success'].append(f"{model_name}/{attack_type}")
                else:
                    stats['failed'].append(f"{model_name}/{attack_type}")
        
        # 输出统计
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog(f"重新合并完成!")
        self.logger.printLog(f"{'='*80}")
        self.logger.printLog(f"成功: {len(stats['success'])} 个")
        for item in stats['success']:
            self.logger.printLog(f"  ✓ {item}")
        
        if stats['failed']:
            self.logger.printLog(f"\n失败: {len(stats['failed'])} 个")
            for item in stats['failed']:
                self.logger.printLog(f"  ✗ {item}")
        
        return stats
    
    def remerge_and_evaluate(self, model_name=None, attack_type=None):
        """
        重新合并并评测
        
        Args:
            model_name: 模型名称，如果为None则处理所有模型
            attack_type: 攻击类型，如果为None则处理所有攻击类型
        """
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog(f"重新合并并评测")
        self.logger.printLog(f"{'='*80}")
        
        # 重新合并
        if model_name and attack_type:
            # 处理指定的模型和攻击类型
            success = self.remerge_model_attack_type(model_name, attack_type)
            if not success:
                self.logger.printLog("重新合并失败，跳过评测")
                return
        elif model_name:
            # 处理指定模型的所有攻击类型
            model_path = os.path.join(self.result_root, model_name)
            if not os.path.exists(model_path):
                self.logger.printLog(f"错误: 模型目录不存在 {model_path}")
                return
            
            for attack_type_dir in os.listdir(model_path):
                attack_path = os.path.join(model_path, attack_type_dir)
                if os.path.isdir(attack_path):
                    self.remerge_model_attack_type(model_name, attack_type_dir)
        else:
            # 处理所有模型
            self.remerge_all()
        
        # 评测
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog(f"开始评测")
        self.logger.printLog(f"{'='*80}")
        
        evaluator = GraphEvaluator(self.ground_truth_path, self.result_root)
        results = evaluator.evaluate_all_models()
        
        # 保存评测结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(self.result_root, f"evaluation_results_{timestamp}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.printLog(f"\n评测结果已保存到: {output_path}")
        
        # 输出简要统计
        self.logger.printLog(f"\n{'='*80}")
        self.logger.printLog(f"评测结果摘要")
        self.logger.printLog(f"{'='*80}")
        
        for model_name, model_results in results.items():
            if model_name == 'evaluation_time':
                continue
            
            self.logger.printLog(f"\n{model_name}:")
            for attack_type, metrics in model_results.items():
                if isinstance(metrics, dict) and 'topic_f1' in metrics:
                    self.logger.printLog(f"  {attack_type}:")
                    self.logger.printLog(f"    - 主题F1: {metrics['topic_f1']:.4f}")
                    self.logger.printLog(f"    - 序列F1 (长度2): {metrics['sequence_f1_len2']:.4f}")
                    self.logger.printLog(f"    - 序列F1 (长度3): {metrics['sequence_f1_len3']:.4f}")


def remerge_only(model_name=None, attack_type=None, result_root=DEFAULT_RESULT_ROOT):
    """
    只重新合并，不评测
    
    Args:
        model_name: 模型名称 (如 'model1', 'model2')，不指定则处理所有模型
        attack_type: 攻击类型 (如 'suicide_ied')，不指定则处理所有攻击类型
        result_root: 结果根目录路径
    
    Returns:
        dict: 成功和失败的统计信息
    """
    tool = RemergeTool(result_root, DEFAULT_GROUND_TRUTH)
    
    if model_name and attack_type:
        # 处理指定的模型和攻击类型
        success = tool.remerge_model_attack_type(model_name, attack_type)
        return {'success': [f"{model_name}/{attack_type}"] if success else [], 
                'failed': [] if success else [f"{model_name}/{attack_type}"]}
    elif model_name:
        # 处理指定模型的所有攻击类型
        model_path = os.path.join(result_root, model_name)
        stats = {'success': [], 'failed': []}
        if os.path.exists(model_path):
            for attack_type_dir in os.listdir(model_path):
                attack_path = os.path.join(model_path, attack_type_dir)
                if os.path.isdir(attack_path):
                    success = tool.remerge_model_attack_type(model_name, attack_type_dir)
                    if success:
                        stats['success'].append(f"{model_name}/{attack_type_dir}")
                    else:
                        stats['failed'].append(f"{model_name}/{attack_type_dir}")
        return stats
    else:
        # 处理所有模型
        return tool.remerge_all()


def remerge_and_evaluate(model_name=None, attack_type=None, 
                         result_root=DEFAULT_RESULT_ROOT, 
                         ground_truth_path=DEFAULT_GROUND_TRUTH):
    """
    重新合并并评测
    
    Args:
        model_name: 模型名称 (如 'model1', 'model2')，不指定则处理所有模型
        attack_type: 攻击类型 (如 'suicide_ied')，不指定则处理所有攻击类型
        result_root: 结果根目录路径
        ground_truth_path: ground truth数据路径
    
    Returns:
        dict: 评测结果
    """
    tool = RemergeTool(result_root, ground_truth_path)
    tool.remerge_and_evaluate(model_name, attack_type)


def main():
    """
    主函数 - 默认重新合并所有模型并评测
    
    如果需要自定义参数，请直接调用以下函数：
    - remerge_only(model_name, attack_type, result_root) - 只重新合并
    - remerge_and_evaluate(model_name, attack_type, result_root, ground_truth_path) - 重新合并并评测
    """
    # 默认：重新合并所有模型并评测
    remerge_and_evaluate()


if __name__ == '__main__':
    main()

