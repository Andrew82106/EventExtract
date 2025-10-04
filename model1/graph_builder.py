# -*- coding: utf-8 -*-
"""
图构建模块
负责使用LLM生成事件图
"""
import json
import networkx as nx
from llm_service import LLMService
from prompts import EXTRACT_EVENT_FLOW_PROMPT, STANDARDIZE_EVENT_TYPES_PROMPT
from event_validator import EventTypeValidator


class GraphBuilder:
    """图构建器类"""
    
    def __init__(self, llm_service, event_ontology, event_types=None):
        """
        初始化图构建器
        
        Args:
            llm_service (LLMService): LLM服务实例
            event_ontology (str): 事件本体描述文本
            event_types (list): 事件类型列表，用于验证（可选）
        """
        self.llm_service = llm_service
        self.event_ontology = event_ontology
        
        # 初始化验证器
        if event_types:
            self.validator = EventTypeValidator(event_types)
            print(f"  - 事件类型验证器已启用 (本体包含 {len(event_types)} 个事件类型)")
        else:
            self.validator = None
            print(f"  - 警告: 事件类型验证器未启用")
    
    def build_graph_from_text(self, text, text_id):
        """
        从单个文本构建事件图
        
        Args:
            text (str): 输入文本
            text_id (str): 文本标识符
            
        Returns:
            networkx.DiGraph: 事件图，如果失败返回None
        """
        # 第一步：提取事件流程
        print(f"  - 正在提取事件流程...")
        event_flow = self._extract_event_flow(text)
        if not event_flow:
            print(f"  - 提取事件流程失败")
            return None
        
        print(f"  - 提取到 {len(event_flow.get('events', []))} 个事件")
        
        # 第二步：标准化事件类型并构建图
        print(f"  - 正在标准化事件类型并构建图...")
        graph_data = self._standardize_and_build_graph(event_flow)
        if not graph_data:
            print(f"  - 标准化失败")
            return None
        
        # 转换为networkx图
        graph = self._convert_to_networkx(graph_data, text_id)
        print(f"  - 成功构建图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        return graph
    
    def _extract_event_flow(self, text):
        """
        使用LLM提取事件流程
        
        Args:
            text (str): 输入文本
            
        Returns:
            dict: 事件流程数据
        """
        # 如果文本太长，截取前3000字符
        if len(text) > 3000:
            text = text[:3000] + "..."
        
        prompt = EXTRACT_EVENT_FLOW_PROMPT.format(text=text)
        result = self.llm_service.call_api(prompt)
        
        if isinstance(result, dict) and 'events' in result:
            return result
        return None
    
    def _standardize_and_build_graph(self, event_flow):
        """
        标准化事件类型并构建图结构
        
        Args:
            event_flow (dict): 事件流程数据
            
        Returns:
            dict: 图数据（包含nodes和edges）
        """
        prompt = STANDARDIZE_EVENT_TYPES_PROMPT.format(
            event_flow=json.dumps(event_flow, ensure_ascii=False, indent=2),
            event_ontology=self.event_ontology
        )
        result = self.llm_service.call_api(prompt)
        
        if not isinstance(result, dict) or 'nodes' not in result or 'edges' not in result:
            return None
        
        # 如果启用了验证器，进行验证和修复
        if self.validator:
            validated_result, validation_report = self.validator.validate_graph_data(
                result, 
                auto_fix=True
            )
            
            # 打印验证结果
            total = validation_report['total_nodes']
            valid = validation_report['valid_nodes']
            fixed = validation_report['fixed_nodes']
            invalid = validation_report['invalid_nodes']
            
            print(f"  - 事件类型验证: {total}个节点, {valid}个有效, {fixed}个已修复, {invalid}个无效")
            
            if validation_report['node_issues']:
                print(f"  - 发现 {len(validation_report['node_issues'])} 个验证问题:")
                for issue_info in validation_report['node_issues'][:3]:  # 只显示前3个
                    print(f"    * 节点 {issue_info['node_id']}: {len(issue_info['issues'])} 个问题")
                if len(validation_report['node_issues']) > 3:
                    print(f"    * ... 还有 {len(validation_report['node_issues']) - 3} 个节点有问题")
            
            return validated_result
        else:
            # 未启用验证器，直接返回
            return result
    
    def _convert_to_networkx(self, graph_data, text_id):
        """
        将图数据转换为networkx图对象
        
        Args:
            graph_data (dict): 图数据
            text_id (str): 文本标识符
            
        Returns:
            networkx.DiGraph: 有向图
        """
        G = nx.DiGraph()
        
        # 添加节点
        for node in graph_data.get('nodes', []):
            node_id = f"{text_id}_{node['id']}"
            G.add_node(
                node_id,
                event_type=node.get('event_type', ''),
                event_subtype=node.get('event_subtype', ''),
                event_sub_subtype=node.get('event_sub_subtype', ''),
                description=node.get('description', ''),
                source_text=text_id
            )
        
        # 添加边
        for edge in graph_data.get('edges', []):
            source = f"{text_id}_{edge['source']}"
            target = f"{text_id}_{edge['target']}"
            if G.has_node(source) and G.has_node(target):
                G.add_edge(source, target, relation=edge.get('relation', 'before'))
        
        return G
    
    def save_graph_to_json(self, graph, output_path):
        """
        将图保存为JSON文件
        
        Args:
            graph (networkx.DiGraph): 图对象
            output_path (str): 输出路径
        """
        graph_data = {
            'nodes': [],
            'edges': []
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
    
    def get_validation_summary(self):
        """
        获取验证统计摘要
        
        Returns:
            dict: 验证统计信息，如果未启用验证器返回None
        """
        if self.validator:
            return self.validator.get_validation_summary()
        return None
    
    def print_validation_report(self):
        """打印详细的验证报告"""
        if self.validator:
            self.validator.print_validation_report()
        else:
            print("验证器未启用，无法生成报告")
    
    def reset_validation_stats(self):
        """重置验证统计"""
        if self.validator:
            self.validator.reset_stats()

