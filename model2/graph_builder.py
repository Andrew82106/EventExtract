# -*- coding: utf-8 -*-
"""
图构建模块 - Model2
负责使用LLM生成事件图，包含意义段分割和迭代验证
"""
import json
import networkx as nx
from llm_service import LLMService
from prompts import (
    SEGMENT_TEXT_PROMPT,
    EXTRACT_EVENT_FLOW_PROMPT,
    STANDARDIZE_EVENT_TYPES_WITH_SUPPORT_PROMPT,
    CHECK_SEGMENT_COVERAGE_PROMPT,
    APPLY_MODIFICATIONS_PROMPT
)
from event_validator import EventTypeValidator


class GraphBuilder:
    """图构建器类 - Model2版本"""
    
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
        
        # 初始化验证器（启用严格模式）
        if event_types:
            self.validator = EventTypeValidator(
                event_types, 
                strict_mode=True,  # 启用严格模式，拒绝无效节点
                similarity_threshold=0.7  # 提高相似度阈值
            )
            print(f"  - 事件类型验证器已启用 (严格模式, 本体包含 {len(event_types)} 个事件类型)")
        else:
            self.validator = None
            print(f"  - 警告: 事件类型验证器未启用")
    
    def build_graph_from_text(self, text, text_id):
        """
        从单个文本构建事件图（基于意义段的迭代方法）
        
        Args:
            text (str): 输入文本
            text_id (str): 文本标识符
            
        Returns:
            networkx.DiGraph: 事件图，如果失败返回None
        """
        # 第一步：将文本拆分为意义段
        print(f"  - 步骤1: 拆分意义段...")
        segments = self._segment_text(text)
        if not segments:
            print(f"  - 拆分意义段失败")
            return None
        
        print(f"  - 拆分得到 {len(segments)} 个意义段")
        
        # 第二步：从整个文本提取事件流程
        print(f"  - 步骤2: 提取事件流程...")
        event_flow = self._extract_event_flow(text)
        if not event_flow:
            print(f"  - 提取事件流程失败")
            return None
        
        print(f"  - 提取到 {len(event_flow.get('events', []))} 个事件")
        
        # 第三步：标准化事件类型并构建初始图（带支撑文本）
        print(f"  - 步骤3: 标准化事件类型并构建初始图...")
        graph_data = self._standardize_and_build_graph_with_support(text, event_flow)
        if not graph_data:
            print(f"  - 构建初始图失败")
            return None
        
        print(f"  - 初始图: {len(graph_data.get('nodes', []))} 个节点, {len(graph_data.get('edges', []))} 条边")
        
        # 第四步：迭代检查意义段覆盖情况并修改图
        print(f"  - 步骤4: 迭代验证意义段覆盖...")
        graph_data = self._iterative_segment_verification(segments, graph_data, text)
        
        print(f"  - 验证后图: {len(graph_data.get('nodes', []))} 个节点, {len(graph_data.get('edges', []))} 条边")
        
        # 转换为networkx图
        graph = self._convert_to_networkx(graph_data, text_id)
        print(f"  - 成功构建图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        return graph
    
    def _segment_text(self, text):
        """
        使用LLM将文本拆分为意义段
        
        Args:
            text (str): 输入文本
            
        Returns:
            list: 意义段列表
        """
        # 如果文本太长，截取前3000字符
        if len(text) > 3000:
            text = text[:3000] + "..."
        
        prompt = SEGMENT_TEXT_PROMPT.format(text=text)
        result = self.llm_service.call_api(prompt)
        
        if isinstance(result, dict) and 'segments' in result:
            return result['segments']
        return None
    
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
    
    def _standardize_and_build_graph_with_support(self, original_text, event_flow):
        """
        标准化事件类型并构建图结构（包含支撑文本）
        
        Args:
            original_text (str): 原始文本
            event_flow (dict): 事件流程数据
            
        Returns:
            dict: 图数据（包含nodes和edges）
        """
        # 如果文本太长，截取前3000字符
        if len(original_text) > 3000:
            original_text = original_text[:3000] + "..."
        
        prompt = STANDARDIZE_EVENT_TYPES_WITH_SUPPORT_PROMPT.format(
            original_text=original_text,
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
            
            return validated_result
        else:
            # 未启用验证器，直接返回
            return result
    
    def _iterative_segment_verification(self, segments, graph_data, original_text):
        """
        迭代检查每个意义段是否在图中有体现，并进行修改
        
        Args:
            segments (list): 意义段列表
            graph_data (dict): 当前图数据
            original_text (str): 原始文本
            
        Returns:
            dict: 更新后的图数据
        """
        modifications_made = 0
        
        for i, segment in enumerate(segments, 1):
            segment_content = segment.get('content', '')
            segment_core_meaning = segment.get('core_meaning', '')
            
            print(f"    - 检查意义段 {i}/{len(segments)}")
            
            # 检查这个意义段是否被图覆盖
            coverage_result = self._check_segment_coverage(
                segment_content, 
                segment_core_meaning, 
                graph_data
            )
            
            if not coverage_result:
                print(f"      * 跳过: 检查失败")
                continue
            
            is_covered = coverage_result.get('is_covered', False)
            suggestions = coverage_result.get('suggestions', [])
            
            if is_covered:
                print(f"      * 已覆盖")
            else:
                print(f"      * 未覆盖，应用 {len(suggestions)} 个修改建议")
                # 应用修改建议
                if suggestions:
                    graph_data = self._apply_modifications(graph_data, suggestions)
                    if graph_data:
                        modifications_made += 1
                        print(f"      * 修改成功")
                    else:
                        print(f"      * 修改失败")
        
        print(f"  - 共应用了 {modifications_made} 次修改")
        return graph_data
    
    def _check_segment_coverage(self, segment_content, segment_core_meaning, graph_data):
        """
        检查意义段是否在图中有体现
        
        Args:
            segment_content (str): 意义段内容
            segment_core_meaning (str): 意义段核心语义
            graph_data (dict): 当前图数据
            
        Returns:
            dict: 检查结果
        """
        prompt = CHECK_SEGMENT_COVERAGE_PROMPT.format(
            segment_content=segment_content,
            segment_core_meaning=segment_core_meaning,
            current_graph=json.dumps(graph_data, ensure_ascii=False, indent=2)
        )
        
        result = self.llm_service.call_api(prompt)
        
        if isinstance(result, dict):
            return result
        return None
    
    def _apply_modifications(self, graph_data, modifications):
        """
        应用修改建议更新图结构
        
        Args:
            graph_data (dict): 当前图数据
            modifications (list): 修改建议列表
            
        Returns:
            dict: 更新后的图数据
        """
        prompt = APPLY_MODIFICATIONS_PROMPT.format(
            current_graph=json.dumps(graph_data, ensure_ascii=False, indent=2),
            modifications=json.dumps(modifications, ensure_ascii=False, indent=2),
            event_ontology=self.event_ontology
        )
        
        result = self.llm_service.call_api(prompt)
        
        if not isinstance(result, dict) or 'nodes' not in result or 'edges' not in result:
            return graph_data  # 如果修改失败，返回原图
        
        # 如果启用了验证器，对修改后的图进行验证
        if self.validator:
            validated_result, validation_report = self.validator.validate_graph_data(
                result, 
                auto_fix=True
            )
            return validated_result
        
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
                support_text=node.get('support_text', ''),
                source_text=text_id
            )
        
        # 添加边
        for edge in graph_data.get('edges', []):
            source = f"{text_id}_{edge['source']}"
            target = f"{text_id}_{edge['target']}"
            if G.has_node(source) and G.has_node(target):
                G.add_edge(
                    source, 
                    target, 
                    relation=edge.get('relation', 'before'),
                    support_text=edge.get('support_text', '')
                )
        
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

