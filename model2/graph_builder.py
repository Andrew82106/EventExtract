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
    CLASSIFY_EVENT_TYPE_STEP1_PROMPT,
    CLASSIFY_EVENT_TYPE_STEP2_PROMPT,
    CLASSIFY_EVENT_TYPE_STEP3_PROMPT,
    BUILD_GRAPH_WITH_CLASSIFIED_EVENTS_PROMPT,
    CHECK_SEGMENT_RELEVANCE_PROMPT,
    EXTRACT_VERBS_FROM_SEGMENT_PROMPT,
    QUICK_CHECK_VERB_COVERAGE_PROMPT,
    CHECK_VERB_COVERAGE_PROMPT
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
        self.event_types = event_types
        
        # 构建层次化分类索引
        if event_types:
            self._build_hierarchical_indices()
        
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
        
        # 用于记录低置信度分类
        self.low_confidence_classifications = []
    
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
    
    def _build_hierarchical_indices(self):
        """构建层次化分类索引，用于快速查询各层级的可选项"""
        self.top_types = set()  # 顶层类型集合
        self.subtypes_by_type = {}  # {type: set(subtypes)}
        self.sub_subtypes_by_type_subtype = {}  # {(type, subtype): set(sub_subtypes)}
        
        for event in self.event_types:
            event_type = event['type']
            subtype = event['subtype'] if event['subtype'] else ''
            sub_subtype = event['sub_subtype'] if event['sub_subtype'] else ''
            
            # 添加顶层类型
            self.top_types.add(event_type)
            
            # 添加子类型
            if event_type not in self.subtypes_by_type:
                self.subtypes_by_type[event_type] = set()
            if subtype:
                self.subtypes_by_type[event_type].add(subtype)
            
            # 添加子子类型
            key = (event_type, subtype)
            if key not in self.sub_subtypes_by_type_subtype:
                self.sub_subtypes_by_type_subtype[key] = set()
            if sub_subtype:
                self.sub_subtypes_by_type_subtype[key].add(sub_subtype)
        
        print(f"  - 层次化索引构建完成: {len(self.top_types)} 个顶层类型")
    
    def _get_types_with_definitions(self, type_names):
        """
        获取顶层类型及其定义的格式化字符串
        
        Args:
            type_names (list): 类型名称列表
            
        Returns:
            str: 格式化的类型定义字符串
        """
        definitions = []
        # 为每个类型收集其第一个定义（从完整本体中）
        type_defs = {}
        for event in self.event_types:
            event_type = event['type']
            if event_type not in type_defs and event.get('definition'):
                type_defs[event_type] = event['definition']
        
        for type_name in type_names:
            if type_name in type_defs:
                definitions.append(f"- {type_name}: {type_defs[type_name]}")
            else:
                definitions.append(f"- {type_name}")
        
        return "\n".join(definitions)
    
    def _get_subtypes_with_definitions(self, parent_type, subtype_names):
        """
        获取子类型及其定义的格式化字符串
        
        Args:
            parent_type (str): 父类型名称
            subtype_names (list): 子类型名称列表
            
        Returns:
            str: 格式化的子类型定义字符串
        """
        definitions = []
        # 为每个子类型收集其定义
        subtype_defs = {}
        for event in self.event_types:
            if event['type'] == parent_type and event.get('subtype') and event.get('definition'):
                subtype = event['subtype']
                if subtype not in subtype_defs:
                    subtype_defs[subtype] = event['definition']
        
        for subtype_name in subtype_names:
            if subtype_name in subtype_defs:
                definitions.append(f"- {subtype_name}: {subtype_defs[subtype_name]}")
            else:
                definitions.append(f"- {subtype_name}")
        
        return "\n".join(definitions)
    
    def _get_sub_subtypes_with_definitions(self, parent_type, parent_subtype, sub_subtype_names):
        """
        获取子子类型及其定义的格式化字符串
        
        Args:
            parent_type (str): 父类型名称
            parent_subtype (str): 父子类型名称
            sub_subtype_names (list): 子子类型名称列表
            
        Returns:
            str: 格式化的子子类型定义字符串
        """
        definitions = []
        # 为每个子子类型收集其定义
        sub_subtype_defs = {}
        for event in self.event_types:
            if (event['type'] == parent_type and 
                event.get('subtype') == parent_subtype and 
                event.get('sub_subtype') and 
                event.get('definition')):
                sub_subtype = event['sub_subtype']
                if sub_subtype not in sub_subtype_defs:
                    sub_subtype_defs[sub_subtype] = event['definition']
        
        for sub_subtype_name in sub_subtype_names:
            if sub_subtype_name in sub_subtype_defs:
                definitions.append(f"- {sub_subtype_name}: {sub_subtype_defs[sub_subtype_name]}")
            else:
                definitions.append(f"- {sub_subtype_name}")
        
        return "\n".join(definitions)
    
    def _classify_event_hierarchical(self, event_description):
        """
        层次化分类：分三步对事件进行分类
        
        Args:
            event_description (str): 事件描述
            
        Returns:
            dict: 分类结果 {'type': str, 'subtype': str, 'sub_subtype': str, 'confidence': float}
        """
        if not self.event_types:
            return None
        
        # 第一步：选择顶层类型（Type）
        available_types = sorted(list(self.top_types))
        # 构建带定义的类型描述
        types_with_definitions = self._get_types_with_definitions(available_types)
        prompt1 = CLASSIFY_EVENT_TYPE_STEP1_PROMPT.format(
            event_description=event_description,
            available_types=types_with_definitions
        )
        result1 = self.llm_service.call_api(prompt1)
        
        if not isinstance(result1, dict) or 'selected_type' not in result1:
            print(f"      ! 第一步分类失败")
            return None
        
        selected_type = result1.get('selected_type', '').strip()
        confidence1 = result1.get('confidence', 0.5)
        
        if selected_type not in self.top_types:
            print(f"      ! 选择的类型 '{selected_type}' 不在本体中")
            return None
        
        # 第二步：选择子类型（Subtype）
        available_subtypes = sorted(list(self.subtypes_by_type.get(selected_type, set())))
        if not available_subtypes:
            print(f"      ! 类型 '{selected_type}' 没有子类型")
            return None
        
        # 构建带定义的子类型描述
        subtypes_with_definitions = self._get_subtypes_with_definitions(selected_type, available_subtypes)
        prompt2 = CLASSIFY_EVENT_TYPE_STEP2_PROMPT.format(
            event_description=event_description,
            selected_type=selected_type,
            available_subtypes=subtypes_with_definitions
        )
        result2 = self.llm_service.call_api(prompt2)
        
        if not isinstance(result2, dict) or 'selected_subtype' not in result2:
            print(f"      ! 第二步分类失败")
            return None
        
        selected_subtype = result2.get('selected_subtype', '').strip()
        confidence2 = result2.get('confidence', 0.5)
        
        if selected_subtype not in available_subtypes:
            print(f"      ! 选择的子类型 '{selected_subtype}' 不在本体中")
            return None
        
        # 第三步：选择子子类型（Sub_subtype）
        key = (selected_type, selected_subtype)
        available_sub_subtypes = sorted(list(self.sub_subtypes_by_type_subtype.get(key, set())))
        if not available_sub_subtypes:
            print(f"      ! 组合 '{selected_type}.{selected_subtype}' 没有子子类型")
            return None
        
        # 构建带定义的子子类型描述
        sub_subtypes_with_definitions = self._get_sub_subtypes_with_definitions(
            selected_type, selected_subtype, available_sub_subtypes
        )
        prompt3 = CLASSIFY_EVENT_TYPE_STEP3_PROMPT.format(
            event_description=event_description,
            selected_type=selected_type,
            selected_subtype=selected_subtype,
            available_sub_subtypes=sub_subtypes_with_definitions
        )
        result3 = self.llm_service.call_api(prompt3)
        
        if not isinstance(result3, dict) or 'selected_sub_subtype' not in result3:
            print(f"      ! 第三步分类失败")
            return None
        
        selected_sub_subtype = result3.get('selected_sub_subtype', '').strip()
        confidence3 = result3.get('confidence', 0.5)
        
        if selected_sub_subtype not in available_sub_subtypes:
            print(f"      ! 选择的子子类型 '{selected_sub_subtype}' 不在本体中")
            return None
        
        # 综合置信度：取三步的平均值
        overall_confidence = (confidence1 + confidence2 + confidence3) / 3.0
        
        return {
            'type': selected_type,
            'subtype': selected_subtype,
            'sub_subtype': selected_sub_subtype,
            'confidence': overall_confidence
        }
    
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
        标准化事件类型并构建图结构（采用层次化分类策略）
        
        Args:
            original_text (str): 原始文本
            event_flow (dict): 事件流程数据
            
        Returns:
            dict: 图数据（包含nodes和edges）
        """
        # 如果文本太长，截取前3000字符
        if len(original_text) > 3000:
            original_text = original_text[:3000] + "..."
        
        # 步骤1: 对每个事件进行层次化分类
        events = event_flow.get('events', [])
        classified_events = []
        
        print(f"    - 开始层次化分类 {len(events)} 个事件...")
        
        for event in events:
            event_desc = event.get('description', '')
            
            # 三步层次化分类
            classified = self._classify_event_hierarchical(event_desc)
            
            if classified:
                classified_events.append({
                    'event_id': event.get('event_id'),
                    'description': event_desc,
                    'event_type': classified['type'],
                    'event_subtype': classified['subtype'],
                    'event_sub_subtype': classified['sub_subtype'],
                    'confidence': classified['confidence']
                })
                
                # 记录低置信度分类
                if classified['confidence'] < 0.6:
                    self.low_confidence_classifications.append({
                        'description': event_desc,
                        'classification': f"{classified['type']}.{classified['subtype']}.{classified['sub_subtype']}",
                        'confidence': classified['confidence']
                    })
                    print(f"      ! 低置信度分类 ({classified['confidence']:.2f}): {event_desc[:50]}...")
        
        if not classified_events:
            print(f"    - 层次化分类失败，没有成功分类的事件")
            return None
        
        print(f"    - 成功分类 {len(classified_events)} 个事件")
        
        # 步骤2: 构建图结构
        prompt = BUILD_GRAPH_WITH_CLASSIFIED_EVENTS_PROMPT.format(
            original_text=original_text,
            classified_events=json.dumps(classified_events, ensure_ascii=False, indent=2)
        )
        result = self.llm_service.call_api(prompt)
        
        if not isinstance(result, dict) or 'nodes' not in result or 'edges' not in result:
            return None
        
        # 如果启用了验证器，进行验证和修复
        # 注意：由于已经进行了层次化分类，这里主要是最后的保障
        if self.validator:
            # 保存原始设置
            original_strict_mode = self.validator.strict_mode
            original_threshold = self.validator.similarity_threshold
            
            # 临时使用宽松策略
            # 1. 关闭严格模式（允许保留无效节点但尝试修复）
            # 2. 降低相似度阈值到0.5，提高修复成功率
            self.validator.strict_mode = False
            self.validator.similarity_threshold = 0.5
            
            validated_result, validation_report = self.validator.validate_graph_data(
                result, 
                auto_fix=True
            )
            
            # 恢复原始设置
            self.validator.strict_mode = original_strict_mode
            self.validator.similarity_threshold = original_threshold
            
            # 打印验证结果
            total = validation_report['total_nodes']
            valid = validation_report['valid_nodes']
            fixed = validation_report['fixed_nodes']
            invalid = validation_report['invalid_nodes']
            rejected = validation_report['rejected_nodes']
            
            if invalid > 0:
                print(f"  - 事件类型验证: {total}个节点, {valid}个有效, {fixed}个已修复, {invalid}个无效但保留")
            else:
                print(f"  - 事件类型验证: {total}个节点, {valid}个有效, {fixed}个已修复")
            
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
            
            # 步骤8.1：判断文本段是否对事件有推进作用
            is_relevant = self._check_segment_relevance(
                original_text,
                segment_content,
                segment_core_meaning
            )
            
            if not is_relevant:
                print(f"      * 跳过: 无推进作用")
                continue
            
            # 步骤8.2：基于动词的覆盖检查
            coverage_result = self._check_segment_coverage(
                segment_content, 
                segment_core_meaning, 
                graph_data
            )
            
            if not coverage_result:
                print(f"      * 跳过: 检查失败")
                continue
            
            suggestions = coverage_result.get('suggestions', [])
            
            if not suggestions:
                print(f"      * 所有动词事件已覆盖")
            else:
                print(f"      * 收到 {len(suggestions)} 个修改建议")
                # 步骤8.3和8.4：验证修改意见并应用到图结构
                success, graph_data = self._validate_and_apply_modifications(graph_data, suggestions)
                if success:
                    modifications_made += 1
                    print(f"      * 修改成功")
                else:
                    print(f"      * 修改失败或被拒绝")
        
        print(f"  - 共应用了 {modifications_made} 次修改")
        return graph_data
    
    def _check_segment_relevance(self, original_text, segment_content, segment_core_meaning):
        """
        步骤8.1：判断文本段是否对事件有推进作用
        
        Args:
            original_text (str): 原始完整文本
            segment_content (str): 意义段内容
            segment_core_meaning (str): 意义段核心语义
            
        Returns:
            bool: 是否有推进作用
        """
        # 如果文本太长，截取前3000字符
        if len(original_text) > 3000:
            original_text = original_text[:3000] + "..."
        
        prompt = CHECK_SEGMENT_RELEVANCE_PROMPT.format(
            original_text=original_text,
            segment_content=segment_content,
            segment_core_meaning=segment_core_meaning
        )
        
        result = self.llm_service.call_api(prompt)
        
        if isinstance(result, dict) and 'is_relevant' in result:
            return result['is_relevant']
        return False  # 如果失败，保守地认为无推进作用
    
    def _check_segment_coverage(self, segment_content, segment_core_meaning, graph_data):
        """
        步骤8.2：基于动词的意义段覆盖检查（优化版：先检查后分类）
        
        Args:
            segment_content (str): 意义段内容
            segment_core_meaning (str): 意义段核心语义
            graph_data (dict): 当前图数据
            
        Returns:
            dict: 检查结果
        """
        # 步骤1：提取意义段中的关键动词
        print(f"        * 步骤1: 提取动词...")
        verbs_result = self._extract_verbs_from_segment(segment_content)
        
        if not verbs_result or 'verbs' not in verbs_result:
            print(f"        ! 动词提取失败")
            return None
        
        # 过滤出表示事件的动词
        event_verbs = [v for v in verbs_result['verbs'] if v.get('is_event', False)]
        
        if not event_verbs:
            print(f"        * 未提取到事件动词，跳过")
            return {'is_covered': True, 'suggestions': [], 'explanation': '无事件动词'}
        
        print(f"        * 提取到 {len(event_verbs)} 个事件动词")
        
        # 步骤2：快速预检查哪些动词已经在图中覆盖
        print(f"        * 步骤2: 快速检查覆盖情况...")
        quick_check = self._quick_check_verb_coverage(event_verbs, graph_data)
        
        if not quick_check:
            print(f"        ! 快速检查失败")
            return None
        
        # 找出未覆盖的动词
        uncovered_verbs = [
            v for v in quick_check.get('coverage_check', []) 
            if not v.get('is_covered', False)
        ]
        
        covered_count = len(event_verbs) - len(uncovered_verbs)
        print(f"        * 已覆盖: {covered_count}/{len(event_verbs)}, 需分类: {len(uncovered_verbs)}")
        
        if not uncovered_verbs:
            # 所有动词都已覆盖
            return {'is_covered': True, 'suggestions': [], 'explanation': '所有动词事件已覆盖'}
        
        # 步骤3：只对未覆盖的动词进行分类
        print(f"        * 步骤3: 对未覆盖的动词分类...")
        
        # 从原始event_verbs中找到未覆盖的动词对象
        uncovered_verb_names = [v['verb'] for v in uncovered_verbs]
        verbs_to_classify = [
            v for v in event_verbs 
            if v.get('verb', '') in uncovered_verb_names
        ]
        
        classified_verbs = self._classify_verb_events(verbs_to_classify)
        
        if not classified_verbs:
            print(f"        ! 动词分类失败")
            return None
        
        print(f"        * 成功分类 {len(classified_verbs)} 个动词事件")
        
        # 步骤4：检查分类后的动词事件并生成修改建议
        print(f"        * 步骤4: 生成修改建议...")
        coverage_result = self._check_verb_coverage(classified_verbs, graph_data)
        
        if not coverage_result:
            print(f"        ! 生成建议失败")
            return None
        
        return coverage_result
    
    def _extract_verbs_from_segment(self, segment_content):
        """
        从意义段中提取关键动词及其语义
        
        Args:
            segment_content (str): 意义段内容
            
        Returns:
            dict: 动词列表
        """
        prompt = EXTRACT_VERBS_FROM_SEGMENT_PROMPT.format(
            segment_content=segment_content
        )
        
        result = self.llm_service.call_api(prompt)
        
        if isinstance(result, dict) and 'verbs' in result:
            return result
        return None
    
    def _quick_check_verb_coverage(self, event_verbs, graph_data):
        """
        快速预检查动词是否已在图中覆盖（无需分类）
        
        Args:
            event_verbs (list): 事件动词列表
            graph_data (dict): 当前图数据
            
        Returns:
            dict: 快速检查结果
        """
        # 准备图节点信息
        graph_nodes = []
        for node in graph_data.get('nodes', []):
            graph_nodes.append({
                'id': node.get('id', ''),
                'description': node.get('description', ''),
                'event_type': f"{node.get('event_type', '')}.{node.get('event_subtype', '')}.{node.get('event_sub_subtype', '')}"
            })
        
        # 准备动词信息
        verbs_info = []
        for verb in event_verbs:
            verbs_info.append({
                'verb': verb.get('verb', ''),
                'semantic_description': verb.get('semantic_description', '')
            })
        
        prompt = QUICK_CHECK_VERB_COVERAGE_PROMPT.format(
            verbs_list=json.dumps(verbs_info, ensure_ascii=False, indent=2),
            graph_nodes=json.dumps(graph_nodes, ensure_ascii=False, indent=2)
        )
        
        result = self.llm_service.call_api(prompt)
        
        if isinstance(result, dict) and 'coverage_check' in result:
            return result
        return None
    
    def _classify_verb_events(self, event_verbs):
        """
        对提取的动词进行事件分类
        
        Args:
            event_verbs (list): 事件动词列表
            
        Returns:
            list: 分类后的动词事件列表
        """
        classified_verbs = []
        
        # 获取顶层类型及定义
        available_types = sorted(list(self.top_types))
        types_with_definitions = self._get_types_with_definitions(available_types)
        
        # 为每个动词进行层次化分类
        for verb in event_verbs:
            verb_desc = verb.get('semantic_description', '')
            
            # 使用层次化分类
            classified = self._classify_event_hierarchical(verb_desc)
            
            if classified:
                classified_verb = {
                    'verb': verb.get('verb', ''),
                    'semantic_description': verb_desc,
                    'subject': verb.get('subject', ''),
                    'object': verb.get('object', ''),
                    'event_type': classified['type'],
                    'event_subtype': classified['subtype'],
                    'event_sub_subtype': classified['sub_subtype'],
                    'confidence': classified['confidence'],
                    'temporal_order': verb.get('temporal_order', 1)
                }
                classified_verbs.append(classified_verb)
        
        return classified_verbs
    
    def _check_verb_coverage(self, classified_verbs, graph_data):
        """
        检查每个动词事件是否在图中有体现
        
        Args:
            classified_verbs (list): 分类后的动词事件列表
            graph_data (dict): 当前图数据
            
        Returns:
            dict: 覆盖检查结果
        """
        prompt = CHECK_VERB_COVERAGE_PROMPT.format(
            classified_verbs=json.dumps(classified_verbs, ensure_ascii=False, indent=2),
            current_graph=json.dumps(graph_data, ensure_ascii=False, indent=2)
        )
        
        result = self.llm_service.call_api(prompt)
        
        if isinstance(result, dict):
            return result
        return None
    
    def _validate_and_apply_modifications(self, graph_data, suggestions):
        """
        步骤8.3和8.4：验证修改意见的合法性并应用到图结构
        
        Args:
            graph_data (dict): 当前图数据
            suggestions (list): 修改建议列表
            
        Returns:
            tuple: (success, updated_graph_data)
        """
        # 创建一个工作副本，避免修改原数据
        working_graph = {
            'nodes': [node.copy() for node in graph_data.get('nodes', [])],
            'edges': [edge.copy() for edge in graph_data.get('edges', [])]
        }
        
        # 创建临时的networkx图用于环检测
        temp_nx_graph = nx.DiGraph()
        
        # 添加现有节点到临时图
        for node in working_graph['nodes']:
            temp_nx_graph.add_node(node['id'])
        
        # 添加现有边到临时图
        for edge in working_graph['edges']:
            if temp_nx_graph.has_node(edge['source']) and temp_nx_graph.has_node(edge['target']):
                temp_nx_graph.add_edge(edge['source'], edge['target'])
        
        # 跟踪是否有成功的修改
        any_success = False
        
        # 处理每个修改建议
        for suggestion in suggestions:
            action = suggestion.get('action', '')
            details = suggestion.get('details', {})
            
            if action == 'add_node':
                success = self._apply_add_node(working_graph, temp_nx_graph, details)
                if success:
                    any_success = True
                    
            elif action == 'add_edge':
                success = self._apply_add_edge(working_graph, temp_nx_graph, details)
                if success:
                    any_success = True
                    
            elif action == 'modify_node':
                success = self._apply_modify_node(working_graph, details)
                if success:
                    any_success = True
        
        return any_success, working_graph
    
    def _apply_add_node(self, graph_data, nx_graph, details):
        """
        应用添加节点操作（步骤8.4）
        采用"宽松接受，严格验证"策略：降低相似度阈值来修复，但只添加合法节点
        
        Args:
            graph_data (dict): 图数据
            nx_graph (nx.DiGraph): networkx图对象（用于环检测）
            details (dict): 节点详细信息
            
        Returns:
            bool: 是否成功
        """
        node_id = details.get('id')
        event_type = details.get('event_type', '').strip()
        event_subtype = details.get('event_subtype', '').strip()
        event_sub_subtype = details.get('event_sub_subtype', '').strip()
        
        # 检查节点ID是否已存在
        if any(node['id'] == node_id for node in graph_data['nodes']):
            print(f"        ! 节点ID {node_id} 已存在，跳过")
            return False
        
        # 步骤8.3：验证事件类型是否在本体中，采用宽松的相似度阈值进行修复
        if self.validator:
            # 保存原始相似度阈值
            original_threshold = self.validator.similarity_threshold
            
            # 临时降低相似度阈值到0.5，提高修复成功率
            self.validator.similarity_threshold = 0.5
            
            temp_node = {
                'id': node_id,
                'event_type': event_type,
                'event_subtype': event_subtype,
                'event_sub_subtype': event_sub_subtype
            }
            is_valid, fixed_node, issues = self.validator.validate_node(temp_node, auto_fix=True)
            
            # 恢复原始阈值
            self.validator.similarity_threshold = original_threshold
            
            if not is_valid:
                print(f"        ! 节点 {node_id} 事件类型 ({event_type}.{event_subtype}.{event_sub_subtype}) 无效且无法修复，拒绝添加")
                return False
            
            # 使用修复后的事件类型
            fixed_event_type = fixed_node['event_type']
            fixed_event_subtype = fixed_node['event_subtype']
            fixed_event_sub_subtype = fixed_node['event_sub_subtype']
            
            # 如果修复后的类型与原始类型不同，打印修复信息
            if (fixed_event_type != event_type or 
                fixed_event_subtype != event_subtype or 
                fixed_event_sub_subtype != event_sub_subtype):
                print(f"        * 修复节点类型: {event_type}.{event_subtype}.{event_sub_subtype}")
                print(f"          -> {fixed_event_type}.{fixed_event_subtype}.{fixed_event_sub_subtype}")
            
            event_type = fixed_event_type
            event_subtype = fixed_event_subtype
            event_sub_subtype = fixed_event_sub_subtype
        
        # 添加节点到图数据
        new_node = {
            'id': node_id,
            'event_type': event_type,
            'event_subtype': event_subtype,
            'event_sub_subtype': event_sub_subtype,
            'description': details.get('description', ''),
            'support_text': details.get('support_text', '')
        }
        graph_data['nodes'].append(new_node)
        nx_graph.add_node(node_id)
        
        print(f"        + 添加节点: {node_id} ({event_type}.{event_subtype}.{event_sub_subtype})")
        return True
    
    def _apply_add_edge(self, graph_data, nx_graph, details):
        """
        应用添加边操作（步骤8.4）
        检查：节点存在性、边重复、是否成环
        
        Args:
            graph_data (dict): 图数据
            nx_graph (nx.DiGraph): networkx图对象（用于环检测）
            details (dict): 边详细信息
            
        Returns:
            bool: 是否成功
        """
        source = details.get('source')
        target = details.get('target')
        
        # 步骤8.3.1：检查源节点是否存在
        if not nx_graph.has_node(source):
            print(f"        ! 源节点 {source} 不存在，跳过边的添加")
            return False
        
        # 步骤8.3.2：检查目标节点是否存在
        if not nx_graph.has_node(target):
            print(f"        ! 目标节点 {target} 不存在，跳过边的添加")
            return False
        
        # 步骤8.3.3：检查边是否已存在
        if any(e['source'] == source and e['target'] == target for e in graph_data['edges']):
            print(f"        ! 边 {source} -> {target} 已存在，跳过")
            return False
        
        # 步骤8.3.4：检查是否会形成环（使用networkx的环检测）
        # 临时添加边到nx图中
        nx_graph.add_edge(source, target)
        
        try:
            # 使用simple_cycles检测是否有环
            cycles = list(nx.simple_cycles(nx_graph))
            if cycles:
                # 发现环，撤销添加并拒绝
                nx_graph.remove_edge(source, target)
                print(f"        ! 添加边 {source} -> {target} 会形成环，拒绝")
                if len(cycles[0]) <= 5:  # 只打印小环的详情
                    print(f"          环: {' -> '.join(str(n) for n in cycles[0])}")
                return False
        except Exception as e:
            # 如果检测失败，保守地拒绝
            nx_graph.remove_edge(source, target)
            print(f"        ! 环检测异常，拒绝添加边 {source} -> {target}: {str(e)}")
            return False
        
        # 边已经在nx_graph中，现在添加到图数据中
        new_edge = {
            'source': source,
            'target': target,
            'relation': details.get('relation', 'before'),
            'support_text': details.get('support_text', '')
        }
        graph_data['edges'].append(new_edge)
        
        print(f"        + 添加边: {source} -> {target}")
        return True
    
    def _apply_modify_node(self, graph_data, details):
        """
        应用修改节点操作（步骤8.4）
        
        Args:
            graph_data (dict): 图数据
            details (dict): 修改详细信息
            
        Returns:
            bool: 是否成功
        """
        node_id = details.get('id')
        
        # 查找节点
        target_node = None
        for node in graph_data['nodes']:
            if node['id'] == node_id:
                target_node = node
                break
        
        if not target_node:
            print(f"        ! 节点 {node_id} 不存在，无法修改")
            return False
        
        # 更新节点属性（只更新提供的字段）
        if 'description' in details:
            target_node['description'] = details['description']
        if 'support_text' in details:
            target_node['support_text'] = details['support_text']
        
        print(f"        * 修改节点: {node_id}")
        return True
    
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
    
    def get_low_confidence_report(self):
        """
        获取低置信度分类报告
        
        Returns:
            dict: 低置信度分类统计和详情
        """
        total_count = len(self.low_confidence_classifications)
        
        if total_count == 0:
            return {
                'total_low_confidence': 0,
                'average_confidence': None,
                'classifications': []
            }
        
        avg_confidence = sum(c['confidence'] for c in self.low_confidence_classifications) / total_count
        
        return {
            'total_low_confidence': total_count,
            'average_confidence': avg_confidence,
            'classifications': self.low_confidence_classifications
        }
    
    def print_low_confidence_report(self):
        """打印低置信度分类报告"""
        report = self.get_low_confidence_report()
        
        if report['total_low_confidence'] == 0:
            print("\n没有低置信度分类记录")
            return
        
        print("\n" + "="*60)
        print("低置信度分类报告 (置信度 < 0.6)")
        print("="*60)
        print(f"总数: {report['total_low_confidence']}")
        print(f"平均置信度: {report['average_confidence']:.3f}")
        print("\n详细列表:")
        print("-"*60)
        
        for i, item in enumerate(report['classifications'][:10], 1):
            print(f"\n{i}. 置信度: {item['confidence']:.3f}")
            print(f"   分类: {item['classification']}")
            print(f"   描述: {item['description'][:100]}...")
        
        if report['total_low_confidence'] > 10:
            print(f"\n... 还有 {report['total_low_confidence'] - 10} 条记录")
        
        print("="*60 + "\n")
    
    def reset_low_confidence_stats(self):
        """重置低置信度分类统计"""
        self.low_confidence_classifications = []

