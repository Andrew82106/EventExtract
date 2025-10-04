# -*- coding: utf-8 -*-
"""
事件类型验证器
负责验证事件类型是否符合本体定义
"""
from difflib import SequenceMatcher


class EventTypeValidator:
    """事件类型验证器类（增强版 - 严格模式）"""
    
    def __init__(self, event_types, strict_mode=True, similarity_threshold=0.7):
        """
        初始化验证器
        
        Args:
            event_types (list): 从本体加载的事件类型列表
                每个元素是字典: {'type': str, 'subtype': str, 'sub_subtype': str, ...}
            strict_mode (bool): 严格模式，拒绝无法修复的节点（默认True）
            similarity_threshold (float): 相似度匹配阈值（默认0.7）
        """
        self.event_types = event_types
        self.strict_mode = strict_mode
        self.similarity_threshold = similarity_threshold
        self._build_indices()
        self.validation_stats = {
            'total_validated': 0,
            'valid_nodes': 0,
            'invalid_nodes': 0,
            'auto_fixed_nodes': 0,
            'rejected_nodes': 0,  # 新增：被拒绝的节点数
            'issues': []
        }
    
    def _build_indices(self):
        """构建快速查询索引"""
        # 构建完整的三层类型组合索引
        self.valid_combinations = set()
        # 构建各层级的有效值索引
        self.valid_types = set()
        self.valid_subtypes_by_type = {}  # {type: set(subtypes)}
        self.valid_sub_subtypes_by_subtype = {}  # {(type, subtype): set(sub_subtypes)}
        
        for event in self.event_types:
            event_type = event['type']
            subtype = event['subtype'] if event['subtype'] else ''
            sub_subtype = event['sub_subtype'] if event['sub_subtype'] else ''
            
            # 添加到类型集合
            self.valid_types.add(event_type)
            
            # 添加完整组合
            combination = (event_type, subtype, sub_subtype)
            self.valid_combinations.add(combination)
            
            # 构建层级索引
            if event_type not in self.valid_subtypes_by_type:
                self.valid_subtypes_by_type[event_type] = set()
            if subtype:
                self.valid_subtypes_by_type[event_type].add(subtype)
            
            type_subtype_key = (event_type, subtype)
            if type_subtype_key not in self.valid_sub_subtypes_by_subtype:
                self.valid_sub_subtypes_by_subtype[type_subtype_key] = set()
            if sub_subtype:
                self.valid_sub_subtypes_by_subtype[type_subtype_key].add(sub_subtype)
    
    def validate_node(self, node, auto_fix=True):
        """
        验证单个节点的事件类型
        
        Args:
            node (dict): 节点数据，包含event_type, event_subtype, event_sub_subtype
            auto_fix (bool): 是否自动修复无效类型
            
        Returns:
            tuple: (is_valid, fixed_node, issues)
                - is_valid: 是否有效
                - fixed_node: 修复后的节点（如果auto_fix=True）
                - issues: 问题列表
        """
        self.validation_stats['total_validated'] += 1
        
        event_type = node.get('event_type', '').strip()
        event_subtype = node.get('event_subtype', '').strip()
        event_sub_subtype = node.get('event_sub_subtype', '').strip()
        
        issues = []
        fixed_node = node.copy()
        
        # 检查三层组合是否存在于本体中
        combination = (event_type, event_subtype, event_sub_subtype)
        
        if combination in self.valid_combinations:
            self.validation_stats['valid_nodes'] += 1
            return True, fixed_node, []
        
        # 组合无效，记录问题并尝试修复
        self.validation_stats['invalid_nodes'] += 1
        
        # 检查各层级的问题
        if event_type not in self.valid_types:
            issues.append(f"无效的事件类型: '{event_type}'")
            if auto_fix:
                # 查找最相似的类型
                best_match = self._find_best_match(event_type, self.valid_types)
                if best_match:
                    issues.append(f"  -> 自动修复为: '{best_match}'")
                    fixed_node['event_type'] = best_match
                    event_type = best_match
                    self.validation_stats['auto_fixed_nodes'] += 1
        
        if event_subtype and event_type in self.valid_subtypes_by_type:
            valid_subtypes = self.valid_subtypes_by_type[event_type]
            if event_subtype not in valid_subtypes:
                issues.append(f"无效的子类型: '{event_subtype}' (在类型 '{event_type}' 下)")
                if auto_fix:
                    best_match = self._find_best_match(event_subtype, valid_subtypes)
                    if best_match:
                        issues.append(f"  -> 自动修复为: '{best_match}'")
                        fixed_node['event_subtype'] = best_match
                        event_subtype = best_match
                        self.validation_stats['auto_fixed_nodes'] += 1
                    else:
                        # 如果找不到匹配，使用空字符串
                        issues.append(f"  -> 未找到匹配的子类型，清空")
                        fixed_node['event_subtype'] = ''
                        event_subtype = ''
        
        if event_sub_subtype and (event_type, event_subtype) in self.valid_sub_subtypes_by_subtype:
            valid_sub_subtypes = self.valid_sub_subtypes_by_subtype[(event_type, event_subtype)]
            if event_sub_subtype not in valid_sub_subtypes:
                issues.append(f"无效的子子类型: '{event_sub_subtype}' (在类型 '{event_type} > {event_subtype}' 下)")
                if auto_fix:
                    best_match = self._find_best_match(event_sub_subtype, valid_sub_subtypes)
                    if best_match:
                        issues.append(f"  -> 自动修复为: '{best_match}'")
                        fixed_node['event_sub_subtype'] = best_match
                        self.validation_stats['auto_fixed_nodes'] += 1
                    else:
                        issues.append(f"  -> 未找到匹配的子子类型，清空")
                        fixed_node['event_sub_subtype'] = ''
        
        # 记录问题
        if issues:
            self.validation_stats['issues'].append({
                'node_id': node.get('id', 'unknown'),
                'original': combination,
                'fixed': (fixed_node.get('event_type'), 
                         fixed_node.get('event_subtype'), 
                         fixed_node.get('event_sub_subtype')),
                'issues': issues
            })
        
        # 再次验证修复后的组合
        fixed_combination = (
            fixed_node.get('event_type', ''),
            fixed_node.get('event_subtype', ''),
            fixed_node.get('event_sub_subtype', '')
        )
        is_valid = fixed_combination in self.valid_combinations
        
        return is_valid, fixed_node, issues
    
    def validate_graph_data(self, graph_data, auto_fix=True):
        """
        验证整个图数据
        
        Args:
            graph_data (dict): 图数据，包含nodes和edges
            auto_fix (bool): 是否自动修复
            
        Returns:
            tuple: (validated_graph_data, validation_report)
        """
        if not graph_data or 'nodes' not in graph_data:
            return graph_data, {"error": "无效的图数据"}
        
        validated_graph = {
            'nodes': [],
            'edges': graph_data.get('edges', [])
        }
        
        validation_report = {
            'total_nodes': len(graph_data['nodes']),
            'valid_nodes': 0,
            'invalid_nodes': 0,
            'fixed_nodes': 0,
            'rejected_nodes': 0,
            'node_issues': []
        }
        
        # 用于跟踪被拒绝的节点ID
        rejected_node_ids = set()
        
        for node in graph_data['nodes']:
            is_valid, fixed_node, issues = self.validate_node(node, auto_fix)
            
            if is_valid:
                validation_report['valid_nodes'] += 1
                validated_graph['nodes'].append(fixed_node)
            else:
                # 节点无效
                validation_report['invalid_nodes'] += 1
                validation_report['node_issues'].append({
                    'node_id': node.get('id'),
                    'issues': issues
                })
                
                # 严格模式：拒绝无法修复或修复后仍然无效的节点
                if self.strict_mode:
                    validation_report['rejected_nodes'] += 1
                    rejected_node_ids.add(node.get('id'))
                    self.validation_stats['rejected_nodes'] += 1
                    # 不添加到validated_graph中
                elif auto_fix:
                    # 非严格模式 + auto_fix：保留修复后的节点
                    validation_report['fixed_nodes'] += 1
                    validated_graph['nodes'].append(fixed_node)
                else:
                    # 非严格模式 + 不auto_fix：保留但标记
                    fixed_node['_validation_failed'] = True
                    validated_graph['nodes'].append(fixed_node)
        
        # 如果有节点被拒绝，需要清理相关的边
        if rejected_node_ids:
            cleaned_edges = []
            for edge in validated_graph['edges']:
                source_id = edge.get('source', '').split('_')[-1] if '_' in edge.get('source', '') else edge.get('source', '')
                target_id = edge.get('target', '').split('_')[-1] if '_' in edge.get('target', '') else edge.get('target', '')
                
                # 只保留两端节点都未被拒绝的边
                if source_id not in rejected_node_ids and target_id not in rejected_node_ids:
                    cleaned_edges.append(edge)
            
            validated_graph['edges'] = cleaned_edges
        
        return validated_graph, validation_report
    
    def _find_best_match(self, target, candidates):
        """
        查找最相似的候选项
        
        Args:
            target (str): 目标字符串
            candidates (set/list): 候选项集合
            
        Returns:
            str: 最佳匹配，如果没有足够好的匹配返回None
        """
        if not target or not candidates:
            return None
        
        best_match = None
        best_ratio = 0.0
        
        for candidate in candidates:
            if not candidate:
                continue
            ratio = SequenceMatcher(None, target.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        
        # 使用配置的相似度阈值
        return best_match if best_ratio >= self.similarity_threshold else None
    
    def get_validation_summary(self):
        """
        获取验证统计摘要
        
        Returns:
            dict: 验证统计信息
        """
        return {
            'total_validated': self.validation_stats['total_validated'],
            'valid_nodes': self.validation_stats['valid_nodes'],
            'invalid_nodes': self.validation_stats['invalid_nodes'],
            'auto_fixed_nodes': self.validation_stats['auto_fixed_nodes'],
            'rejected_nodes': self.validation_stats['rejected_nodes'],
            'validation_rate': (self.validation_stats['valid_nodes'] / 
                              self.validation_stats['total_validated'] * 100) 
                              if self.validation_stats['total_validated'] > 0 else 0,
            'total_issues': len(self.validation_stats['issues']),
            'strict_mode': self.strict_mode,
            'similarity_threshold': self.similarity_threshold
        }
    
    def get_detailed_issues(self):
        """
        获取详细的问题列表
        
        Returns:
            list: 详细问题列表
        """
        return self.validation_stats['issues']
    
    def reset_stats(self):
        """重置验证统计"""
        self.validation_stats = {
            'total_validated': 0,
            'valid_nodes': 0,
            'invalid_nodes': 0,
            'auto_fixed_nodes': 0,
            'rejected_nodes': 0,
            'issues': []
        }
    
    def print_validation_report(self):
        """打印验证报告"""
        summary = self.get_validation_summary()
        
        print("\n" + "="*60)
        print("事件类型验证报告")
        print("="*60)
        print(f"验证模式: {'严格模式' if summary['strict_mode'] else '宽松模式'}")
        print(f"相似度阈值: {summary['similarity_threshold']}")
        print(f"总验证节点数: {summary['total_validated']}")
        print(f"有效节点数: {summary['valid_nodes']}")
        print(f"无效节点数: {summary['invalid_nodes']}")
        print(f"自动修复节点数: {summary['auto_fixed_nodes']}")
        print(f"拒绝节点数: {summary['rejected_nodes']}")
        print(f"验证通过率: {summary['validation_rate']:.2f}%")
        print(f"问题总数: {summary['total_issues']}")
        
        if self.validation_stats['issues']:
            print("\n详细问题列表:")
            print("-"*60)
            for i, issue_info in enumerate(self.validation_stats['issues'][:10], 1):
                print(f"\n节点 {issue_info['node_id']}:")
                print(f"  原始类型: {issue_info['original']}")
                print(f"  修复后: {issue_info['fixed']}")
                for issue in issue_info['issues']:
                    print(f"  {issue}")
            
            if len(self.validation_stats['issues']) > 10:
                print(f"\n... 还有 {len(self.validation_stats['issues']) - 10} 个问题")
        
        print("="*60 + "\n")

