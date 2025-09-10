# -*- coding: utf-8 -*-
"""
Prompt模板配置文件
包含所有用于事件图构建的提示词模板
"""


class PromptTemplates:
    """提示词模板类，包含所有阶段的提示词"""
    
    @staticmethod
    def get_templates():
        """
        获取所有提示词模板
        
        Returns:
            dict: 包含各阶段提示词模板的字典
        """
        return {
            # 1. 实体识别（动态类型）
            "ner": """请从以下文本中识别所有有意义的实体，并为每个实体标注合理的类型（类型名称自定，需体现实体本质，如"人名""地点""武器""组织"等）。
输出格式：仅返回JSON数组，每个元素包含"实体文本"和"类型"，例如：
[{{"实体文本": "张三", "类型": "人名"}}, {{"实体文本": "炸药", "类型": "武器"}}]
文本：{text}
""",
            
            # 2. 事件识别与论元抽取（动态论元角色）
            "event_extraction": """请从以下文本中识别事件，每个事件包含：
- 触发词：表示事件的动词（如"购买""爆炸"）
- 事件类型：基于触发词的语义（如"购买行为""袭击事件"）
- 论元：事件的参与者，需标注角色（角色名称必须来自于给出的实体列表）
输出格式：仅返回JSON数组，例如：
[{{"触发词": "购买", "事件类型": "购买行为", "论元": {{"购买者": "张三", "被购买物品": "炸药"}}}}]
文本：{text}
实体列表：{entity_list}
""",
            
            # 3. 共指消解（合并同一实体的不同表述）
            "coreference": """请找出以下实体列表中所有共指实体（指向同一对象的不同表述），输出格式：
仅返回JSON数组，每个元素为{{"统一实体": "xxx", "指代表述": ["a", "b", "c"]}}，例如：
[{{"统一实体": "张三", "指代表述": ["张三", "他", "该恐怖分子"]}}]
实体列表：{entity_list}
""",
            
            # 4. 实体关系抽取（动态关系类型）
            "entity_relation": """请从以下实体列表中提取实体间的有意义关系（关系名称自定，如"拥有""位于""隶属于"），输出格式：
仅返回JSON数组，每个元素为{{"实体1": "a", "关系类型": "r", "实体2": "b"}}，例如：
[{{"实体1": "二手卡车", "关系类型": "装载", "实体2": "炸药"}}]
实体列表：{entity_list}
""",
            
            # 5. 事件时间关系推理
            "temporal_relation": """请结合文本内容，分析所给相关事件的时间顺序，判断事件间的"先于""后于""同时"关系，输出格式：
仅返回JSON数组，每个元素为{{"事件类型1": "事件类型名称1", "时间关系": "r", "事件类型2": "事件类型名称2"}}，例如：
[{{"事件类型1": "购买行为", "时间关系": "先于", "事件2": "组装行为"}}]
文本：{text}
已有事件列表：{event_list}
"""
        }
    
    @staticmethod
    def get_ner_template():
        """获取实体识别模板"""
        return PromptTemplates.get_templates()["ner"]
    
    @staticmethod
    def get_event_extraction_template():
        """获取事件识别模板"""
        return PromptTemplates.get_templates()["event_extraction"]
    
    @staticmethod
    def get_coreference_template():
        """获取共指消解模板"""
        return PromptTemplates.get_templates()["coreference"]
    
    @staticmethod
    def get_entity_relation_template():
        """获取实体关系抽取模板"""
        return PromptTemplates.get_templates()["entity_relation"]
    
    @staticmethod
    def get_temporal_relation_template():
        """获取时间关系推理模板"""
        return PromptTemplates.get_templates()["temporal_relation"]
