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
        ONTOLOGY = "ABS - Abstract, non-tangible artifacts（抽象非实体物品）、AML - Animal（动物）、BAL - Ballot for an election（选举选票）、BOD - Identifiable body part（身体部位）、COM - Tangible product or article of trade（有形商品）、FAC - Functional man-made structure（功能性人造结构）、GPE - Geopolitical entities（地缘政治实体）、INF - Information object（信息对象）、LAW - Law（法律）、LOC - Geographical entities（地理实体）、MHI - Medical condition or health issue（医疗健康问题）、MON - Monetary payment（货币支付）、NAT - Natural resources（自然资源）、ORG - Organizations（组织）、PER - Persons（人）、PLA - Plants/flora（植物）、PTH - Pathogens（病原体）、RES - Results of a voting event（投票结果）、SEN - Judicial sentence（司法判决）、SID - Sides of a conflict（冲突方）、TTL - Person's title or job role（头衔/职位）、VAL - Numerical or non-numerical value（数值/非数值）、VEH - Vehicles（载具）、WEA - Weapons（武器）"
        return {
            # 0. 实体分类

            "ontology": ONTOLOGY,

            # 1. 实体识别（动态类型）
            "ner": """请从以下文本中抽取部分实体，实体名称和文本中的一致，
            只需要找以下提到的24种类型的实体即可，不要抽取无关的实体，没有对应的实体输出空列表即可：""" + ONTOLOGY + """。
            输出格式：仅返回JSON数组，每个元素包含"实体文本"和"类型"，例如：
            [{{"实体文本": "张三", "类型": "PER"}}, {{"实体文本": "炸药", "类型": "WEA"}}]
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
