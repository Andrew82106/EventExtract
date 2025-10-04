# -*- coding: utf-8 -*-
"""
提示词模板
"""

# 第一步：归纳事件发展流程
EXTRACT_EVENT_FLOW_PROMPT = """你是一个专业的事件分析专家。请仔细阅读以下文本，归纳出文本中描述的事件发展流程。

文本内容：
{text}

请按照时间顺序，归纳出文本中的关键事件及其发展流程。对于每个事件，请用简洁的语言描述事件的核心内容。

要求：
1. 按照时间先后顺序列出事件
2. 每个事件用一句话简洁描述
3. 只包含文本中明确提到的事件
4. 事件之间要有逻辑连贯性

请以JSON格式返回结果，格式如下：
{{
    "events": [
        {{"event_id": 1, "description": "事件描述1"}},
        {{"event_id": 2, "description": "事件描述2"}},
        ...
    ]
}}
"""

# 第二步：标准化事件类型并构建图结构
STANDARDIZE_EVENT_TYPES_PROMPT = """你是一个专业的事件分类专家。我已经从一段文本中提取出了事件发展流程，现在需要你将这些事件标准化为预定义的事件类型，并构建事件之间的时序关系图。

提取的事件流程：
{event_flow}

预定义的事件类型本体：
{event_ontology}

请完成以下任务：
1. 将每个提取的事件映射到最合适的预定义事件类型（使用Type > Subtype > Sub-subtype的层级结构）
2. 如果某个事件无法很好地映射到任何预定义类型，选择最接近的类型
3. 构建事件之间的时序关系（前后关系）

请以JSON格式返回结果，格式如下：
{{
    "nodes": [
        {{
            "id": "node_1",
            "event_type": "事件类型",
            "event_subtype": "事件子类型",
            "event_sub_subtype": "事件子子类型",
            "description": "原始事件描述"
        }},
        ...
    ],
    "edges": [
        {{
            "source": "node_1",
            "target": "node_2",
            "relation": "before"
        }},
        ...
    ]
}}

注意：
- 每个node的id必须唯一
- edges中的source和target必须对应nodes中的id
- relation统一使用"before"表示时序关系（source在target之前发生）
"""

