# -*- coding: utf-8 -*-
"""
提示词模板 - Model2
包含意义段分割和迭代验证的提示词
"""

# 第一步：将文本拆分为意义段
SEGMENT_TEXT_PROMPT = """你是一个专业的文本分析专家。请仔细阅读以下文本，按照句子的语义将文本拆分成多个意义段。

文本内容：
{text}

要求：
1. 每个意义段应该包含一个或多个相关的句子，表达一个完整的语义单元
2. 意义段之间应该有清晰的语义边界
3. 每个意义段应该简洁明了，易于理解
4. 保持原文内容，不要修改或省略

请以JSON格式返回结果，格式如下：
{{
    "segments": [
        {{
            "segment_id": 1,
            "content": "第一个意义段的完整内容",
            "core_meaning": "这个意义段的核心语义概括"
        }},
        {{
            "segment_id": 2,
            "content": "第二个意义段的完整内容",
            "core_meaning": "这个意义段的核心语义概括"
        }},
        ...
    ]
}}
"""

# 第二步：归纳事件发展流程
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

# 第三步：标准化事件类型并构建图结构（带支撑文本）
STANDARDIZE_EVENT_TYPES_WITH_SUPPORT_PROMPT = """你是一个专业的事件分类专家。我已经从一段文本中提取出了事件发展流程，现在需要你将这些事件标准化为预定义的事件类型，并构建事件之间的时序关系图。

原始文本：
{original_text}

提取的事件流程：
{event_flow}

预定义的事件类型本体：
{event_ontology}

请完成以下任务：
1. 将每个提取的事件映射到最合适的预定义事件类型（使用Type.Subtype.Sub_subtype的三层结构）
2. 如果某个事件无法很好地映射到任何预定义类型，选择最接近的类型
3. 构建事件之间的时序关系（前后关系）
4. 对于每个节点和边，从原始文本中找到支撑文本（即文本中哪些句子支持这个事件或关系）

请以JSON格式返回结果，格式如下：
{{
    "nodes": [
        {{
            "id": "node_1",
            "event_type": "事件类型",
            "event_subtype": "事件子类型",
            "event_sub_subtype": "事件子子类型",
            "description": "原始事件描述",
            "support_text": "从原始文本中摘取的支撑这个事件的文本片段"
        }},
        ...
    ],
    "edges": [
        {{
            "source": "node_1",
            "target": "node_2",
            "relation": "before",
            "support_text": "支撑这个时序关系的文本片段"
        }},
        ...
    ]
}}

【重要约束】：
- 必须严格使用本体中的事件类型，不能创造新类型
- 事件类型必须是完整的三层结构：Type.Subtype.Sub_subtype
- 每个node的id必须唯一
- edges中的source和target必须对应nodes中的id
- relation统一使用"before"表示时序关系（source在target之前发生）
- support_text应该是原始文本中的实际句子或段落
- 不要使用文本中的普通词汇（如人名、地名等）作为事件类型
- 示例正确格式：Conflict.Attack.DetonateExplode, Life.Die.Unspecified
- 示例错误格式：Conflict.Barred, Attack.Attack.Unspecified, Cognitive.IdentifyCategorize
"""

# 第四步：检查意义段是否在图中有体现
CHECK_SEGMENT_COVERAGE_PROMPT = """你是一个专业的事件图验证专家。我已经从文本中构建了一个事件图，现在需要你检查某个意义段的核心语义是否在图中得到了正确的体现。

意义段内容：
{segment_content}

意义段核心语义：
{segment_core_meaning}

当前事件图结构：
{current_graph}

请分析：
1. 这个意义段的核心语义是否已经在图中有所体现？
2. 如果有体现，是通过哪些节点或边来体现的？
3. 如果没有体现或体现不充分，应该如何修改图结构？

请以JSON格式返回结果，格式如下：
{{
    "is_covered": true/false,
    "covered_by": ["node_1", "node_2", "edge_1_2"],  // 如果is_covered为true，列出相关的节点和边
    "suggestions": [
        {{
            "action": "add_node",  // 可能的值：add_node, add_edge, modify_node, modify_edge
            "details": {{
                "id": "new_node_1",
                "event_type": "...",
                "event_subtype": "...",
                "event_sub_subtype": "...",
                "description": "...",
                "support_text": "..."
            }}
        }},
        {{
            "action": "add_edge",
            "details": {{
                "source": "node_1",
                "target": "new_node_1",
                "relation": "before",
                "support_text": "..."
            }}
        }}
    ],
    "explanation": "对检查结果的解释说明"
}}

注意：
- 如果is_covered为true，suggestions可以为空列表
- 如果is_covered为false，必须提供具体的修改建议
- 修改建议要符合事件类型本体的定义
"""

# 第五步：应用修改建议更新图结构
APPLY_MODIFICATIONS_PROMPT = """你是一个专业的图结构编辑专家。请根据提供的修改建议，更新事件图结构。

当前图结构：
{current_graph}

修改建议：
{modifications}

预定义的事件类型本体：
{event_ontology}

请应用这些修改建议，并返回更新后的完整图结构。

要求：
1. 确保所有新增的节点ID唯一
2. 确保所有边的source和target都指向存在的节点
3. 保持图的一致性和连贯性
4. 新增的事件类型必须符合本体定义

【严格约束】：
- 必须严格使用本体中的事件类型，绝对不能创造新类型
- 事件类型必须是完整的三层结构：Type.Subtype.Sub_subtype（用点号分隔）
- 如果修改建议中的事件类型不在本体中，必须选择最接近的标准类型替代
- 不要使用文本中的普通词汇（如人名、地名、形容词等）作为事件类型
- 优先复用已有的事件类型，减少新节点的创建
- 每个文本最多新增3-5个节点，避免过度生成
- 示例正确格式：Conflict.Attack.DetonateExplode, Life.Die.Unspecified
- 示例错误格式：Conflict.Barred, Attack.Attack.Unspecified, Cognitive.IdentifyCategorize

请以JSON格式返回完整的更新后的图结构，格式如下：
{{
    "nodes": [
        {{
            "id": "node_1",
            "event_type": "...",
            "event_subtype": "...",
            "event_sub_subtype": "...",
            "description": "...",
            "support_text": "..."
        }},
        ...
    ],
    "edges": [
        {{
            "source": "node_1",
            "target": "node_2",
            "relation": "before",
            "support_text": "..."
        }},
        ...
    ]
}}
"""

