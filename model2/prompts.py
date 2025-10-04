# -*- coding: utf-8 -*-
"""
提示词模板 - Model2
包含意义段分割和迭代验证的提示词
"""

# 判断文本是否包含攻击事件故事（用于智能文本采样）
JUDGE_TEXT_RELEVANCE_PROMPT = """你是一个专业的安全事件分析专家。我将给你一个文本的开头部分和攻击类型，请判断这个文本是否讲述了关于该类型攻击的事件故事。

攻击类型：{attack_type}

文本开头（前100字符）：
{text_preview}

请判断：
1. 这个文本是否描述了一个具体的{attack_type}攻击事件？
2. 文本是否讲述了攻击的过程、细节或相关事件？
3. 文本是否包含事件的时间、地点、人物、经过等要素？

【应该选择的文本（is_relevant = true）】：
- 描述具体攻击事件的新闻报道
- 讲述攻击发生经过的叙述文本
- 包含攻击细节（如何准备、如何实施、造成什么结果）的文本
- 案例分析、事件回顾类文本

【不应该选择的文本（is_relevant = false）】：
- 纯理论探讨、学术分析（没有具体事件）
- 统计数据、图表说明
- 法律条文、政策文件
- 定义、概念解释
- 与攻击无关的背景介绍

请以JSON格式返回结果：
{{
    "is_relevant": true/false,
    "confidence": 0.90,
    "reasoning": "判断理由的简短说明（一句话）"
}}

注意：
- is_relevant 表示该文本是否应该被选入采样库
- confidence 是0到1之间的浮点数，表示判断的信心程度
- 即使文本只有100字符，也要尽可能准确判断
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

# 第三步-1：层次化分类 - 选择顶层事件类型（Type）
CLASSIFY_EVENT_TYPE_STEP1_PROMPT = """你是一个专业的事件分类专家。我将给你一个事件描述，请从给定的事件类型列表中选择最匹配的**顶层事件类型（Type）**。

事件描述：
{event_description}

可选的顶层事件类型及其定义：
{available_types}

【分类标准】：
1. 仔细阅读每个类型的定义，理解其真正含义
2. 关注事件描述的**核心动作**，而不是事件发生的背景或上下文
3. 不要被表面词汇误导，要根据定义判断事件的本质
4. 特别注意：同一个文本中可能包含多种不同类型的事件，不要只关注文本主题

【正确示例】：
- 事件："恐怖分子引爆了炸弹" → 选择 "Conflict"（核心动作是攻击）✓
- 事件："受害者在爆炸中死亡" → 选择 "Life"（核心动作是死亡）✓ 而不是 "Conflict"✗
- 事件："嫌疑人制造了爆炸装置" → 选择 "Manufacture"（核心动作是制造）✓ 而不是 "Conflict"✗
- 事件："调查人员检查了爆炸现场" → 选择 "Cognitive"（核心动作是检查/调查）✓ 而不是 "Conflict"✗
- 事件："嫌疑人被逮捕" → 选择 "Justice"（核心动作是逮捕）✓ 而不是 "Conflict"✗
- 事件："军队撤离了该地区" → 选择 "Movement"（核心动作是移动）✓ 而不是 "Conflict"✗

【错误模式 - 务必避免】：
- ✗ 因为文本是关于恐怖袭击的，就将所有事件都归为"Conflict"
- ✗ 看到"爆炸"、"袭击"等词就选"Conflict"，忽略了事件的实际动作
- ✗ 将准备阶段（制造、计划）归为"Conflict"，应该是"Manufacture"或"Cognitive"
- ✗ 将结果阶段（死亡、受伤）归为"Conflict"，应该是"Life"或"Medical"

请以JSON格式返回结果：
{{
    "selected_type": "选择的顶层类型",
    "confidence": 0.95,
    "reasoning": "选择理由的简短说明（一句话）"
}}

注意：confidence 是0到1之间的浮点数，表示你对这个分类的信心程度。即使confidence较低，也必须给出一个分类。
"""

# 第三步-2：层次化分类 - 选择事件子类型（Subtype）
CLASSIFY_EVENT_TYPE_STEP2_PROMPT = """你是一个专业的事件分类专家。我将给你一个事件描述和已选择的顶层类型，请从给定的子类型列表中选择最匹配的**事件子类型（Subtype）**。

事件描述：
{event_description}

已选择的顶层类型：
{selected_type}

该类型下可选的子类型及其定义：
{available_subtypes}

【分类标准】：
1. 仔细阅读每个子类型的定义
2. 在已确定顶层类型的基础上，进一步细化事件的具体类别
3. 注意区分相似但不同的子类型
4. 选择最具体、最准确描述该事件的子类型

【正确示例】：
- 已选Type: "Conflict", 事件: "引爆炸弹" → 选择 "Attack"（攻击）✓ 而不是 "Demonstrate"（示威）✗
- 已选Type: "Life", 事件: "在爆炸中死亡" → 选择 "Die"（死亡）✓ 而不是 "Injure"（受伤）✗
- 已选Type: "Movement", 事件: "嫌疑人逃离现场" → 选择 "TransportPerson"（人员运输）✓ 而不是 "TransportArtifact"（物品运输）✗
- 已选Type: "Cognitive", 事件: "调查现场" → 选择 "Inspection"（检查）✓ 而不是 "TeachingTrainingLearning"（教学）✗

【错误模式 - 务必避免】：
- ✗ 混淆"Attack"和"Demonstrate"，将和平示威归为攻击
- ✗ 混淆"Die"和"Injure"，将死亡事件标记为受伤
- ✗ 没有仔细阅读定义就凭直觉选择

请以JSON格式返回结果：
{{
    "selected_subtype": "选择的子类型",
    "confidence": 0.90,
    "reasoning": "选择理由的简短说明（一句话）"
}}

注意：confidence 是0到1之间的浮点数。即使confidence较低，也必须给出一个分类。
"""

# 第三步-3：层次化分类 - 选择事件子子类型（Sub_subtype）
CLASSIFY_EVENT_TYPE_STEP3_PROMPT = """你是一个专业的事件分类专家。我将给你一个事件描述、已选择的顶层类型和子类型，请从给定的子子类型列表中选择最匹配的**事件子子类型（Sub_subtype）**。

事件描述：
{event_description}

已选择的类型：
- 顶层类型（Type）: {selected_type}
- 子类型（Subtype）: {selected_subtype}

该组合下可选的子子类型及其定义：
{available_sub_subtypes}

【分类标准】：
1. 仔细阅读每个子子类型的定义
2. 这是最细粒度的分类，要选择最精确描述事件的子子类型
3. 如果事件描述不够具体，可以选择"Unspecified"（未指定）
4. 优先选择具体的子子类型而不是"Unspecified"

【正确示例】：
- Type: "Conflict", Subtype: "Attack", 事件: "引爆炸弹" → 选择 "DetonateExplode"（引爆）✓ 而不是 "Unspecified"✗
- Type: "Life", Subtype: "Die", 事件: "死亡" → 选择 "Unspecified"（因为没有更具体信息）✓
- Type: "Movement", Subtype: "TransportPerson", 事件: "驾车离开" → 选择 "Unspecified"（因为没有更具体的运输方式）✓

【错误模式 - 务必避免】：
- ✗ 当有明确具体信息时选择"Unspecified"
- ✗ 选择与事件描述不符的具体类型
- ✗ 没有仔细阅读定义就凭直觉选择

请以JSON格式返回结果：
{{
    "selected_sub_subtype": "选择的子子类型",
    "confidence": 0.85,
    "reasoning": "选择理由的简短说明（一句话）"
}}

注意：confidence 是0到1之间的浮点数。即使confidence较低，也必须给出一个分类。
"""

# 第三步-4：构建图结构（使用层次化分类的结果）
BUILD_GRAPH_WITH_CLASSIFIED_EVENTS_PROMPT = """你是一个专业的事件图构建专家。我已经为每个事件完成了类型分类，现在需要你构建事件之间的时序关系图。

原始文本：
{original_text}

已分类的事件列表：
{classified_events}

请完成以下任务：
1. 为每个事件分配唯一的节点ID（格式：node_1, node_2, ...）
2. 从原始文本中找到支撑每个事件的文本片段
3. 构建事件之间的时序关系（前后关系）
4. 为每条边找到支撑该时序关系的文本片段

请以JSON格式返回结果：
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
- 每个node的id必须唯一
- edges中的source和target必须对应nodes中的id
- relation统一使用"before"表示时序关系（source在target之前发生）
- support_text应该是原始文本中的实际句子或段落
- 直接使用提供的事件类型，不要修改
"""

# 第四步：判断文本段是否对事件有推进作用
CHECK_SEGMENT_RELEVANCE_PROMPT = """你是一个专业的文本分析专家。请判断下面这个意义段是否对整个文本描述的事件流程有推进作用。

原始完整文本：
{original_text}

当前意义段内容：
{segment_content}

意义段核心语义：
{segment_core_meaning}

请判断这个意义段是否包含以下内容：
1. 描述了重要的事件动作（如攻击、死亡、移动、制造等）
2. 描述了关键的状态变化（如从计划到执行、从准备到完成等）
3. 描述了重要的因果关系或时序关系
4. 对理解整个事件流程有实质性帮助

如果只是以下内容，则判断为无推进作用：
- 纯粹的背景描述（如地点介绍、人物背景等）
- 修饰性内容（如形容词堆砌、情感描述等）
- 与事件流程无关的信息（如评论、分析、引用等）

请以JSON格式返回结果：
{{
    "is_relevant": true/false,
    "reasoning": "判断理由的简短说明"
}}
"""

# 第五步-1：从意义段中提取关键动词及其语义
EXTRACT_VERBS_FROM_SEGMENT_PROMPT = """你是一个专业的语言分析专家。请从给定的意义段中提取所有关键动词，并描述每个动词代表的事件语义。

意义段内容：
{segment_content}

请完成以下任务：
1. 识别意义段中的所有关键动词（重点关注描述动作、状态变化的动词）
2. 对每个动词，提取其语义含义和相关的主体、客体
3. 判断该动词是否描述了一个事件（而非状态描述、连接词等）
4. 注意区分不同时态和嵌套关系（如"报道说发生了袭击"中，"报道"和"袭击"是两个不同的事件）

【示例】：
意义段："The Ministry condemned the terrorist attack in strongest terms."
提取结果：
- 动词1: "condemned"（谴责）
  * 主体: The Ministry
  * 客体: the terrorist attack
  * 语义: 外交部谴责恐怖袭击
  * 是否为事件: true（这是一个谴责行为）
  
- 动词2: "attack"（袭击，虽然是名词形式但代表动作）
  * 主体: terrorist
  * 客体: (未明确)
  * 语义: 恐怖分子发动袭击
  * 是否为事件: true（这是一个攻击事件）
  * 时序关系: 发生在condemned之前

请以JSON格式返回结果：
{{
    "verbs": [
        {{
            "verb": "动词原形",
            "verb_in_text": "文中出现的形式",
            "subject": "主体",
            "object": "客体",
            "semantic_description": "该动词代表的完整事件语义（一句话）",
            "is_event": true/false,
            "temporal_order": 1
        }},
        ...
    ]
}}

注意：
- temporal_order 表示事件发生的时序（1表示最早发生，2表示后发生，以此类推）
- 只提取表示实际事件的动词，忽略助动词、系动词等
- 如果一个动词不表示事件（如"is"、"have"等），设置 is_event 为 false
"""

# 第五步-2：快速预检查动词是否已在图中覆盖
QUICK_CHECK_VERB_COVERAGE_PROMPT = """你是一个专业的事件图验证专家。我从意义段中提取了多个动词事件，需要你快速判断每个动词事件是否已经在当前图中有对应的节点。

提取的动词列表：
{verbs_list}

当前事件图中的节点：
{graph_nodes}

对于每个动词，请判断：
1. 该动词的语义是否与图中某个节点的描述相似或相同？
2. 注意：即使用词不同，只要语义相同就算覆盖（如"爆炸"和"引爆"、"死亡"和"丧生"）

请以JSON格式返回结果：
{{
    "coverage_check": [
        {{
            "verb": "动词原形",
            "semantic_description": "事件语义",
            "is_covered": true/false,
            "covered_by_node": "node_id或null",
            "reason": "判断理由"
        }},
        ...
    ]
}}

【判断原则】：
- 宽松判断：语义相似即可认为已覆盖
- 例如：动词"condemned"（谴责）与节点描述"外交部谴责袭击"相匹配
- 如果不确定，倾向于判断为未覆盖（需要进一步分类确认）
"""

# 第五步-3：检查每个动词事件是否在图中有体现
CHECK_VERB_COVERAGE_PROMPT = """你是一个专业的事件图验证专家。我已经识别出意义段中的多个动词事件，现在需要你逐个检查它们是否在当前图中有体现。

已分类的动词事件列表：
{classified_verbs}

当前事件图结构：
{current_graph}

对于每个动词事件，请检查：
1. 该事件是否已经在图中有对应的节点？
2. 如果有，是哪个节点？
3. 如果没有，是否需要添加新节点？
4. 动词事件之间的时序关系是否正确体现？

请以JSON格式返回结果：
{{
    "verb_coverage": [
        {{
            "verb": "动词原形",
            "semantic_description": "事件语义",
            "is_covered": true/false,
            "covered_by_node": "node_id或null",
            "needs_new_node": true/false
        }},
        ...
    ],
    "suggestions": [
        {{
            "action": "add_node",
            "details": {{
                "id": "new_node_X",
                "event_type": "...",
                "event_subtype": "...",
                "event_sub_subtype": "...",
                "description": "基于动词X的事件描述",
                "support_text": "原始意义段内容"
            }}
        }},
        {{
            "action": "add_edge",
            "details": {{
                "source": "node_A",
                "target": "node_B",
                "relation": "before",
                "support_text": "..."
            }}
        }}
    ],
    "explanation": "检查结果的总体说明"
}}

【重要原则】：
- 同一个意义段中的多个动词事件，应该都有对应的节点
- 根据temporal_order添加时序边
- 避免重复添加已存在的节点
- 优先复用图中已有的相似节点
"""


