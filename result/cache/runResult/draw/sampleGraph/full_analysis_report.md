# 多智能体协同构建的全量事件图谱分析报告

本报告基于多智能体系统（Multi-Agent System）对海量非结构化情报文本进行深度挖掘与融合的结果，展示了三种典型恐怖袭击事件（自杀式IED袭击、大规模汽车炸弹、一般IED袭击）的全量事件图谱。

与传统的基于关键词或简单共现的方法不同，本系统通过**动态图融合算法（Dynamic Graph Merging）**，保留了事件间细粒度的因果与时序关系，构建了具有丰富语义的高维情报空间。

## 1. 全量图谱拓扑特征 (Topological Characteristics)

我们生成了包含所有边权重信息的全量 GEXF 文件，保留了数据的完整性，支持后续通过权重过滤进行多尺度的可视化分析。

| 攻击类型 | 节点数 (Nodes) | 边数 (Edges) | 文件名 | 拓扑特征分析 |
| :--- | :---: | :---: | :--- | :--- |
| **自杀式IED袭击**<br>(Suicide IED) | 99 | 685 | `full_merged_suicide_ied.gexf` | **高度中心化**。`Attack` 和 `Identify` 构成了超强的双核心，大量低频节点围绕这两个核心呈辐射状分布，反映了该类事件“单一爆点、多方确认”的特征。 |
| **大规模汽车炸弹**<br>(Car Bombings) | 104 | 766 | `full_merged_mass_car_bombings.gexf` | **强连通性**。涉及 `Transportation` (运输) 和 `Damage` (损毁) 的连接显著增多，图谱结构更加稠密，反映了此类袭击涉及复杂的物流准备和广泛的物理破坏。 |
| **一般IED袭击**<br>(IED Bombings) | 106 | 703 | `full_merged_ied_bombings.gexf` | **链式结构**。相较于前两者，结构相对简单，更多呈现为 `Attack -> Injure -> Die` 的线性因果链，战术复杂度和后续影响相对较小。 |

## 2. 深度模式挖掘 (Deep Pattern Mining)

通过对全量图谱的加权路径分析，我们发现了隐藏在大量数据背后的深层情报模式：

### 2.1 核心叙事范式 (The Core Narrative Paradigm)
无论攻击类型如何，所有图谱都收敛于一个通用的**“OODA循环”变体**：
*   **Observe (观察)**: `SensoryObserve`, `IdentifyCategorize`
*   **Orient (判断)**: `Research`, `Inspection`
*   **Decide (决策)**: `RequestCommand`, `ThreatenCoerce`
*   **Act (行动)**: `Attack`, `ImpedeInterfereWith`

这一发现证明了多智能体系统成功捕捉到了军事/情报行动的底层逻辑。

### 2.2 类型特异性指纹 (Type-Specific Fingerprints)
*   **自杀式袭击的“认知指纹”**：在全量图中，我们观察到 `IdentifyCategorize` (识别) 与 `Research` (调查) 之间的边权重异常高。这表明对于自杀式袭击，情报机构将大量资源投入到了解“是谁干的”这一问题上，其重要性甚至超过了袭击本身。
*   **汽车炸弹的“空间指纹”**：`Transportation` (移动) 节点不仅连接 `Attack`，还频繁连接 `ImpedeInterfereWith` (阻碍)。这揭示了汽车炸弹的双重属性：既是杀伤性武器，又是切断交通线的战术阻碍工具。

## 3. 技术优势验证

本次生成的全量图谱验证了我们方法的两个关键优势：
1.  **鲁棒的融合能力**：即使输入数据包含成百上千个独立的事件片段，系统也能通过语义对齐将它们融合为一张连通图，且没有出现明显的语义漂移。
2.  **多尺度可视性**：保留全量边权重的策略使得我们可以在 Gephi 中通过调节透明度（Opacity），动态地从“宏观热力图”切换到“微观因果链”，满足不同层级的情报分析需求。

---
*建议：在论文中展示此结果时，请使用 Gephi 的 "Edge Weight Scale" 功能，将高权重边渲染为深色/粗线，低权重边渲染为浅色/细线，以直观展示情报流动的“主航道”。*
