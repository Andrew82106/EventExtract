# mergeGraph - 图融合模块

## 概述

这是一个共享的图融合模块，为所有模型提供统一的图融合算法。

## 功能

将多个事件图融合为一个骨架图，使用基于事件类型的节点合并策略。

## 算法特点

1. **频率过滤**：只保留在至少10%的图中出现的事件类型和关系
2. **事件类型合并**：基于事件类型的完整层级结构（Type > Subtype > Sub-subtype）进行合并
3. **统计信息**：记录每个事件类型和关系的出现次数和百分比

## 使用方法

```python
from mergeGraph import GraphMerger

# 创建融合器实例
merger = GraphMerger()

# 融合多个图
merged_graph = merger.merge_graphs(graph_list)

# 保存融合后的图
merger.save_merged_graph(merged_graph, output_path)
```

## 输出格式

融合后的图包含以下信息：

### 节点属性
- `event_type`: 事件类型
- `event_subtype`: 事件子类型
- `event_sub_subtype`: 事件子子类型
- `occurrence_count`: 出现次数
- `occurrence_percentage`: 出现百分比

### 边属性
- `weight`: 出现次数
- `occurrence_percentage`: 出现百分比

## 融合策略

1. 统计所有图中的事件类型及其出现次数
2. 统计所有图中的边（时序关系）及其出现次数
3. 只保留出现频率 ≥ 10% 的事件类型
4. 只保留出现频率 ≥ 10% 的边
5. 生成融合后的骨架图

## 依赖

- networkx: 图数据结构

## 版本

v1.0.0

