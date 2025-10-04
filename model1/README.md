# 简单询问式构图算法 (Model 1)

## 概述

本模块实现了基于大语言模型的简单询问式事件骨架图构建算法。算法通过分析攻击事件的文本描述，提取事件发展流程，并将其标准化为预定义的事件类型，最终融合多个文本的事件图为一个骨架图。

## 算法流程

1. **选择攻击类型**: 比如 suicide_ied
2. **加载文本数据**: 从 `dataset/processedData/extracted_texts` 文件夹下找到该类型的所有文本文件
3. **文本采样**: 如果文本文件数量大于15，随机抽样选取15个文本
4. **事件流程提取**: 对每个文本，使用大语言模型归纳出文本中事件的发展流程
5. **事件标准化**: 结合 `dataset/kairos_ontology.xlsx` 中的事件类型，将归纳出的发展流程标准化为表中规定的事件类型
6. **图构建**: 使用 networkx 库生成图，将事件类型作为节点，事件关系作为边
7. **图融合**: 采用基于出现频率的图融合算法，将多个图融合为最终的事件骨架图

## 文件结构

```
model1/
├── config.py           # 配置文件（API密钥、路径等）
├── data_loader.py      # 数据加载模块
├── prompts.py          # LLM提示词模板
├── graph_builder.py    # 图构建模块
├── graph_merger.py     # 图融合模块
├── main.py            # 主程序
└── README.md          # 本文件
```

## 使用方法

### 1. 配置API密钥

编辑 `config.py` 文件，填入你的智谱AI API密钥：

```python
ZHIPU_API_KEY = "your_api_key_here"
```

或者在运行时通过命令行参数传入。

### 2. 运行算法

```bash
# 使用默认攻击类型 (suicide_ied)
python model1/main.py

# 指定攻击类型
python model1/main.py --attack_type backpack_ied

# 通过命令行传入API密钥
python model1/main.py --api_key YOUR_API_KEY --attack_type suicide_ied
```

### 3. 查看结果

结果将保存在 `result/graphs/` 目录下：

- `graph_1.json`, `graph_2.json`, ... : 每个文本对应的事件图
- `merged_graph_{attack_type}.json` : 融合后的骨架图

## 输出格式

### 单个文本的事件图

```json
{
  "nodes": [
    {
      "id": "text_1_node_1",
      "event_type": "Conflict",
      "event_subtype": "Attack",
      "event_sub_subtype": "DetonateExplode",
      "description": "原始事件描述",
      "source_text": "text_1"
    }
  ],
  "edges": [
    {
      "source": "text_1_node_1",
      "target": "text_1_node_2",
      "relation": "before"
    }
  ]
}
```

### 融合后的骨架图

```json
{
  "nodes": [
    {
      "id": "Conflict > Attack > DetonateExplode",
      "event_type": "Conflict",
      "event_subtype": "Attack",
      "event_sub_subtype": "DetonateExplode",
      "occurrence_count": 12,
      "occurrence_percentage": 80.0
    }
  ],
  "edges": [
    {
      "source": "Cognitive > IdentifyCategorize > Unspecified",
      "target": "Conflict > Attack > DetonateExplode",
      "weight": 10,
      "occurrence_percentage": 66.7
    }
  ],
  "statistics": {
    "total_nodes": 15,
    "total_edges": 25
  }
}
```

## 依赖项

- zhipuai
- pandas
- networkx
- openpyxl (用于读取Excel文件)

## 注意事项

1. 确保 `dataset/kairos_ontology.xlsx` 文件存在且格式正确
2. 确保 `dataset/processedData/extracted_texts/` 目录包含文本数据
3. API调用需要网络连接和有效的API密钥
4. 大量文本处理可能需要较长时间和一定的API调用费用

## 日志

程序运行过程中会在 `logs/` 目录下生成带时间戳的日志文件，记录详细的执行过程。

