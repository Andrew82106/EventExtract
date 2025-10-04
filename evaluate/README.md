# 事件骨架图评测系统

## 概述

本评测系统用于评测事件骨架图的生成质量，按照`algorithm.md`中定义的评测方法进行评估。

## 评测指标

### 指标1：主题重合F1值

计算算法生成的事件骨架图中包含的事件集合与ground truth中对应的事件骨架图中包含的事件集合之间的F1值。

- **Precision**: 生成的事件类型中有多少在ground truth中
- **Recall**: ground truth中的事件类型有多少被生成
- **F1**: 精确率和召回率的调和平均数

### 指标2：事件序列匹配F1值

计算生成的事件骨架中长度为2或3的事件序列与ground truth中对应序列之间的F1值。

- 评测长度为2的事件序列
- 评测长度为3的事件序列

## 数据格式要求

### 结果数据保存格式

```
result/
├── model1/
│   ├── suicide_ied/
│   │   ├── graph_1.json
│   │   ├── graph_2.json
│   │   ├── ...
│   │   └── merged_graph_suicide_ied.json  # 融合图
│   ├── wiki_ied_bombings/
│   │   └── merged_graph_wiki_ied_bombings.json
│   └── ...
├── model2/
│   └── ...
└── evaluation_results_YYYYMMDD_HHMMSS.json  # 评测结果
```

### Ground Truth数据格式

Ground truth数据位于：`dataset/processedData/extracted_data/event_graphs_dev.json`

格式为：
```json
{
  "attack_type": [
    {
      "schema_id": "...",
      "events": [
        {
          "event_id": "...",
          "event_type": "Type.Subtype.Sub_subtype",
          ...
        }
      ],
      "temporal_relations": [
        {
          "before": "event_id_1",
          "after": "event_id_2"
        }
      ]
    }
  ]
}
```

**注意**：事件序列通过`temporal_relations`字段提取，其中`before`事件在时序上早于`after`事件。

## 使用方法

### 1. 确保结果数据按照规定格式保存

模型的结果应该保存在 `result/{model_name}/{attack_type}/` 目录下，并包含融合图文件（以`merged_graph_`开头的JSON文件）。

### 2. 运行评测

```bash
cd evaluate
python evaluator.py
```

### 3. 查看结果

评测完成后，结果会保存在 `result/evaluation_results_YYYYMMDD_HHMMSS.json` 文件中。

## 评测结果格式

```json
{
  "model1": {
    "suicide_ied": {
      "graph_path": "...",
      "attack_type": "suicide_ied",
      "metrics": {
        "topic_overlap": {
          "precision": 0.75,
          "recall": 0.68,
          "f1": 0.71,
          "generated_types": [...],
          "ground_truth_types": [...],
          "matched_types": [...]
        },
        "sequence_2_matching": {
          "precision": 0.60,
          "recall": 0.55,
          "f1": 0.57,
          "generated_sequences": [...],
          "ground_truth_sequences": [...],
          "matched_sequences": [...]
        },
        "sequence_3_matching": {
          "precision": 0.45,
          "recall": 0.40,
          "f1": 0.42,
          "generated_sequences": [...],
          "ground_truth_sequences": [...],
          "matched_sequences": [...]
        }
      }
    }
  }
}
```

## 注意事项

1. 确保ground truth数据路径正确
2. 确保模型生成的图数据格式符合要求
3. 评测过程会在控制台输出详细的评测信息
4. 评测结果会同时保存到日志文件和JSON文件中

