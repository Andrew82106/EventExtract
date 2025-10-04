# 递进式图合并评测说明

## 功能概述

递进式图合并评测器（`layered_merger_evaluator.py`）用于测试不同层级的图合并效果：

- **层级1**：测试每个单独的图 vs ground truth
- **层级2**：测试两两合并的图 vs ground truth
- **层级3**：测试三个图合并 vs ground truth
- **...**
- **层级N**：测试所有图合并 vs ground truth

## 评测指标

对每个层级，计算以下指标：

### 1. 事件类型匹配（Event Types）
- **F1分数**：综合评价指标
- **Precision**：预测的事件类型中有多少是正确的
- **Recall**：ground truth中的事件类型有多少被预测出来

### 2. 事件序列匹配（长度2）
- 匹配形如 `(EventA -> EventB)` 的事件序列
- 同样计算F1、Precision、Recall

### 3. 事件序列匹配（长度3）
- 匹配形如 `(EventA -> EventB -> EventC)` 的事件序列
- 同样计算F1、Precision、Recall

## 使用方法

### 方法1：直接运行脚本

```bash
cd evaluate
python layered_merger_evaluator.py
```

### 方法2：使用shell脚本

```bash
cd evaluate
./run_layered_evaluation.sh
```

### 方法3：在Python中使用

```python
from layered_merger_evaluator import LayeredMergerEvaluator

evaluator = LayeredMergerEvaluator(
    graphs_dir="/path/to/graphs/folder",
    ground_truth_path="/path/to/ground_truth.json",
    output_dir="/path/to/output"
)

results = evaluator.run_layered_evaluation()
```

## 配置说明

在 `layered_merger_evaluator.py` 的 `main()` 函数中硬编码了以下路径：

```python
GRAPHS_DIR = "/path/to/result/model2/suicide_ied"
GROUND_TRUTH_PATH = "/path/to/dataset/processedData/extracted_data/event_graphs_train.json"
OUTPUT_DIR = "/path/to/result/layered_evaluation"
```

如需测试其他数据，请修改这些路径。

**注意**：`GROUND_TRUTH_PATH` 指向的是包含所有攻击类型的统一 JSON 文件（`event_graphs_train.json`），而不是某个攻击类型的单独文件。该文件的格式为：

```json
{
  "suicide_ied": [
    {
      "events": [...],
      "temporal_relations": [...]
    }
  ],
  "other_attack_type": [...]
}
```

评测器会根据 `GRAPHS_DIR` 的文件夹名称（例如 `suicide_ied`）自动从 ground truth 中提取对应攻击类型的数据。

## 输出结果

### 1. 控制台输出

显示每个层级的平均评测结果：

```
==============================================================
层级 1: 测试 1 个图的合并效果
==============================================================
  测试 15 个单独的图...

  层级 1 平均结果:
    事件类型匹配:
      F1:        0.6523
      Precision: 0.7012
      Recall:    0.6089
    序列匹配(长度2):
      F1:        0.4521
      Precision: 0.5123
      Recall:    0.4034
    ...
```

### 2. JSON结果文件

保存在 `result/layered_evaluation/` 目录下，文件名格式：

```
layered_evaluation_{attack_type}_{timestamp}.json
```

结构如下：

```json
{
  "attack_type": "suicide_ied",
  "total_graphs": 15,
  "max_level": 15,
  "timestamp": "2025-10-04 12:34:56",
  "levels": {
    "1": {
      "level": 1,
      "num_combinations": 15,
      "metrics": [...],
      "average": {
        "event_types": {
          "f1": 0.6523,
          "precision": 0.7012,
          "recall": 0.6089
        },
        "sequences_len2": {...},
        "sequences_len3": {...}
      }
    },
    "2": {...},
    ...
  }
}
```

## 层级说明

### 层级1（单图）
- 测试所有单个图，计算平均指标
- 例如：15个图，测试15次

### 层级2（两图合并）
- 测试所有可能的两图组合（如果组合数 > 50，则随机采样50个）
- 例如：C(15,2) = 105种组合，测试50次（采样）

### 层级3-N（多图合并）
- 测试指定数量的图合并
- 如果组合数过多，采样最多50个组合

### 层级N（全合并）
- 合并所有图，测试1次

## 性能优化

- 当组合数超过50时，自动采样以加快评测速度
- 随机种子固定为42，保证结果可重复
- 采样策略：从所有组合中随机选择50个

## 注意事项

1. **ground truth格式**：
   - 必须使用统一的 `event_graphs_train.json` 文件
   - 文件包含所有攻击类型的 ground truth 数据
   - 评测器会自动根据文件夹名称提取对应的攻击类型数据
2. **图文件格式**：只处理 `graph_X.json` 文件，忽略 `merged_graph_X.json`
3. **内存占用**：层级较高时（如层级10+）可能占用较多内存
4. **运行时间**：完整评测15个图的所有层级大约需要10-30分钟（取决于图的大小）

## 扩展功能

### 限制最大层级

如果只想测试前几个层级，可以传入 `max_level` 参数：

```python
# 只测试层级1-5
results = evaluator.run_layered_evaluation(max_level=5)
```

### 修改采样数量

在代码中找到这行：

```python
max_samples = 50  # 最多采样50个组合
```

根据需要修改数量。

## 故障排查

### 问题1：找不到图文件
- 检查 `GRAPHS_DIR` 路径是否正确
- 确认文件夹下有 `graph_X.json` 文件

### 问题2：找不到ground truth
- 检查 `GROUND_TRUTH_PATH` 是否指向 `event_graphs_train.json` 文件
- 确认JSON文件包含对应的攻击类型（如 `suicide_ied`）
- 确认文件夹名称与 ground truth 中的攻击类型键名一致

### 问题3：内存不足
- 减少 `max_samples` 值
- 使用 `max_level` 限制测试层级

## 版本历史

- v1.0.0 (2025-10-04): 初始版本
  - 支持递进式图合并评测
  - 支持事件类型和序列匹配评测
  - 支持自动采样优化性能

