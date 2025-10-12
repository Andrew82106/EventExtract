# 事件骨架图构建系统

基于大语言模型的攻击事件骨架图自动构建系统，用于从文本中提取事件流程并构建标准化的事件图谱。

## 快速使用指南

### 1. 环境配置

```bash
# 安装依赖
pip install -r model1/requirements.txt  # 或 model2/requirements.txt

# 配置 API 密钥（在 model1/config.py 或 model2/config.py 中）
ZHIPU_API_KEY = "your_api_key_here"
```

### 2. 选择攻击类型

**支持的攻击类型：**
- `suicide_ied` - 自杀式简易爆炸装置（默认）
- `backpack_ied` - 背包式简易爆炸装置
- `road_ied` - 路边简易爆炸装置
- `wiki_drone_strikes` - 无人机袭击
- `wiki_ied_bombings` - 简易爆炸装置爆炸
- `wiki_mass_car_bombings` - 大规模汽车炸弹

**如何更改攻击类型：**

通过命令行参数指定：
```bash
python main.py --attack_type suicide_ied
```

或在 `config.py` 中修改默认值：
```python
# 在 model1/config.py 或 model2/config.py 中
ATTACK_TYPES = [
    'suicide_ied',
    'backpack_ied',
    # ... 其他类型
]
```

### 3. 选择和运行模型

项目提供两种模型算法：

#### Model1：简单询问式构图（推荐用于快速测试）
- **特点**：运行快速，实现简单
- **适用场景**：快速原型、批量处理

```bash
cd model1

# 使用默认攻击类型（suicide_ied）
python main.py

# 指定攻击类型
python main.py --attack_type backpack_ied

# 指定 API 密钥
python main.py --attack_type suicide_ied --api_key YOUR_API_KEY
```

#### Model2：基于意义段的迭代构图（推荐用于高质量结果）
- **特点**：更准确，覆盖更全面，有支撑文本
- **适用场景**：需要高质量结果的场景

```bash
cd model2

# 使用默认攻击类型
python main.py

# 指定攻击类型和 API 密钥
python main.py --attack_type suicide_ied --api_key YOUR_API_KEY
```

**运行结果：**
- 结果保存在 `result/{model_name}/{attack_type}/` 目录
- `graph_1.json`, `graph_2.json`, ... : 单个文本的事件图
- `merged_graph_{attack_type}.json` : 融合后的骨架图

### 4. 评测模型结果

#### 方法1：统一评测（推荐）

评测所有模型和攻击类型：

```bash
cd evaluate
python unified_evaluator.py
```

**如何更改评测的攻击类型：**

在 `unified_evaluator.py` 中修改：
```python
# 找到这一行并修改
ATTACK_TYPES_TO_EVALUATE = ['suicide_ied', 'backpack_ied']  # 添加或删除攻击类型
```

**如何更改评测的模型：**

在 `unified_evaluator.py` 中修改：
```python
# 找到这一行并修改
MODELS_TO_EVALUATE = ['model1', 'model2']  # 添加或删除模型
```

#### 方法2：单独评测某个模型和攻击类型

```bash
cd evaluate
python evaluator.py
```

在 `evaluator.py` 中修改路径：
```python
# 指定要评测的融合图路径
graph_path = "../result/model1/suicide_ied/merged_graph_suicide_ied.json"
# 指定对应的 ground truth 路径
ground_truth_path = "../dataset/processedData/extracted_data/event_graphs_train.json"
# 指定攻击类型
attack_type = "suicide_ied"
```

#### 方法3：递进式图合并评测

测试不同数量的图合并后的效果（1个图、2个图、...、N个图）：

```bash
cd evaluate
python layered_merger_evaluator.py
```

**配置评测参数：**

在 `layered_merger_evaluator.py` 的 `main()` 函数中修改：
```python
# 指定包含单个图的目录
GRAPHS_DIR = "../result/model2/suicide_ied"

# 指定 ground truth 路径
GROUND_TRUTH_PATH = "../dataset/processedData/extracted_data/event_graphs_train.json"

# 指定输出目录
OUTPUT_DIR = "../result/layered_evaluation"
```

#### 方法4：重新合并和评测

修改了图融合算法后，无需重新运行模型，直接重新合并并评测：

```bash
cd evaluate

# 重新合并所有模型的所有攻击类型
python remerge_and_evaluate.py

# 只重新合并 model1 的所有攻击类型
python remerge_and_evaluate.py --model model1

# 只重新合并 model1 的 suicide_ied
python remerge_and_evaluate.py --model model1 --attack_type suicide_ied

# 只重新合并，不评测
python remerge_and_evaluate.py --no_evaluate
```

### 5. 查看评测结果

评测结果保存在：
- `result/unified_evaluation/` - 统一评测结果
- `result/layered_evaluation/` - 递进式评测结果

**评测指标说明：**
- **事件类型匹配 F1**：生成的事件类型与 ground truth 的匹配度
- **序列匹配 F1（长度2）**：两事件序列的匹配度
- **序列匹配 F1（长度3）**：三事件序列的匹配度

## 常用操作场景

### 场景1：测试新的攻击类型

```bash
# 步骤1：准备数据
# 在 dataset/processedData/extracted_texts/ 下创建新目录，如 new_attack/
# 将文本文件放入该目录

# 步骤2：运行模型
cd model1  # 或 model2
python main.py --attack_type new_attack

# 步骤3：评测（如果有 ground truth）
cd ../evaluate
# 在 unified_evaluator.py 中添加 'new_attack' 到评测列表
python unified_evaluator.py
```

### 场景2：对比不同模型的效果

```bash
# 步骤1：运行两个模型
cd model1
python main.py --attack_type suicide_ied

cd ../model2
python main.py --attack_type suicide_ied

# 步骤2：统一评测对比
cd ../evaluate
python unified_evaluator.py

# 查看结果文件，对比 model1 和 model2 的 F1 分数
```

### 场景3：修改图融合算法后重新评测

```bash
# 步骤1：修改融合算法
# 编辑 mergeGraph/graph_merger.py 中的融合逻辑

# 步骤2：重新合并并评测（无需重新运行模型）
cd evaluate
python remerge_and_evaluate.py

# 查看新的评测结果
```

### 场景4：批量处理多个攻击类型

```bash
# 创建批处理脚本
cd model1  # 或 model2

# 方法1：使用 shell 循环
for attack in suicide_ied backpack_ied road_ied; do
    python main.py --attack_type $attack
done

# 方法2：修改 config.py 中的配置，然后批量运行
```

## 项目结构

```
├── dataset/                          # 数据集
│   ├── kairos_ontology.xlsx          # 事件本体（三层分类）
│   └── processedData/
│       ├── extracted_texts/          # 文本数据（按攻击类型分类）
│       └── extracted_data/           # Ground truth
├── model1/                           # 简单询问式构图
├── model2/                           # 基于意义段的迭代构图
├── mergeGraph/                       # 图融合模块
├── evaluate/                         # 评测系统
├── result/                           # 运行结果
│   ├── model1/{attack_type}/         
│   ├── model2/{attack_type}/
│   ├── unified_evaluation/           # 统一评测结果
│   └── layered_evaluation/           # 递进式评测结果
└── llm_service.py                    # LLM 服务封装
```

## 高级配置

### 修改 LLM 模型

在 `llm_service.py` 中配置：
```python
self.advanced_model = "glm-4.6"      # 高级模型（更准确，费用高）
self.basic_model = "glm-z1-flash"    # 基础模型（快速，免费）
```

### 修改采样数量

在 `model1/config.py` 或 `model2/config.py` 中：
```python
MAX_TEXT_SAMPLES = 20  # 每个攻击类型最多处理多少个文本
```

### 启用断点续传

在 `config.py` 中：
```python
ENABLE_RESUME = True  # 跳过已处理的文本，从断点继续
```

## 附加说明

- **算法详情**：查看 `algorithm.md`
- **Model1 详情**：查看 `model1/README.md`
- **Model2 详情**：查看 `model2/README.md`
- **评测详情**：查看 `evaluate/README.md` 和 `evaluate/LAYERED_EVALUATION_README.md`

## 常见问题

**Q: API 调用失败怎么办？**
A: 检查网络连接和 API 密钥，确认配额充足。

**Q: 如何添加新的攻击类型？**
A: 在 `dataset/processedData/extracted_texts/` 下创建新目录，放入文本文件，然后运行 `python main.py --attack_type new_type`

**Q: 如何对比两个模型的效果？**
A: 分别运行两个模型，然后使用 `python unified_evaluator.py` 统一评测。

**Q: 修改了图融合算法如何快速测试？**
A: 使用 `python remerge_and_evaluate.py` 重新合并和评测，无需重新运行模型。
