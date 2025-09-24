# 事件图构建系统

这是一个用于从文本中构建事件图的系统，已拆分为三个独立的模块。

- 数据：https://github.com/limanling/temporal-graph-schema/tree/main/data
- 数据分类（ontology）：https://github.com/NextCenturyCorporation/kairos-pub/tree/master/data-format/ontology
- resin事件提取工具：https://github.com/RESIN-KAIROS/RESIN-pipeline-public/tree/api

## 文件结构

### 1. `llm_service.py` - 大语言模型服务
- **功能**: 封装智谱AI API调用逻辑
- **主要类**: `LLMService`
- **职责**: 
  - 处理API调用和响应解析
  - 提供各种NLP任务的接口方法
  - 错误处理和异常管理

### 2. `prompt_templates.py` - 提示词模板配置
- **功能**: 管理所有提示词模板
- **主要类**: `PromptTemplates`
- **职责**:
  - 存储和管理所有阶段的提示词模板
  - 提供模板获取方法
  - 便于模板的维护和修改

### 3. `eventExtract.py` - 主程序（事件图构建器）
- **功能**: 核心业务逻辑和可视化
- **主要类**: `DynamicEventGraphBuilder`
- **职责**:
  - 协调各个NLP任务
  - 构建完整的事件图
  - 提供可视化功能

## 使用方法

### 基本用法（推荐）
```python
from graph_visualizer import visualize_event_graph

# 构建事件图
text = "你的文本内容..."
API_KEY = "API key"
# 方式4：一键可视化输出 HTML
html_path = visualize_event_graph(text, API_KEY)
print(f"可视化已生成：{html_path}")
```

### 高级用法
```python
from llm_service import LLMService
from prompt_templates import PromptTemplates

# 直接使用LLM服务
llm_service = LLMService(api_key="your_api_key")
templates = PromptTemplates.get_templates()

# 单独调用特定任务
entities = llm_service.extract_entities(text, templates["ner"])
events = llm_service.extract_events(text, templates["event_extraction"])
```

## 依赖项

- `zhipuai`: 智谱AI Python SDK
- `networkx`: 图处理库
- `matplotlib`: 静态可视化
- `pyvis`: 交互式可视化
- `jinja2`: HTML模板引擎（pyvis依赖）

## 安装依赖

```bash
pip install zhipuai networkx matplotlib pyvis jinja2
```

## 配置

1. 将您的智谱AI API密钥替换到代码中的 `API_KEY` 变量
2. 根据需要修改 `prompt_templates.py` 中的提示词模板
3. 调整 `llm_service.py` 中的模型参数（如temperature、top_p等）

## 优势

- **模块化**: 代码职责清晰，便于维护
- **可扩展**: 易于添加新的NLP任务或修改提示词
- **可复用**: LLM服务和模板可以独立使用
- **易测试**: 各模块可以独立测试
