# -*- coding: utf-8 -*-
from llm_service import LLMService
from prompt_templates import PromptTemplates
from graph_visualizer import visualize_event_graph


def main():
    """主函数：演示如何使用拆分后的代码"""
    
    # API密钥（请替换为您的实际密钥）
    API_KEY = "6c735ef4e888bbfa5f66175870a62901.1c1jin8KHtZqNW90"
    
    # 示例文本
    SAMPLE_TEXT = "GhostWriter 至少从 2017 年开始活跃，此前观察到它冒充立陶宛、拉脱维亚和波兰的记者，向当地观众传播虚假信息和反北约言论。近日，波兰政府警告称，GhostWriter 近期一直关注波兰，试图侵入电子邮件账户以收集敏感信息，并控制社交媒体账户以传播虚假信息。"
    
    print("=== 使用拆分后的代码构建事件图 ===")
    
    # 方式1：使用完整的事件图构建器（推荐）
    # print("\n1. 使用完整的事件图构建器：")
    # builder = DynamicEventGraphBuilder(api_key=API_KEY)
    # event_graph = builder.build_event_graph(SAMPLE_TEXT)
    
    # if event_graph:
    #     print("事件图构建成功！")
    #     print(f"实体数量: {len(event_graph['nodes']['entities'])}")
    #     print(f"事件数量: {len(event_graph['nodes']['events'])}")
    # else:
    #     print("事件图构建失败！")
    
    # 方式2：直接使用LLM服务（高级用法）
    # print("\n2. 直接使用LLM服务：")
    # llm_service = LLMService(api_key=API_KEY)
    # templates = PromptTemplates.get_templates()
    
    # 单独调用实体识别
    #entities = llm_service.extract_entities(SAMPLE_TEXT, templates["ner"])
    # print(f"识别到 {len(entities)} 个实体")
    
    # 单独调用事件识别
    #events = llm_service.extract_events(SAMPLE_TEXT, entities, templates["event_extraction"])
    # print(f"识别到 {len(events)} 个事件")

    # 共指消解
    #resolved_list = llm_service.resolve_coreference(SAMPLE_TEXT, templates["coreference"])
    # print(f"共指消解结果：{resolved_list}")

    # 实体关系抽取
    #entity_relations = llm_service.extract_entity_relations(SAMPLE_TEXT, templates["entity_relation"])
    # print(f"识别到 {len(entity_relations)} 个实体关系")

    # 事件时间关系推理
    #temporal_relations = llm_service.extract_temporal_relations(SAMPLE_TEXT, events, templates["temporal_relation"])
    # print(f"识别到 {len(temporal_relations)} 个事件时间关系")
    
    # 方式3：使用单独的模板方法
    # print("\n3. 使用单独的模板方法：")
    # ner_template = PromptTemplates.get_ner_template()
    # print("实体识别模板已获取")

    # 方式4：一键可视化输出 HTML
    print("\n4. 生成事件图可视化（HTML）到 result/：")
    html_path = visualize_event_graph(SAMPLE_TEXT, API_KEY)
    print(f"可视化已生成：{html_path}")


if __name__ == "__main__":
    main()
