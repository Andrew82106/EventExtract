# -*- coding: utf-8 -*-
import os
import json
from typing import List, Dict, Tuple
from llm_service import LLMService
from prompt_templates import PromptTemplates
from evaluation import SchemaEvaluator
import numpy as np


def extract_text_and_event_graph(json_data: Dict) -> Dict:
    """
    从单条JSON数据中提取原文本和事件图信息
    """
    result = {
        "original_texts": [],  # 原文本集合
        "event_graph": {       # 事件图信息
            "clusters": []     # 事件集群（每个cluster包含steps）
        }
    }
    
    # 提取原文本（来自provenanceData中的sentence或text）
    schemas = json_data.get("schemas", [])
    for schema in schemas:
        provenance_data = schema.get("provenanceData", [])
        for item in provenance_data:
            # 优先取完整句子，若没有则取text字段
            if "sentence" in item:
                result["original_texts"].append(item["sentence"])
            elif "text" in item:
                result["original_texts"].append(item["text"])
    
    # 提取事件图（来自steps，包含事件类型、名称、参与者等）
    for schema in schemas:
        cluster_info = {
            "cluster_id": schema.get("@id"),
            "cluster_name": schema.get("name"),
            "steps": []  # 事件步骤（事件图的核心）
        }
        steps = schema.get("steps", [])
        for step in steps:
            event_step = {
                "event_id": step.get("@id"),
                "event_type": step.get("@type"),  # 事件类型（如Movement.Transportation.Unspecified）
                "event_name": step.get("name"),   # 事件名称
                "participants": step.get("participants", [])  # 事件参与者
            }
            cluster_info["steps"].append(event_step)
        result["event_graph"]["clusters"].append(cluster_info)
    
    # 去重原文本（避免重复）
    result["original_texts"] = list(set(result["original_texts"]))
    return result


def load_ground_truth(file_path: str) -> List[Dict]:
    """加载JSONL文件中所有标注数据的真实标签"""
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json_data = json.loads(line)
                # 使用extract_text_and_event_graph提取数据
                extracted = extract_text_and_event_graph(json_data)
                
                # 将提取的数据转换为评估所需的格式
                # 合并所有原文本
                full_text = " ".join(extracted["original_texts"])
                
                # 提取实体信息（从participants中）
                entities = []
                for cluster in extracted["event_graph"]["clusters"]:
                    for step in cluster["steps"]:
                        for participant in step["participants"]:
                            entity_info = {
                                "实体文本": participant.get("name", ""),
                                "类型": participant.get("entityTypes", "").split(".")[-1] if participant.get("entityTypes") else "PER"
                            }
                            if entity_info["实体文本"] and entity_info["类型"]:
                                entities.append(entity_info)
                
                # 提取事件信息
                events = []
                for cluster in extracted["event_graph"]["clusters"]:
                    for step in cluster["steps"]:
                        # 从事件名称中提取触发词（去掉参数占位符）
                        event_name = step.get("event_name", "")
                        trigger_word = ""
                        if event_name:
                            # 移除<arg1>, <arg2>等占位符，保留动词
                            import re
                            trigger_word = re.sub(r'<arg\d+>', '', event_name).strip()
                        
                        event_info = {
                            "触发词": trigger_word,
                            "事件类型": step.get("event_type", "").split(".")[-1] if step.get("event_type") else ""
                        }
                        if event_info["触发词"] and event_info["事件类型"]:
                            events.append(event_info)
                
                all_data.append({
                    "text": full_text,
                    "entities": entities,
                    "events": events,
                    "relations": [],  # 暂时为空，需要根据实际数据结构补充
                    "temporal_relations": [],  # 暂时为空，需要根据实际数据结构补充
                    "coreferences": []  # 暂时为空，需要根据实际数据结构补充
                })
            except json.JSONDecodeError as e:
                print(f"解析JSON行时出错: {e}")
                continue
    return all_data


def normalize_entity(entity: str, coref_map: Dict[str, str]) -> str:
    """使用共指映射规范化实体名称"""
    return coref_map.get(entity, entity)


def calculate_entity_f1(predicted: List[Dict], ground_truth: List[Dict], coref_map: Dict[str, str], 
                       evaluator: SchemaEvaluator) -> Tuple[float, float, float]:
    """计算实体识别的F1值（使用词相似度匹配）"""
    # 应用共指消解
    normalized_predicted = []
    for ent in predicted:
        text = normalize_entity(ent.get("实体文本", ""), coref_map)
        ent_type = ent.get("类型", "")
        if text and ent_type:
            normalized_predicted.append({"实体文本": text, "类型": ent_type})
    
    # 使用新的评估器
    result = evaluator.evaluate_entities(normalized_predicted, ground_truth)
    return (result["precision"], result["recall"], result["f1"])


def calculate_relation_f1(predicted: List[Dict], ground_truth: List[Dict], coref_map: Dict[str, str], 
                         evaluator: SchemaEvaluator) -> Tuple[float, float, float]:
    """计算实体关系的F1值（使用词相似度匹配）"""
    # 应用共指消解
    normalized_predicted = []
    for rel in predicted:
        e1 = normalize_entity(rel.get("实体1", ""), coref_map)
        e2 = normalize_entity(rel.get("实体2", ""), coref_map)
        rel_type = rel.get("关系类型", "")
        if e1 and e2 and rel_type:
            normalized_predicted.append({"实体1": e1, "实体2": e2, "关系类型": rel_type})
    
    # 使用新的评估器
    result = evaluator.evaluate_relations(normalized_predicted, ground_truth)
    return (result["precision"], result["recall"], result["f1"])


def calculate_event_f1(predicted: List[Dict], ground_truth: List[Dict], coref_map: Dict[str, str], 
                      evaluator: SchemaEvaluator) -> Tuple[float, float, float]:
    """计算事件识别的F1值（使用词相似度匹配）"""
    # 使用新的评估器
    result = evaluator.evaluate_events(predicted, ground_truth)
    return (result["precision"], result["recall"], result["f1"])


def calculate_temporal_f1(predicted: List[Dict], ground_truth: List[Dict], 
                         evaluator: SchemaEvaluator) -> Tuple[float, float, float]:
    """计算时间关系的F1值（使用词相似度匹配）"""
    # 使用新的评估器
    result = evaluator.evaluate_temporal_relations(predicted, ground_truth)
    return (result["precision"], result["recall"], result["f1"])


def evaluate_file(llm_service: LLMService, templates: Dict, file_path: str, evaluator: SchemaEvaluator) -> List[Dict]:
    """评估单个文件中的所有数据项并返回各项指标"""
    print(f"处理文件: {file_path}")
    all_ground_truth = load_ground_truth(file_path)
    all_results = []
    
    for i, ground_truth in enumerate(all_ground_truth):
        print(f"  处理第 {i+1} 条数据")
        text = ground_truth["text"]
        if not text:
            print(f"  第 {i+1} 条数据文本为空，跳过")
            continue
            
        # 1. 实体识别
        entities_pred = llm_service.extract_entities(text, templates["ner"]) or []

        # 2. 事件识别
        events_pred = llm_service.extract_events(text, entities_pred, templates["event_extraction"]) or []

        # 3. 共指消解
        entity_names = [e.get("实体文本") for e in entities_pred if isinstance(e, dict)]
        coref_pred = llm_service.resolve_coreference(entity_names, templates["coreference"]) or []
        coref_map = {}
        for item in coref_pred:
            canonical = item.get("统一实体")
            mentions = item.get("指代表述", [])
            if canonical:
                for m in mentions:
                    if isinstance(m, str) and m:
                        coref_map[m] = canonical

        # 4. 实体关系抽取
        relations_pred = llm_service.extract_entity_relations(entity_names, templates["entity_relation"]) or []

        # 5. 时间关系抽取
        temporal_pred = llm_service.extract_temporal_relations(text, events_pred, templates["temporal_relation"]) or []

        # 计算各项F1值（使用新的评估器）
        ent_p, ent_r, ent_f1 = calculate_entity_f1(entities_pred, ground_truth["entities"], coref_map, evaluator)
        rel_p, rel_r, rel_f1 = calculate_relation_f1(relations_pred, ground_truth["relations"], coref_map, evaluator)
        evt_p, evt_r, evt_f1 = calculate_event_f1(events_pred, ground_truth["events"], coref_map, evaluator)
        tmp_p, tmp_r, tmp_f1 = calculate_temporal_f1(temporal_pred, ground_truth["temporal_relations"], evaluator)

        all_results.append({
            "entity": (ent_p, ent_r, ent_f1),
            "relation": (rel_p, rel_r, rel_f1),
            "event": (evt_p, evt_r, evt_f1),
            "temporal": (tmp_p, tmp_r, tmp_f1)
        })
        print("*"*80)
        print(f"  实体识别F1: {ent_f1:.2f}")
        print(f"  实体关系抽取F1: {rel_f1:.2f}")
        print(f"  事件识别F1: {evt_f1:.2f}")
        print(f"  时间关系抽取F1: {tmp_f1:.2f}")
        print("*"*80)
    
    return all_results


def main():
    """主函数：批量处理测试数据并计算平均F1值"""
    # API密钥（请替换为您的实际密钥）
    API_KEY = "34dc26d36ea34c3fb4712d530f267368.UFOypFDFH7mF0Zif"

    # 测试数据路径
    TEST_DATA_PATH = "/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/师兄方法/dataset/LDC_schema_corpus_ce_split/dev"

    # 初始化服务
    llm_service = LLMService(api_key=API_KEY)
    templates = PromptTemplates.get_templates()
    evaluator = SchemaEvaluator(similarity_threshold=0.7)  # 可调整相似度阈值

    # 收集所有JSON文件
    json_files = [f for f in os.listdir(TEST_DATA_PATH) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个测试文件")

    # 存储所有文件的评估结果
    all_results = []

    # 批量处理
    for json_file in json_files:
        file_path = os.path.join(TEST_DATA_PATH, json_file)
        try:
            file_results = evaluate_file(llm_service, templates, file_path, evaluator)
            all_results.extend(file_results)  # 扩展而不是追加
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
            continue

    # 计算平均值
    if all_results:
        entity_f1s = [r["entity"][2] for r in all_results]
        relation_f1s = [r["relation"][2] for r in all_results]
        event_f1s = [r["event"][2] for r in all_results]
        temporal_f1s = [r["temporal"][2] for r in all_results]

        avg_entity_f1 = np.mean(entity_f1s)
        avg_relation_f1 = np.mean(relation_f1s)
        avg_event_f1 = np.mean(event_f1s)
        avg_temporal_f1 = np.mean(temporal_f1s)
        overall_f1 = np.mean([avg_entity_f1, avg_relation_f1, avg_event_f1, avg_temporal_f1])

        # 输出结果
        print("\n===== 评估结果 =====")
        print(f"实体识别平均F1值: {avg_entity_f1:.4f}")
        print(f"实体关系平均F1值: {avg_relation_f1:.4f}")
        print(f"事件识别平均F1值: {avg_event_f1:.4f}")
        print(f"时间关系平均F1值: {avg_temporal_f1:.4f}")
        print(f"总体平均F1值: {overall_f1:.4f}")
    else:
        print("没有成功处理任何测试文件")


if __name__ == "__main__":
    main()