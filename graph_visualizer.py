# -*- coding: utf-8 -*-
"""
事件图可视化模块
将 LLM 输出（实体、事件、共指、实体关系、时间关系）汇总并可视化为 HTML（vis-network）。
"""
import os
import json
from typing import Dict, List, Tuple

from llm_service import LLMService
from prompt_templates import PromptTemplates


def _build_coref_map(coref_list: List[Dict]) -> Dict[str, str]:
    """将共指列表转换为：提及 -> 统一实体 名称 的映射。"""
    mention_to_canonical: Dict[str, str] = {}
    if not isinstance(coref_list, list):
        return mention_to_canonical
    for item in coref_list:
        try:
            canonical = item.get("统一实体")
            mentions = item.get("指代表述", [])
            if not canonical:
                continue
            for m in mentions:
                if isinstance(m, str) and m:
                    mention_to_canonical[m] = canonical
        except Exception:
            continue
    return mention_to_canonical


def _canonicalize(name: str, coref_map: Dict[str, str]) -> str:
    if not name:
        return name
    return coref_map.get(name, name)


def _normalize_entities(entities: List[Dict], coref_map: Dict[str, str]) -> List[Dict]:
    """应用共指映射，规范化实体文本。"""
    normalized: List[Dict] = []
    if not isinstance(entities, list):
        return normalized
    for ent in entities:
        text = ent.get("实体文本")
        ent_type = ent.get("类型")
        normalized.append({
            "实体文本": _canonicalize(text, coref_map),
            "类型": ent_type,
        })
    return normalized


def _collect_entity_nodes(entities: List[Dict]) -> Dict[str, Dict]:
    """根据实体列表构建去重后的实体节点字典：name -> node"""
    nodes: Dict[str, Dict] = {}
    for ent in entities:
        name = ent.get("实体文本")
        ent_type = ent.get("类型", "实体")
        if not name:
            continue
        if name not in nodes:
            nodes[name] = {
                "id": f"entity::{name}",
                "label": f"{name}\n[{ent_type}]" if ent_type else name,
                "shape": "circle",
                "color": {
                    "background": "#E3F2FD",
                    "border": "#1E88E5"
                }
            }
    return nodes


def _collect_event_nodes_and_edges(
    events: List[Dict],
    coref_map: Dict[str, str],
    entity_nodes: Dict[str, Dict]
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """从事件列表中构建事件节点和 事件-实体 边。"""
    event_nodes: Dict[str, Dict] = {}
    arg_edges: List[Dict] = []
    if not isinstance(events, list):
        return event_nodes, arg_edges

    for idx, ev in enumerate(events):
        trigger = ev.get("触发词") or ev.get("事件触发词") or "事件"
        ev_type = ev.get("事件类型") or "事件"
        event_id = f"event::{ev_type}::{idx}"
        if event_id not in event_nodes:
            event_nodes[event_id] = {
                "id": event_id,
                "label": f"{ev_type}\n<{trigger}>",
                "shape": "triangle",
                "color": {
                    "background": "#FFECB3",
                    "border": "#FB8C00"
                }
            }

        # 论元边：实体 -> 事件
        arguments = ev.get("论元", {})
        if isinstance(arguments, dict):
            for role, arg_entity in arguments.items():
                if not isinstance(role, str):
                    continue
                if not isinstance(arg_entity, str):
                    continue
                canonical_ent = _canonicalize(arg_entity, coref_map)
                if canonical_ent not in entity_nodes:
                    # 如果实体未在实体列表中，也创建一个通用实体节点
                    entity_nodes[canonical_ent] = {
                        "id": f"entity::{canonical_ent}",
                        "label": canonical_ent,
                        "shape": "circle",
                        "color": {
                            "background": "#E3F2FD",
                            "border": "#1E88E5"
                        }
                    }
                arg_edges.append({
                    "from": entity_nodes[canonical_ent]["id"],
                    "to": event_id,
                    "label": role
                })
    return event_nodes, arg_edges


def _collect_entity_relation_edges(relations: List[Dict], coref_map: Dict[str, str], entity_nodes: Dict[str, Dict]) -> List[Dict]:
    edges: List[Dict] = []
    if not isinstance(relations, list):
        return edges
    for rel in relations:
        e1 = _canonicalize(rel.get("实体1"), coref_map)
        r = rel.get("关系类型") or rel.get("关系") or "关系"
        e2 = _canonicalize(rel.get("实体2"), coref_map)
        if not e1 or not e2:
            continue
        # 确保节点存在
        for name in (e1, e2):
            if name not in entity_nodes:
                entity_nodes[name] = {
                    "id": f"entity::{name}",
                    "label": name,
                    "shape": "circle",
                    "color": {
                        "background": "#E3F2FD",
                        "border": "#1E88E5"
                    }
                }
        edges.append({
            "from": entity_nodes[e1]["id"],
            "to": entity_nodes[e2]["id"],
            "label": r
        })
    return edges


def _collect_temporal_edges(temporal_list: List[Dict], event_nodes: Dict[str, Dict]) -> List[Dict]:
    """时间关系边：事件 -> 事件。根据事件类型名称匹配最相近的事件节点。"""
    edges: List[Dict] = []
    if not isinstance(temporal_list, list):
        return edges

    # 预构建：事件类型 到 节点ID 列表 的倒排索引
    type_to_event_ids: Dict[str, List[str]] = {}
    for ev_id, node in event_nodes.items():
        label = node.get("label", "")
        # label 形如 "事件类型\n<触发词>"，取第一行
        ev_type = label.split("\n")[0] if label else "事件"
        type_to_event_ids.setdefault(ev_type, []).append(ev_id)

    for item in temporal_list:
        # 兼容不同键名
        t1 = item.get("事件类型1") or item.get("事件1") or item.get("事件甲")
        rel = item.get("时间关系") or item.get("关系") or "时间关系"
        t2 = item.get("事件类型2") or item.get("事件2") or item.get("事件乙")
        if not t1 or not t2:
            continue
        # 选择每个类型的第一个事件节点（若有多个）
        ids1 = type_to_event_ids.get(t1, [])
        ids2 = type_to_event_ids.get(t2, [])
        if not ids1 or not ids2:
            continue
        edges.append({
            "from": ids1[0],
            "to": ids2[0],
            "label": rel,
            "arrows": "to"
        })
    return edges


def visualize_event_graph(text: str, api_key: str, output_html: str = None) -> str:
    """
    从文本出发，调用 LLMService，整合输出并生成交互式 HTML 事件图。

    Args:
        text: 输入文本
        api_key: 智谱 API Key
        output_html: 结果 HTML 路径（可选，不传则默认写入 result/event_graph.html）

    Returns:
        实际写入的 HTML 文件路径
    """
    if not output_html:
        output_html = os.path.join(os.path.dirname(__file__), "result", "event_graph.html")

    # 确保目录存在
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    # 准备服务与模板
    llm = LLMService(api_key=api_key)
    tmpls = PromptTemplates.get_templates()

    # 1) 实体
    entities = llm.extract_entities(text, tmpls["ner"]) or []

    # 2) 事件
    events = llm.extract_events(text, entities, tmpls["event_extraction"]) or []

    # 3) 共指
    # 注意：接口定义为 resolve_coreference(entity_list, template)
    # 这里传入实体文本列表，增强效果
    entity_names = [e.get("实体文本") for e in entities if isinstance(e, dict)]
    coref = llm.resolve_coreference(entity_names, tmpls["coreference"]) or []
    coref_map = _build_coref_map(coref)

    # 应用共指规范化实体
    entities_norm = _normalize_entities(entities, coref_map)

    # 4) 实体关系
    relations = llm.extract_entity_relations(entity_names, tmpls["entity_relation"]) or []

    # 5) 时间关系
    temporal = llm.extract_temporal_relations(text, events, tmpls["temporal_relation"]) or []

    # 构建节点与边
    entity_nodes = _collect_entity_nodes(entities_norm)
    event_nodes, arg_edges = _collect_event_nodes_and_edges(events, coref_map, entity_nodes)
    relation_edges = _collect_entity_relation_edges(relations, coref_map, entity_nodes)
    temporal_edges = _collect_temporal_edges(temporal, event_nodes)

    # 汇总为 vis-network 数据
    nodes = list(entity_nodes.values()) + list(event_nodes.values())
    edges = []
    edges.extend(arg_edges)
    edges.extend(relation_edges)
    edges.extend(temporal_edges)

    # 边去重：按 (from, to, label) 简单去重，保留首个
    def _deduplicate_edges(input_edges):
        seen = set()
        unique = []
        for e in input_edges:
            k = (e.get("from"), e.get("to"), e.get("label"))
            if k in seen:
                continue
            seen.add(k)
            unique.append(e)
        return unique

    edges = _deduplicate_edges(edges)

    html_content = _render_vis_network_html(nodes, edges)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html


def _render_vis_network_html(nodes: List[Dict], edges: List[Dict]) -> str:
    """渲染一个独立的 HTML，引用项目自带的 vis 资源（相对路径）。"""
    # 路径从 result/ 下的 HTML 指向 lib/
    css_path = "../lib/vis-9.1.2/vis-network.css"
    js_path = "../lib/vis-9.1.2/vis-network.min.js"

    # 预格式化 JSON 数据
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)

    # 简单布局参数
    return f"""
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>事件图可视化</title>
  <link rel=\"stylesheet\" href=\"{css_path}\" />
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; margin: 0; }}
    #mynetwork {{ width: 100vw; height: 100vh; border-top: 1px solid #eee; }}
    .legend {{ position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,.9); padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 4px 0; }}
    .shape-circle {{ width: 12px; height: 12px; background: #E3F2FD; border: 2px solid #1E88E5; border-radius: 50%; }}
    .shape-triangle {{ width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-bottom: 14px solid #FB8C00; }}
  </style>
  <script src=\"{js_path}\"></script>
  <script>
    const nodes = {nodes_json};
    const edges = {edges_json};
    window.addEventListener('DOMContentLoaded', () => {{
      const container = document.getElementById('mynetwork');
      const data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
      const options = {{
        layout: {{
          improvedLayout: true
        }},
        physics: {{
          enabled: true,
          solver: 'forceAtlas2Based',
          stabilization: {{ iterations: 150 }}
        }},
        interaction: {{ hover: true, tooltipDelay: 150 }}
      }};
      new vis.Network(container, data, options);
    }});
  </script>
  </head>
<body>
  <div class=\"legend\">
    <div class=\"legend-item\"><span class=\"shape-circle\"></span> 实体（名词）</div>
    <div class=\"legend-item\"><span class=\"shape-triangle\"></span> 事件（动词）</div>
  </div>
  <div id=\"mynetwork\"></div>
</body>
</html>
"""


