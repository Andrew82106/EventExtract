# -*- coding: utf-8 -*-
"""
图可视化工具
输入单图(@graph_*.json)或融合图(@merged_graph_*.json)路径，
生成交互式可视化HTML到项目根目录visualize/下。

扩展：支持从ground truth大JSON (extracted_data/event_graphs_*.json) 中
按attack_type选择一个schema进行可视化；未指定schema_name时随机选择，
指定schema_name时只渲染对应schema。
"""
import os
import sys
import json
from datetime import datetime
import random


# 将项目根目录加入路径，便于引用lib静态资源
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def _load_graph_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _convert_to_vis_data(graph_json):
    """
    将两种可能的图格式规范化为vis-network的数据结构：
    - 单图: nodes[{id,event_type,event_subtype,event_sub_subtype,description?...}],
            edges[{source,target,relation?}]
    - 融合图: nodes[{id(=type_key), occurrence_*...}],
            edges[{source,target,weight,...}]
    """
    nodes = []
    edges = []

    if 'nodes' not in graph_json:
        return nodes, edges

    # 规范化节点
    for n in graph_json.get('nodes', []):
        node_id = n.get('id')
        # 标题与标签
        event_type = n.get('event_type', '')
        event_subtype = n.get('event_subtype', '')
        event_sub_subtype = n.get('event_sub_subtype', '')
        description = n.get('description') or ''
        occurrence = n.get('occurrence_count')
        occ_pct = n.get('occurrence_percentage')

        # label 优先使用层级类型或id
        if event_type:
            parts = [event_type]
            if event_subtype:
                parts.append(event_subtype)
            if event_sub_subtype:
                parts.append(event_sub_subtype)
            label = ' > '.join(parts)
        else:
            # 融合图的节点id通常就是类型键
            label = str(node_id)

        title_lines = [label]
        if description:
            title_lines.append(description)
        if occurrence is not None:
            title_lines.append(f"出现次数: {occurrence}")
        if occ_pct is not None:
            title_lines.append(f"出现占比: {occ_pct:.2f}%")

        nodes.append({
            'id': node_id,
            'label': label,
            'title': '\n'.join(title_lines)
        })

    # 规范化边
    for e in graph_json.get('edges', []):
        source = e.get('source')
        target = e.get('target')
        relation = e.get('relation')
        weight = e.get('weight')
        occ_pct = e.get('occurrence_percentage')

        title_parts = []
        if relation:
            title_parts.append(f"关系: {relation}")
        if weight is not None:
            title_parts.append(f"权重: {weight}")
        if occ_pct is not None:
            title_parts.append(f"出现占比: {occ_pct:.2f}%")

        edges.append({
            'from': source,
            'to': target,
            'arrows': 'to',
            'title': '\n'.join(title_parts) if title_parts else None,
            'value': float(weight) if weight is not None else 1.0
        })

    return nodes, edges


def generate_visualization(file_path, output_dir=None):
    """
    从图JSON文件生成交互式可视化HTML

    Args:
        file_path (str): 图JSON路径（单图或融合图）
        output_dir (str): 输出目录，默认<project_root>/visualize

    Returns:
        str: 生成的HTML文件路径
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'visualize')

    _ensure_output_dir(output_dir)

    graph_json = _load_graph_json(file_path)
    nodes, edges = _convert_to_vis_data(graph_json)

    # 静态资源路径（相对HTML位置）
    rel_vis_js = os.path.relpath(
        os.path.join(PROJECT_ROOT, 'lib', 'vis-9.1.2', 'vis-network.min.js'),
        output_dir
    )
    rel_vis_css = os.path.relpath(
        os.path.join(PROJECT_ROOT, 'lib', 'vis-9.1.2', 'vis-network.css'),
        output_dir
    )

    # 文件名
    model_type = "model1" if "model1" in file_path else ("model2" if "model2" in file_path else "model3")
    base = str(os.path.splitext(os.path.basename(file_path))[0]) + "_" + model_type
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_name = f"{base}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_name)

    # 生成HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Graph Visualization - {base}</title>
  <link rel=\"stylesheet\" href=\"{rel_vis_css}\" />
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; }}
    #mynetwork {{ width: 100%; height: 100vh; border: 1px solid #ddd; }}
    .toolbar {{ position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); z-index: 10; }}
    .toolbar button {{ margin-right: 6px; }}
  </style>
  <script src=\"{rel_vis_js}\"></script>
  <script>
    const nodes = {json.dumps(nodes, ensure_ascii=False)};
    const edges = {json.dumps(edges, ensure_ascii=False)};

    let network;

    function init() {{
      const container = document.getElementById('mynetwork');
      const data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
      const options = {{
        autoResize: true,
        physics: {{
          enabled: true,
          solver: 'forceAtlas2Based',
          stabilization: {{ iterations: 200 }}
        }},
        interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true, keyboard: true }},
        nodes: {{ shape: 'dot', size: 16, font: {{ size: 14 }}, borderWidth: 1 }},
        edges: {{ arrows: 'to', smooth: {{ type: 'dynamic' }}, color: {{ opacity: 0.6 }} }}
      }};
      network = new vis.Network(container, data, options);
    }}

    function fitView() {{
      if (network) {{ network.fit({{ animation: true }}); }}
    }}

    function togglePhysics() {{
      if (!network) return;
      const p = network.physics.options.enabled;
      network.setOptions({{ physics: {{ enabled: !p }} }});
    }}

    window.addEventListener('load', init);
  </script>
  </head>
<body>
  <div class=\"toolbar\">
    <button onclick=\"fitView()\">适配视图</button>
    <button onclick=\"togglePhysics()\">切换物理布局</button>
  </div>
  <div id=\"mynetwork\"></div>
</body>
</html>
"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"可视化已生成: {html_path}")
    return html_path


def generate_visualization_from_ground_truth(gt_file_path, attack_type, schema_name=None, output_dir=None):
    """
    从ground truth大JSON中选择一个schema并生成可视化HTML。

    Args:
        gt_file_path (str): ground truth JSON路径（extracted_data/event_graphs_*.json）
        attack_type (str): 顶层攻击类型键，例如 'suicide_ied'
        schema_name (str|None): 指定schema_name；为None时随机选择
        output_dir (str|None): 输出目录，默认<project_root>/visualize

    Returns:
        str: 生成的HTML文件路径
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'visualize')

    _ensure_output_dir(output_dir)

    with open(gt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if attack_type not in data or not isinstance(data[attack_type], list) or len(data[attack_type]) == 0:
        raise ValueError(f"ground truth中未找到攻击类型或为空: {attack_type}")

    schemas = data[attack_type]

    # 选择schema
    selected_schema = None
    if schema_name:
        for schema in schemas:
            if schema.get('schema_name') == schema_name:
                selected_schema = schema
                break
        if selected_schema is None:
            raise ValueError(f"未找到指定schema_name: {schema_name}")
    else:
        selected_schema = random.choice(schemas)

    # 将schema转为通用的graph_json结构
    nodes = []
    edges = []

    for ev in selected_schema.get('events', []):
        ev_id = ev.get('event_id')
        ev_type = ev.get('event_type', '')  # 形如 A.B.C
        ev_name = ev.get('event_name', '')

        t1 = t2 = t3 = ''
        if ev_type:
            parts = ev_type.split('.')
            if len(parts) > 0:
                t1 = parts[0]
            if len(parts) > 1:
                t2 = parts[1]
            if len(parts) > 2:
                t3 = parts[2]

        node = {
            'id': ev_id,
            'event_type': t1,
            'event_subtype': t2,
            'event_sub_subtype': t3
        }
        if ev_name:
            node['description'] = ev_name
        nodes.append(node)

    for rel in selected_schema.get('temporal_relations', []):
        b = rel.get('before')
        a = rel.get('after')
        if b and a:
            edges.append({
                'source': b,
                'target': a,
                'relation': 'before'
            })

    graph_json = {'nodes': nodes, 'edges': edges}
    vis_nodes, vis_edges = _convert_to_vis_data(graph_json)

    # 生成HTML
    rel_vis_js = os.path.relpath(
        os.path.join(PROJECT_ROOT, 'lib', 'vis-9.1.2', 'vis-network.min.js'),
        output_dir
    )
    rel_vis_css = os.path.relpath(
        os.path.join(PROJECT_ROOT, 'lib', 'vis-9.1.2', 'vis-network.css'),
        output_dir
    )

    base = f"gt_{attack_type}_{selected_schema.get('schema_name', 'random')}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_name = f"{base}_{timestamp}.html"
    html_path = os.path.join(output_dir, html_name)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Graph Visualization - {base}</title>
  <link rel=\"stylesheet\" href=\"{rel_vis_css}\" />
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; }}
    #mynetwork {{ width: 100%; height: 100vh; border: 1px solid #ddd; }}
    .toolbar {{ position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); z-index: 10; }}
    .toolbar button {{ margin-right: 6px; }}
  </style>
  <script src=\"{rel_vis_js}\"></script>
  <script>
    const nodes = {json.dumps(vis_nodes, ensure_ascii=False)};
    const edges = {json.dumps(vis_edges, ensure_ascii=False)};

    let network;

    function init() {{
      const container = document.getElementById('mynetwork');
      const data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
      const options = {{
        autoResize: true,
        physics: {{
          enabled: true,
          solver: 'forceAtlas2Based',
          stabilization: {{ iterations: 200 }}
        }},
        interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true, keyboard: true }},
        nodes: {{ shape: 'dot', size: 16, font: {{ size: 14 }}, borderWidth: 1 }},
        edges: {{ arrows: 'to', smooth: {{ type: 'dynamic' }}, color: {{ opacity: 0.6 }} }}
      }};
      network = new vis.Network(container, data, options);
    }}

    function fitView() {{
      if (network) {{ network.fit({{ animation: true }}); }}
    }}

    function togglePhysics() {{
      if (!network) return;
      const p = network.physics.options.enabled;
      network.setOptions({{ physics: {{ enabled: !p }} }});
    }}

    window.addEventListener('load', init);
  </script>
  </head>
<body>
  <div class=\"toolbar\">
    <button onclick=\"fitView()\">适配视图</button>
    <button onclick=\"togglePhysics()\">切换物理布局</button>
  </div>
  <div id=\"mynetwork\"></div>
</body>
</html>
"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"可视化已生成: {html_path}")
    return html_path


if __name__ == '__main__':
    """generate_visualization_from_ground_truth(
        gt_file_path=os.path.join(PROJECT_ROOT, 'dataset', 'processedData', 'extracted_data', 'event_graphs_train.json'),
        attack_type='suicide_ied'
    )"""
    for i in range(26, 29):
        generate_visualization(f"/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/model2/suicide_ied/graph_{i}.json")
    
    # generate_visualization(f"/Users/andrewlee/Nutstore Files/我的坚果云/情报杂志/code/result/model2/suicide_ied/merged_graph_suicide_ied.json")