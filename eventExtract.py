import json
from llm_service import LLMService
from prompt_templates import PromptTemplates


class DynamicEventGraphBuilder:
    def __init__(self, api_key):
        """
        初始化动态事件图构建器
        
        Args:
            api_key (str): 智谱AI的API密钥
        """
        self.llm_service = LLMService(api_key)
        self.templates = PromptTemplates.get_templates()

    def extract_entities(self, text):
        """阶段1：动态实体识别（不限制类型）"""
        return self.llm_service.extract_entities(text, self.templates["ner"])

    def extract_events(self, text):
        """阶段2：事件识别与动态论元抽取"""
        return self.llm_service.extract_events(text, self.templates["event_extraction"])

    def resolve_coreference(self, text):
        """阶段3：共指消解"""
        return self.llm_service.resolve_coreference(text, self.templates["coreference"])

    def extract_entity_relations(self, text):
        """阶段4：动态实体关系抽取"""
        return self.llm_service.extract_entity_relations(text, self.templates["entity_relation"])

    def extract_temporal_relations(self, text):
        """阶段5：事件时间关系推理"""
        return self.llm_service.extract_temporal_relations(text, self.templates["temporal_relation"])

    def build_event_graph(self, text):
        """整合所有阶段，生成动态事件图"""
        print("=== 开始构建事件图（动态实体类型） ===")

        # 1. 实体识别（动态类型）
        entities = self.extract_entities(text)
        print(f"\n1. 动态实体识别: {json.dumps(entities, ensure_ascii=False, indent=2)}")

        # 2. 事件识别与论元抽取
        events = self.extract_events(text)
        print(f"\n2. 事件与论元: {json.dumps(events, ensure_ascii=False, indent=2)}")

        # 3. 共指消解
        coreferences = self.resolve_coreference(text)
        print(f"\n3. 共指消解结果: {json.dumps(coreferences, ensure_ascii=False, indent=2)}")

        # 4. 实体关系抽取
        entity_relations = self.extract_entity_relations(text)
        print(f"\n4. 实体关系: {json.dumps(entity_relations, ensure_ascii=False, indent=2)}")

        # 5. 时间关系推理
        temporal_relations = self.extract_temporal_relations(text)
        print(f"\n5. 事件时间关系: {json.dumps(temporal_relations, ensure_ascii=False, indent=2)}")

        # 组装事件图
        event_graph = {
            "nodes": {
                "entities": entities,  # 动态类型实体
                "events": events  # 动态类型事件
            },
            "edges": {
                "entity_relations": entity_relations,  # 实体-实体边
                "temporal_relations": temporal_relations,  # 事件-事件时间边
                "event_arguments": [  # 事件-实体论元边
                    {
                        "事件触发词": event["触发词"],
                        "实体": arg_value,
                        "角色": arg_key
                    }
                    for event in events
                    for arg_key, arg_value in event["论元"].items()
                ]
            }
        }

        print("\n=== 动态事件图构建完成 ===")
        return event_graph


# -*- coding: utf-8 -*-
import os
import json
import webbrowser
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from pyvis.network import Network


# 如果你的 DynamicEventGraphBuilder 定义在别的模块，改成 from yourmodule import DynamicEventGraphBuilder
# 这里假设它和可视化代码在同一文件中，或已经从 eventExtract 导入。
# from eventExtract import DynamicEventGraphBuilder

# ----------------- 辅助：自动选择中文字体 -----------------
def find_chinese_font():
    # 按优先级列出一些常见中文字体（macOS/Linux/Windows）
    candidates = [
        "PingFang SC", "Songti SC", "Heiti SC", "SimHei", "Noto Sans CJK SC",
        "Microsoft YaHei", "WenQuanYi Zen Hei", "AR PL UKai CN"
    ]
    sys_fonts = {os.path.basename(f): f for f in font_manager.findSystemFonts(fontpaths=None, fontext='ttf')}
    # 尝试用 font_manager 找到最合适的字体
    for cand in candidates:
        # 用 FontProperties 试探是否能找到
        prop = FontProperties(fname=None, family=cand)
        try:
            fpath = font_manager.findfont(prop, fallback_to_default=False)
            if fpath and os.path.exists(fpath):
                return fpath
        except Exception:
            continue
    # 如果上面没找到，退回到 matplotlib 的默认但尝试常见名字
    for cand in candidates:
        try:
            return font_manager.findfont(FontProperties(family=cand))
        except Exception:
            continue
    # 直接返回 matplotlib 默认字体路径
    return font_manager.findfont(font_manager.FontProperties())


# 设置 matplotlib 使用中文字体（全局）
chinese_font_path = find_chinese_font()
chinese_prop = FontProperties(fname=chinese_font_path)
matplotlib.rcParams['font.family'] = chinese_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False  # 保证负号显示


# ----------------- 构建 networkx 图的函数（和之前逻辑一致） -----------------
def build_networkx_graph(event_graph):
    G = nx.DiGraph()
    entity_map = {}
    # 添加实体节点
    for i, ent in enumerate(event_graph["nodes"]["entities"]):
        node_id = f"ent_{i}"
        label = ent.get("实体文本") or ent.get("text") or str(ent)
        subtype = ent.get("类型") or "实体"
        G.add_node(node_id, label=label, ntype="entity", subtype=subtype)
        entity_map.setdefault(label, []).append(node_id)
    # 添加事件节点
    event_map = {}
    for j, evt in enumerate(event_graph["nodes"]["events"]):
        node_id = f"evt_{j}"
        trigger = evt.get("触发词") or f"事件{j}"
        etype = evt.get("事件类型") or "事件"
        label = f"{trigger}\n({etype})"
        G.add_node(node_id, label=label, ntype="event", subtype=etype, trigger=trigger)
        event_map.setdefault(trigger, []).append(node_id)
    # 实体关系边
    for rel in event_graph.get("edges", {}).get("entity_relations", []):
        a = rel.get("实体1");
        b = rel.get("实体2");
        rtype = rel.get("关系类型") or "关系"
        if a in entity_map and b in entity_map:
            for na in entity_map[a]:
                for nb in entity_map[b]:
                    G.add_edge(na, nb, label=rtype, etype="entity_relation")
    # 时间关系
    for trel in event_graph.get("edges", {}).get("temporal_relations", []):
        t1 = trel.get("事件1");
        t2 = trel.get("事件2");
        r = trel.get("时间关系")
        if t1 in event_map and t2 in event_map:
            for n1 in event_map[t1]:
                for n2 in event_map[t2]:
                    G.add_edge(n1, n2, label=r, etype="temporal")
    # 事件-论元边
    for arg in event_graph.get("edges", {}).get("event_arguments", []):
        trigger = arg.get("事件触发词");
        ent_text = arg.get("实体");
        role = arg.get("角色")
        if trigger in event_map and ent_text in entity_map:
            for ne in event_map[trigger]:
                for nt in entity_map[ent_text]:
                    G.add_edge(ne, nt, label=role, etype="argument")
    return G


# ----------------- matplotlib 静态可视化（中文支持） -----------------
def visualize_matplotlib(G, figsize=(12, 8), save_path=None):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "entity"]
    event_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "event"]
    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_shape="o", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=event_nodes, node_shape="s", node_size=1200)
    # 用中文字体的 FontProperties 绘制标签
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_family=chinese_prop.get_name())
    nx.draw_networkx_edges(G, pos, arrows=True)
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    # draw_networkx_edge_labels 没有直接 fontproperties 参数，但可以设置 font_family
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family=chinese_prop.get_name())
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"已保存静态图片到: {save_path}")
    plt.show()


# ----------------- pyvis 交互式可视化（稳妥写法） -----------------
def visualize_pyvis(G, output_html="event_graph.html", height="800px", width="100%", open_in_browser=True):
    # 检查 jinja2（pyvis 生成 html 时依赖）
    try:
        import jinja2  # noqa: F401
    except Exception:
        raise RuntimeError("pyvis 在生成 HTML 时需要 jinja2。请先运行：pip install jinja2")

    net = Network(height=height, width=width, directed=True)
    for n, d in G.nodes(data=True):
        label = d.get("label", n)
        title = f"类型: {d.get('ntype')}\n子类型: {d.get('subtype')}"
        shape = "square" if d.get("ntype") == "event" else "dot"
        net.add_node(n, label=label, title=title, shape=shape)
    for u, v, d in G.edges(data=True):
        title = d.get("label", "")
        # pyvis 在 label/ title 上可以显示中文；若出现编码问题可在 html 中做 meta utf-8（pyvis 默认支持）
        net.add_edge(u, v, title=title, label=title)
    # 用更稳妥的写法：先试 net.write_html，如果失败再尝试 net.show
    try:
        net.write_html(output_html)
    except Exception as e:
        print("net.write_html 失败，尝试 net.show，错误：", e)
        try:
            net.show(output_html)
        except Exception as e2:
            raise RuntimeError(f"pyvis 输出 HTML 失败：{e2}")
    print(f"交互式图已生成: {output_html}")
    if open_in_browser:
        webbrowser.open("file://" + os.path.abspath(output_html))


# ----------------- 主流程：用你原来的 builder 生成 event_graph（实时） -----------------
if __name__ == "__main__":
    # 请替换为你的 API_KEY（如果 builder 需要）
    API_KEY = "6c735ef4e888bbfa5f66175870a62901.1c1jin8KHtZqNW90"
    SAMPLE_TEXT = "GhostWriter 至少从 2017 年开始活跃，此前观察到它冒充立陶宛、拉脱维亚和波兰的记者，向当地观众传播虚假信息和反北约言论。近日，波兰政府警告称，GhostWriter 近期一直关注波兰，试图侵入电子邮件账户以收集敏感信息，并控制社交媒体账户以传播虚假信息。俄罗斯黑客攻击的目标不仅包括公共领域和国家组织，还涉及战略能源和军备供应商以及其他重要实体。其公告中的一个案例是“GhostWriter”组织建立了冒充政府域名 gov.pl 的网站，向据称得到欧洲资金支持的波兰居民宣传虚假经济补偿。单击嵌入式按钮“以了解有关该程序的更多信息”会将受害者带到一个钓鱼网站，进而窃取其个人信息或进行进一步的恶意活动。此外，针对波兰议会网站 ('sejm.gov.pl') 的 DDoS（分布式拒绝服务）攻击，被归因于亲俄罗斯的所谓黑客活动主义者 NoName057(16)。该攻击发生在议会通过决议承认俄罗斯是支持恐怖主义的国家后的第二天，导致公众无法访问该网站，严重影响了议会的正常运作和公众信息的获取。攻击行动或事件情报01显示，攻击者对哥伦比亚银行发起 SQL 注入攻击，以部署 BitRAT 恶意软件，从而实现对目标系统的远程控制和数据窃取。"


    builder = DynamicEventGraphBuilder(api_key=API_KEY)

    # 这里使用原始流程：所有阶段调用由 builder.build_event_graph 完成
    # 它会调用 extract_entities / extract_events / resolve_coreference / extract_entity_relations / extract_temporal_relations
    print("开始调用原始解析流程，注意：这会调用你的 LLM API 并返回解析结果...")
    event_graph = builder.build_event_graph(SAMPLE_TEXT)  # <-- 使用你原来的逻辑生成图

    # 如果 build_event_graph 返回 None 或出错，给出保护
    if not event_graph:
        raise RuntimeError("事件图生成失败，请检查 LLM 接口调用和返回值格式。")

    # 转网络并可视化
    G = build_networkx_graph(event_graph)
    visualize_matplotlib(G, save_path="event_graph.png")
    # 生成交互式 HTML（会自动打开）
    visualize_pyvis(G, output_html="event_graph.html")