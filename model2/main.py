# -*- coding: utf-8 -*-
"""
基于意义段的迭代构图算法 - 主程序
按照algorithm.md中第二种算法的步骤执行
"""
import os
import sys
import shutil
import argparse

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_service import LLMService
from utils import Logger
from config import ZHIPU_API_KEY, ATTACK_TYPES, get_graphs_path
from data_loader import DataLoader
from graph_builder import GraphBuilder
from mergeGraph import GraphMerger


def main(attack_type=None, api_key=None):
    """
    主函数：执行基于意义段的迭代构图算法
    
    Args:
        attack_type (str): 攻击类型，如果为None则使用默认的suicide_ied
        api_key (str): API密钥，如果为None则使用配置文件中的密钥
    """
    # 初始化日志
    logger = Logger()
    logger.printLog("=" * 80)
    logger.printLog("开始执行基于意义段的迭代构图算法 (Model2)")
    logger.printLog("=" * 80)
    
    # 1. 选择攻击类型
    if attack_type is None:
        attack_type = 'suicide_ied'
    
    if attack_type not in ATTACK_TYPES:
        logger.printLog(f"警告: 攻击类型 {attack_type} 不在预定义列表中")
    
    logger.printLog(f"步骤1: 选择攻击类型 - {attack_type}")
    
    # 2. 初始化服务
    logger.printLog("步骤2: 初始化服务...")
    
    # 使用传入的API密钥或配置文件中的密钥
    current_api_key = api_key if api_key else ZHIPU_API_KEY
    if current_api_key == "your_api_key_here":
        logger.printLog("错误: 请在config.py中配置ZHIPU_API_KEY或通过命令行参数传入")
        return
    
    llm_service = LLMService(api_key=current_api_key)
    data_loader = DataLoader(llm_service=llm_service)  # 传递LLM服务，用于智能采样
    
    # 3. 加载文本数据
    logger.printLog(f"步骤3: 加载攻击类型 {attack_type} 的文本数据...")
    texts = data_loader.load_texts_for_attack_type(attack_type)
    
    if not texts:
        logger.printLog("错误: 未找到任何文本数据")
        return
    
    logger.printLog(f"共加载 {len(texts)} 个文本文件")
    
    # 4. 清空并创建结果目录
    logger.printLog("步骤4: 准备结果目录...")
    graphs_path = get_graphs_path(attack_type)
    if os.path.exists(graphs_path):
        shutil.rmtree(graphs_path)
    os.makedirs(graphs_path, exist_ok=True)
    logger.printLog(f"结果将保存到: {graphs_path}")
    
    # 5. 为每个文本构建事件图（使用基于意义段的迭代方法）
    logger.printLog("步骤5: 开始为每个文本构建事件图（基于意义段的迭代方法）...")
    
    event_ontology = data_loader.get_event_types_description()
    event_types = data_loader.get_event_types()
    graph_builder = GraphBuilder(llm_service, event_ontology, event_types)
    
    graphs = []
    for i, text_data in enumerate(texts, 1):
        logger.printLog(f"\n处理文本 {i}/{len(texts)}: {os.path.basename(text_data['path'])}")
        
        try:
            # 构建图（使用迭代方法）
            graph = graph_builder.build_graph_from_text(
                text_data['content'],
                f"text_{i}"
            )
            
            if graph and graph.number_of_nodes() > 0:
                graphs.append(graph)
                
                # 保存单个图
                graph_path = os.path.join(graphs_path, f"graph_{i}.json")
                graph_builder.save_graph_to_json(graph, graph_path)
                logger.printLog(f"  - 图已保存: {graph_path}")
            else:
                logger.printLog(f"  - 跳过: 未生成有效的图")
                
        except Exception as e:
            logger.printLog(f"  - 错误: {str(e)}")
            import traceback
            logger.printLog(traceback.format_exc())
    
    logger.printLog(f"\n成功构建 {len(graphs)} 个事件图")
    
    # 打印验证报告
    logger.printLog("\n" + "="*80)
    logger.printLog("事件类型验证总报告")
    logger.printLog("="*80)
    graph_builder.print_validation_report()
    
    # 打印低置信度分类报告
    logger.printLog("\n")
    graph_builder.print_low_confidence_report()
    
    # 6. 融合所有图
    logger.printLog("\n步骤6: 融合所有事件图为骨架图...")
    
    if not graphs:
        logger.printLog("错误: 没有可用的图进行融合")
        return
    
    graph_merger = GraphMerger()
    merged_graph = graph_merger.merge_graphs(graphs)
    
    # 保存融合后的图
    merged_graph_path = os.path.join(graphs_path, f"merged_graph_{attack_type}.json")
    graph_merger.save_merged_graph(merged_graph, merged_graph_path)
    
    # 7. 完成
    logger.printLog("\n" + "=" * 80)
    logger.printLog("算法执行完成!")
    logger.printLog(f"算法类型: 基于意义段的迭代构图算法 (Model2)")
    logger.printLog(f"- 处理文本数: {len(texts)}")
    logger.printLog(f"- 成功构建图数: {len(graphs)}")
    logger.printLog(f"- 融合图节点数: {merged_graph.number_of_nodes()}")
    logger.printLog(f"- 融合图边数: {merged_graph.number_of_edges()}")
    logger.printLog(f"- 结果保存路径: {graphs_path}")
    logger.printLog("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基于意义段的迭代构图算法')
    parser.add_argument('--attack_type', type=str, default='suicide_ied',
                        help='攻击类型 (默认: suicide_ied)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='智谱AI API密钥')
    
    args = parser.parse_args()
    
    main(attack_type=args.attack_type, api_key=args.api_key)

