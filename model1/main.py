# -*- coding: utf-8 -*-
"""
简单询问式构图算法 - 主程序
按照algorithm.md中描述的步骤执行
"""
import os
import sys
import shutil
import argparse
import time

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_service import LLMService
from utils import Logger
from config import ZHIPU_API_KEY, ATTACK_TYPES, get_graphs_path, MAX_RETRIES, RETRY_DELAY, ENABLE_RESUME, FAILED_TEXTS_LOG, CACHE_ROOT
from data_loader import DataLoader
from graph_builder import GraphBuilder
from mergeGraph import GraphMerger
import json


def main(attack_type=None, api_key=None):
    """
    主函数：执行简单询问式构图算法
    
    Args:
        attack_type (str): 攻击类型，如果为None则使用默认的suicide_ied
        api_key (str): API密钥，如果为None则使用配置文件中的密钥
    """
    # 记录总开始时间
    total_start_time = time.time()
    time_records = {}
    
    # 初始化日志
    logger = Logger()
    logger.printLog("=" * 80)
    logger.printLog("开始执行简单询问式构图算法 (Model1)")
    logger.printLog("=" * 80)
    
    # 1. 选择攻击类型
    if attack_type is None:
        attack_type = 'suicide_ied'
    
    if attack_type not in ATTACK_TYPES:
        logger.printLog(f"警告: 攻击类型 {attack_type} 不在预定义列表中")
    
    logger.printLog(f"步骤1: 选择攻击类型 - {attack_type}")
    
    # 2. 初始化服务
    step_start = time.time()
    logger.printLog("步骤2: 初始化服务...")
    
    # 使用传入的API密钥或配置文件中的密钥
    current_api_key = api_key if api_key else ZHIPU_API_KEY
    if current_api_key == "your_api_key_here":
        logger.printLog("错误: 请在config.py中配置ZHIPU_API_KEY或通过命令行参数传入")
        return
    
    llm_service = LLMService(api_key=current_api_key, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY)
    data_loader = DataLoader(llm_service=llm_service)  # 传递LLM服务，用于智能采样
    time_records['初始化服务'] = time.time() - step_start
    logger.printLog(f"  LLM重试配置: 最大重试{MAX_RETRIES}次, 延迟{RETRY_DELAY}秒")
    
    # 3. 加载文本数据
    step_start = time.time()
    logger.printLog(f"步骤3: 加载攻击类型 {attack_type} 的文本数据...")
    texts = data_loader.load_texts_for_attack_type(attack_type)
    time_records['加载文本数据'] = time.time() - step_start
    
    if not texts:
        logger.printLog("错误: 未找到任何文本数据")
        return
    
    logger.printLog(f"共加载 {len(texts)} 个文本文件")
    
    # 4. 准备结果目录和断点续传
    logger.printLog("步骤4: 准备结果目录...")
    graphs_path = get_graphs_path(attack_type)
    
    # 检查已处理的文本
    processed_texts = set()
    failed_texts_data = {}
    
    if ENABLE_RESUME and os.path.exists(graphs_path):
        logger.printLog(f"  启用断点续传模式")
        # 扫描已有的图文件
        for filename in os.listdir(graphs_path):
            if filename.startswith('graph_') and filename.endswith('.json'):
                processed_texts.add(filename)
        logger.printLog(f"  发现 {len(processed_texts)} 个已处理的图文件")
        
        # 加载失败记录
        if os.path.exists(FAILED_TEXTS_LOG):
            try:
                with open(FAILED_TEXTS_LOG, 'r', encoding='utf-8') as f:
                    failed_texts_data = json.load(f)
                attack_failed = failed_texts_data.get(attack_type, [])
                logger.printLog(f"  发现 {len(attack_failed)} 个之前失败的文本记录")
            except:
                pass
    else:
        if os.path.exists(graphs_path):
            shutil.rmtree(graphs_path)
        os.makedirs(graphs_path, exist_ok=True)
        logger.printLog(f"  清空旧结果，重新开始")
    
    os.makedirs(graphs_path, exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    logger.printLog(f"结果将保存到: {graphs_path}")
    
    # 5. 为每个文本构建事件图
    step_start = time.time()
    logger.printLog("步骤5: 开始为每个文本构建事件图...")
    
    event_ontology = data_loader.get_event_types_description()
    event_types = data_loader.get_event_types()
    graph_builder = GraphBuilder(llm_service, event_ontology, event_types)
    
    graphs = []
    failed_texts = []
    skipped_count = 0
    
    for i, text_data in enumerate(texts, 1):
        graph_filename = f"graph_{i}.json"
        graph_path = os.path.join(graphs_path, graph_filename)
        
        # 检查是否已处理
        if ENABLE_RESUME and graph_filename in processed_texts:
            skipped_count += 1
            logger.printLog(f"\n跳过文本 {i}/{len(texts)} (已处理): {os.path.basename(text_data['path'])}")
            # 加载已有的图
            try:
                graph = graph_builder.load_graph_from_json(graph_path)
                if graph and graph.number_of_nodes() > 0:
                    graphs.append(graph)
            except:
                logger.printLog(f"  ! 加载已有图失败，将重新处理")
            continue
        
        logger.printLog(f"\n处理文本 {i}/{len(texts)}: {os.path.basename(text_data['path'])}")
        
        try:
            # 构建图
            graph = graph_builder.build_graph_from_text(
                text_data['content'],
                f"text_{i}"
            )
            
            if graph and graph.number_of_nodes() > 0:
                graphs.append(graph)
                
                # 保存单个图
                graph_builder.save_graph_to_json(graph, graph_path)
                logger.printLog(f"  ✓ 图已保存: {graph_path}")
            else:
                logger.printLog(f"  - 跳过: 未生成有效的图")
                # 记录失败
                failed_texts.append({
                    'index': i,
                    'path': text_data['path'],
                    'reason': '未生成有效的图'
                })
                
        except Exception as e:
            logger.printLog(f"  ✗ 错误: {str(e)}")
            # 记录失败
            failed_texts.append({
                'index': i,
                'path': text_data['path'],
                'reason': str(e)
            })
            import traceback
            logger.printLog(traceback.format_exc())
    
    # 保存失败记录
    if failed_texts:
        if not failed_texts_data:
            failed_texts_data = {}
        failed_texts_data[attack_type] = failed_texts
        try:
            with open(FAILED_TEXTS_LOG, 'w', encoding='utf-8') as f:
                json.dump(failed_texts_data, f, ensure_ascii=False, indent=2)
            logger.printLog(f"\n失败文本记录已保存到: {FAILED_TEXTS_LOG}")
        except Exception as e:
            logger.printLog(f"\n保存失败记录出错: {str(e)}")
    
    if skipped_count > 0:
        logger.printLog(f"\n断点续传: 跳过了 {skipped_count} 个已处理的文本")
    
    logger.printLog(f"\n成功构建 {len(graphs)} 个事件图")
    time_records['构建事件图'] = time.time() - step_start
    
    # 打印验证报告
    logger.printLog("\n" + "="*80)
    logger.printLog("事件类型验证总报告")
    logger.printLog("="*80)
    graph_builder.print_validation_report()
    
    # 6. 融合所有图
    step_start = time.time()
    logger.printLog("\n步骤6: 融合所有事件图为骨架图...")
    
    if not graphs:
        logger.printLog("错误: 没有可用的图进行融合")
        return
    
    graph_merger = GraphMerger()
    merged_graph = graph_merger.merge_graphs(graphs)
    
    # 保存融合后的图
    merged_graph_path = os.path.join(graphs_path, f"merged_graph_{attack_type}.json")
    graph_merger.save_merged_graph(merged_graph, merged_graph_path)
    time_records['融合图'] = time.time() - step_start
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 7. 完成
    logger.printLog("\n" + "=" * 80)
    logger.printLog("算法执行完成!")
    logger.printLog(f"- 处理文本数: {len(texts)}")
    if skipped_count > 0:
        logger.printLog(f"  其中跳过（已处理）: {skipped_count}")
    if failed_texts:
        logger.printLog(f"  其中失败: {len(failed_texts)}")
    logger.printLog(f"- 成功构建图数: {len(graphs)}")
    logger.printLog(f"- 融合图节点数: {merged_graph.number_of_nodes()}")
    logger.printLog(f"- 融合图边数: {merged_graph.number_of_edges()}")
    logger.printLog(f"- 结果保存路径: {graphs_path}")
    logger.printLog("\n时间统计:")
    for step_name, step_time in time_records.items():
        logger.printLog(f"  - {step_name}: {step_time:.2f}秒 ({step_time/total_time*100:.1f}%)")
    logger.printLog(f"  总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    # 打印LLM调用统计
    llm_service.print_call_statistics()
    
    logger.printLog("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简单询问式构图算法')
    parser.add_argument('--attack_type', type=str, default='suicide_ied',
                        help='攻击类型 (默认: suicide_ied)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='智谱AI API密钥')
    
    args = parser.parse_args()
    
    main(attack_type=args.attack_type, api_key=args.api_key)

