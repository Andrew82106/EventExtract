# -*- coding: utf-8 -*-
"""
配置文件
"""
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API密钥配置
ZHIPU_API_KEY = "6c735ef4e888bbfa5f66175870a62901.1c1jin8KHtZqNW90"  # 请在这里填入你的智谱AI API密钥

# 数据路径配置
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
EXTRACTED_TEXTS_PATH = os.path.join(DATASET_ROOT, 'processedData', 'extracted_texts')
ONTOLOGY_PATH = os.path.join(DATASET_ROOT, 'kairos_ontology.xlsx')

# 结果保存路径
RESULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
MODEL_NAME = 'model1'
MODEL_RESULT_PATH = os.path.join(RESULT_ROOT, MODEL_NAME)


# 该函数根据攻击类型生成图保存路径
def get_graphs_path(attack_type):
    """获取指定攻击类型的图保存路径"""
    return os.path.join(MODEL_RESULT_PATH, attack_type)


# 采样配置
MAX_TEXT_SAMPLES = 20

# 缓存配置
CACHE_ROOT = os.path.join(RESULT_ROOT, 'cache')
SAMPLING_CACHE_PATH = os.path.join(CACHE_ROOT, 'text_relevance_cache.json')
SAMPLING_RESULTS_CACHE = os.path.join(CACHE_ROOT, 'sampling_results.json')  # 采样结果缓存
USE_SAMPLING_CACHE = False  # 是否启用智能采样缓存

# 断点续传配置
ENABLE_RESUME = False  # 是否启用断点续传（跳过已处理的文本）
FAILED_TEXTS_LOG = os.path.join(CACHE_ROOT, 'failed_texts.json')  # 失败文本记录

# LLM重试配置
MAX_RETRIES = 3  # LLM调用失败时的最大重试次数
RETRY_DELAY = 2  # 重试延迟（秒）

# 攻击类型列表
ATTACK_TYPES = [
    'suicide_ied',
    'backpack_ied',
    'road_ied',
    'wiki_drone_strikes',
    'wiki_ied_bombings',
    'wiki_mass_car_bombings'
]
