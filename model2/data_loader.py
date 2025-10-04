# -*- coding: utf-8 -*-
"""
数据加载模块
负责从文件系统中加载文本和本体数据
"""
import os
import random
import pandas as pd
from config import EXTRACTED_TEXTS_PATH, ONTOLOGY_PATH, MAX_TEXT_SAMPLES


class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        """初始化数据加载器"""
        self.ontology_df = None
        self.event_types = []
        self._load_ontology()
    
    def get_event_types(self):
        """
        获取事件类型列表，用于验证器
        
        Returns:
            list: 事件类型列表
        """
        return self.event_types
    
    def _load_ontology(self):
        """加载事件本体数据"""
        try:
            self.ontology_df = pd.read_excel(ONTOLOGY_PATH)
            # 提取事件类型信息
            self.event_types = []
            for _, row in self.ontology_df.iterrows():
                event_info = {
                    'type': row['Type'],
                    'subtype': row['Subtype'] if pd.notna(row['Subtype']) else '',
                    'sub_subtype': row['Sub-subtype'] if pd.notna(row['Sub-subtype']) else '',
                    'definition': row['Definition'] if pd.notna(row['Definition']) else '',
                    'template': row['Template'] if pd.notna(row['Template']) else ''
                }
                self.event_types.append(event_info)
            print(f"成功加载 {len(self.event_types)} 个事件类型")
        except Exception as e:
            print(f"加载本体数据失败: {str(e)}")
            self.event_types = []
    
    def get_event_types_description(self):
        """
        获取事件类型的文本描述，用于提示词
        
        Returns:
            str: 事件类型的描述文本
        """
        descriptions = []
        for event in self.event_types:
            desc = f"- {event['type']}"
            if event['subtype']:
                desc += f" > {event['subtype']}"
            if event['sub_subtype']:
                desc += f" > {event['sub_subtype']}"
            if event['definition']:
                desc += f": {event['definition']}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def load_texts_for_attack_type(self, attack_type):
        """
        加载指定攻击类型的所有文本
        
        Args:
            attack_type (str): 攻击类型名称
            
        Returns:
            list: 文本列表，每个元素是一个字典 {'path': 文件路径, 'content': 文本内容}
        """
        attack_type_path = os.path.join(EXTRACTED_TEXTS_PATH, attack_type)
        
        if not os.path.exists(attack_type_path):
            print(f"攻击类型路径不存在: {attack_type_path}")
            return []
        
        # 收集所有文本文件
        text_files = []
        for root, dirs, files in os.walk(attack_type_path):
            for file in files:
                if file.endswith('.txt') and file not in ['extraction_statistics.txt']:
                    file_path = os.path.join(root, file)
                    text_files.append(file_path)
        
        print(f"找到 {len(text_files)} 个文本文件")
        
        # 如果文本数量大于最大采样数，进行随机采样
        if len(text_files) > MAX_TEXT_SAMPLES:
            text_files = random.sample(text_files, MAX_TEXT_SAMPLES)
            print(f"采样后保留 {len(text_files)} 个文本文件")
        
        # 读取文本内容
        texts = []
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:  # 只添加非空文本
                        texts.append({
                            'path': file_path,
                            'content': content
                        })
            except Exception as e:
                print(f"读取文件失败 {file_path}: {str(e)}")
        
        print(f"成功加载 {len(texts)} 个文本")
        return texts

