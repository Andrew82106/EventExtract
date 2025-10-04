# -*- coding: utf-8 -*-
"""
数据加载模块
负责从文件系统中加载文本和本体数据
"""
import os
import time
import random
import pandas as pd
from config import EXTRACTED_TEXTS_PATH, ONTOLOGY_PATH, MAX_TEXT_SAMPLES


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, llm_service=None):
        """
        初始化数据加载器
        
        Args:
            llm_service: LLM服务实例，用于智能文本采样（可选）
        """
        self.ontology_df = None
        self.event_types = []
        self.llm_service = llm_service
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
    
    def _judge_text_relevance(self, text_content, attack_type):
        """
        使用LLM判断文本是否讲述了与攻击类型相关的事件故事
        
        Args:
            text_content (str): 文本内容
            attack_type (str): 攻击类型名称
            
        Returns:
            tuple: (is_relevant, confidence) 是否相关和置信度
        """
        if not self.llm_service:
            return False, 0.0
        
        from prompts import JUDGE_TEXT_RELEVANCE_PROMPT
        
        try:
            # 取文本前100个字符作为预览
            text_preview = text_content[:100]
            
            prompt = JUDGE_TEXT_RELEVANCE_PROMPT.format(
                attack_type=attack_type,
                text_preview=text_preview
            )
            result = self.llm_service.call_api(prompt)
            
            if isinstance(result, dict) and 'is_relevant' in result:
                is_relevant = result.get('is_relevant', False)
                confidence = result.get('confidence', 0.5)
                return is_relevant, confidence
            else:
                print(f"  ! 判断失败，返回格式不正确")
                return False, 0.0
        except Exception as e:
            print(f"  ! 判断异常: {str(e)}")
            return False, 0.0
    
    def load_texts_for_attack_type(self, attack_type, use_smart_sampling=True):
        """
        加载指定攻击类型的所有文本（支持基于LLM判断的智能采样）
        
        Args:
            attack_type (str): 攻击类型名称
            use_smart_sampling (bool): 是否使用基于LLM判断的智能采样
            
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
        
        # 如果文本数量小于等于最大采样数，直接加载全部
        if len(text_files) <= MAX_TEXT_SAMPLES:
            print(f"文本数量未超过限制，加载全部文本")
            selected_files = text_files
        else:
            # 需要采样
            if use_smart_sampling and self.llm_service:
                print(f"使用基于LLM判断的智能采样策略...")
                print(f"遍历文本库，判断每个文本是否讲述了{attack_type}攻击的故事...")
                
                selected_texts = []
                total_checked = 0
                # 随机化文本列表序列
                random.seed(time.time())
                random.shuffle(text_files)

                # 遍历所有文本，逐个判断
                for i, file_path in enumerate(text_files):
                    # 如果已经采样够了，就停止
                    if len(selected_texts) >= MAX_TEXT_SAMPLES:
                        print(f"已采样到 {MAX_TEXT_SAMPLES} 个相关文本，停止采样")
                        break
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            if not content:
                                continue
                            
                            total_checked += 1
                            
                            # 使用LLM判断文本是否相关
                            is_relevant, confidence = self._judge_text_relevance(content, attack_type)
                            
                            if is_relevant:
                                selected_texts.append({
                                    'path': file_path,
                                    'content': content,
                                    'confidence': confidence
                                })
                                print(f"  ✓ [{len(selected_texts)}/{MAX_TEXT_SAMPLES}] 选中文本 (置信度: {confidence:.2f}): {os.path.basename(file_path)}")
                            else:
                                # 可选：打印被拒绝的文本（取消注释以查看）
                                # print(f"  ✗ 跳过文本 (置信度: {confidence:.2f}): {os.path.basename(file_path)}")
                                pass
                            
                            # 打印进度
                            if total_checked % 10 == 0:
                                print(f"  进度: 已检查 {total_checked}/{len(text_files)} 个文件，已选中 {len(selected_texts)} 个相关文本")
                    
                    except Exception as e:
                        print(f"  ! 读取文件失败 {file_path}: {str(e)}")
                
                if len(selected_texts) >= MAX_TEXT_SAMPLES:
                    # 如果采样数量足够，取前MAX_TEXT_SAMPLES个
                    selected_texts = selected_texts[:MAX_TEXT_SAMPLES]
                    print(f"智能采样完成，选取了 {len(selected_texts)} 个相关文本")
                    print(f"  平均置信度: {sum(t['confidence'] for t in selected_texts) / len(selected_texts):.2f}")
                    return [{'path': t['path'], 'content': t['content']} for t in selected_texts]
                elif len(selected_texts) > 0:
                    # 如果采样数量不足但有一些，使用这些
                    print(f"警告: 只找到 {len(selected_texts)} 个相关文本（目标: {MAX_TEXT_SAMPLES}）")
                    print(f"  将使用这 {len(selected_texts)} 个文本继续")
                    return [{'path': t['path'], 'content': t['content']} for t in selected_texts]
                else:
                    # 如果一个都没找到，回退到随机采样
                    print("警告: 未找到任何相关文本，回退到随机采样")
                    selected_files = random.sample(text_files, MAX_TEXT_SAMPLES)
            else:
                # 使用随机采样
                print(f"使用随机采样策略...")
                selected_files = random.sample(text_files, MAX_TEXT_SAMPLES)
            
            print(f"采样后保留 {len(selected_files)} 个文本文件")
        
        # 读取文本内容（如果还没有读取）
        texts = []
        for file_path in selected_files:
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

