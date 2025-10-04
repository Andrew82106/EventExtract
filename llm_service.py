# -*- coding: utf-8 -*-
"""
大语言模型服务类
负责与智谱AI API的交互
"""
import json
from zhipuai import ZhipuAI


class LLMService:
    """大语言模型服务类，封装智谱AI API调用"""
    
    def __init__(self, api_key):
        """
        初始化LLM服务
        
        Args:
            api_key (str): 智谱AI的API密钥
        """
        self.client = ZhipuAI(api_key=api_key)
    
    def call_api(self, prompt, temperature=0.4, top_p=0.9):
        """
        通用API调用函数，处理响应解析
        
        Args:
            prompt (str): 输入提示词
            temperature (float): 温度参数，控制随机性
            top_p (float): top_p参数，控制核采样
            
        Returns:
            dict: 解析后的JSON响应，失败时返回空列表
        """
        try:
            response = self.client.chat.completions.create(
                # model="glm-4-flash-250414",  # 最好的模型是glm-4.5，免费的是glm-4-flash-250414
                model="glm-4.5",  # 最好的模型是glm-4.5，免费的是glm-4-flash-250414
                # model="glm-4.5",  # 最好的模型是glm-4.5，免费的是glm-4-flash-250414
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                response_format={
                    "type": "json_object"
                }
            )
            # 解析JSON响应
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return []
    
    def extract_entities(self, text, prompt_template):
        """
        实体识别
        
        Args:
            text (str): 输入文本
            prompt_template (str): 实体识别的提示词模板
            
        Returns:
            list: 实体列表
        """
        prompt = prompt_template.format(text=text)
        return self.call_api(prompt)
    
    def extract_events(self, text, entityList, prompt_template):
        """
        事件识别与论元抽取
        
        Args:
            text (str): 输入文本
            prompt_template (str): 事件识别的提示词模板
            
        Returns:
            list: 事件列表
            :param prompt_template:
            :param text:
            :param entityList:
        """
        prompt = prompt_template.format(text=text, entity_list=[i['实体文本'] for i in entityList])
        return self.call_api(prompt)
    
    def resolve_coreference(self, entity_list, prompt_template):
        """
        共指消解
        
        Args:
            entity_list (str): 输入文本
            prompt_template (str): 共指消解的提示词模板
            
        Returns:
            list: 共指消解结果
        """
        prompt = prompt_template.format(entity_list=entity_list)
        return self.call_api(prompt)
    
    def extract_entity_relations(self, entity_list, prompt_template):
        """
        实体关系抽取
        
        Args:
            entity_list (str): 输入文本
            prompt_template (str): 实体关系抽取的提示词模板
            
        Returns:
            list: 实体关系列表
        """
        prompt = prompt_template.format(entity_list=entity_list)
        return self.call_api(prompt)
    
    def extract_temporal_relations(self, text, event_list, prompt_template):
        """
        事件时间关系推理
        
        Args:
            text (str): 输入文本
            event_list (list): 列表
            prompt_template (str): 时间关系推理的提示词模板
            
        Returns:
            list: 时间关系列表
        """
        prompt = prompt_template.format(text=text, event_list=event_list)
        return self.call_api(prompt)
