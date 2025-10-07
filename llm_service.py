# -*- coding: utf-8 -*-
"""
大语言模型服务类
负责与智谱AI API的交互
"""
import json
import time
from time import sleep

# 尝试导入新的API库，如果失败则使用旧的
try:
    from zai import ZhipuAiClient
    USE_NEW_API = True
except Exception as e:
    from zhipuai import ZhipuAI
    USE_NEW_API = False
    print("警告: 未安装新版智谱AI库(zai)，使用旧版(zhipuai)")
    print(e)


class LLMServiceOld:
    """大语言模型服务类（旧版API），封装智谱AI API调用 - 保留作为备份"""
    
    def __init__(self, api_key, max_retries=3, retry_delay=2):
        """
        初始化LLM服务
        
        Args:
            api_key (str): 智谱AI的API密钥
            max_retries (int): 最大重试次数
            retry_delay (int): 重试延迟（秒）
        """
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
    
    def call_api(self, prompt, temperature=0.4, top_p=0.9):
        """
        通用API调用函数，处理响应解析（带重试机制）
        
        Args:
            prompt (str): 输入提示词
            temperature (float): 温度参数，控制随机性
            top_p (float): top_p参数，控制核采样
            
        Returns:
            dict: 解析后的JSON响应，失败时返回空列表
        """
        self.total_calls += 1
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="glm-4-flash-250414",  # 最好的模型是glm-4.5，免费的是glm-4-flash-250414
                    # model="glm-4.6",  # 最好的模型是glm-4.5，免费的是glm-4-flash-250414
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
                
                # 如果这是重试后成功，记录
                if attempt > 0:
                    self.retried_calls += 1
                    print(f"  ✓ 重试第{attempt}次成功")
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                # 如果是最后一次尝试，记录失败并返回
                if attempt == self.max_retries - 1:
                    self.failed_calls += 1
                    print(f"  ✗ API调用失败（已重试{self.max_retries}次）: {error_msg}")
                    return []
                
                # 否则等待后重试
                print(f"  ! API调用失败（第{attempt + 1}次），{self.retry_delay}秒后重试: {error_msg}")
                time.sleep(self.retry_delay)
        
        return []
    
    def get_call_statistics(self):
        """获取API调用统计信息"""
        success_rate = ((self.total_calls - self.failed_calls) / self.total_calls * 100) if self.total_calls > 0 else 0
        return {
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'retried_calls': self.retried_calls,
            'success_rate': success_rate
        }
    
    def print_call_statistics(self):
        """打印API调用统计信息"""
        stats = self.get_call_statistics()
        print("\n" + "="*60)
        print("LLM API调用统计:")
        print(f"  总调用次数: {stats['total_calls']}")
        print(f"  失败次数: {stats['failed_calls']}")
        print(f"  重试成功次数: {stats['retried_calls']}")
        print(f"  成功率: {stats['success_rate']:.2f}%")
        print("="*60)
    
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


class LLMService:
    """大语言模型服务类（新版API），封装智谱AI API调用"""
    
    def __init__(self, api_key, max_retries=3, retry_delay=2, enable_thinking=False):
        """
        初始化LLM服务
        
        Args:
            api_key (str): 智谱AI的API密钥
            max_retries (int): 最大重试次数
            retry_delay (int): 重试延迟（秒）
            enable_thinking (bool): 是否启用深度思考模式
        """
        if USE_NEW_API:
            self.client = ZhipuAiClient(api_key=api_key)
        else:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=api_key)
        
        self.use_new_api = USE_NEW_API
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_thinking = enable_thinking
        self.advanced_model="glm-4.6"
        self.basic_model="glm-z1-flash"
        # self.basic_model="glm-4-flash-250414"
        self.total_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
        self.advanced_model_calls = 0  # 高级模型调用次数
        self.basic_model_calls = 0      # 低级模型调用次数
    
    def call_api(self, prompt, temperature=0.4, top_p=0.9, max_tokens=None, use_advanced_model=False):
        """
        通用API调用函数，处理响应解析（带重试机制）
        
        Args:
            prompt (str): 输入提示词
            temperature (float): 温度参数，控制随机性
            top_p (float): top_p参数，控制核采样
            max_tokens (int): 最大输出tokens，可选
            use_advanced_model (bool): 是否使用高级模型（glm-4.6），否则使用低级模型（glm-4-flash-250414）
            
        Returns:
            dict: 解析后的JSON响应，失败时返回空列表
        """
        time.sleep(1)
        self.total_calls += 1
        
        # 统计模型使用情况
        if use_advanced_model:
            self.advanced_model_calls += 1
        else:
            self.basic_model_calls += 1
        
        for attempt in range(self.max_retries):
            print(f"    - 调用API第{attempt + 1}次，调用模型: {self.advanced_model if use_advanced_model else self.basic_model}")
            try:
                # 根据参数选择模型
                model_name = self.advanced_model if use_advanced_model else self.basic_model
                
                # 构建基础参数
                call_params = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                
                # 根据API版本添加不同参数
                if self.use_new_api:
                    # 新版API参数
                    if self.enable_thinking:
                        call_params["thinking"] = {"type": "enabled"}
                    if max_tokens:
                        call_params["max_tokens"] = max_tokens
                    # 新版API使用response_format
                    call_params["response_format"] = {"type": "json_object"}
                else:
                    # 旧版API参数
                    call_params["top_p"] = top_p
                    call_params["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**call_params)
                
                # 解析JSON响应
                result = json.loads(response.choices[0].message.content)
                
                # 如果这是重试后成功，记录
                if attempt > 0:
                    self.retried_calls += 1
                    print(f"  ✓ 重试第{attempt}次成功")
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                # 如果是最后一次尝试，记录失败并返回
                if attempt == self.max_retries - 1:
                    self.failed_calls += 1
                    print(f"  ✗ API调用失败（已重试{self.max_retries}次）: {error_msg}")
                    return []
                
                # 否则等待后重试
                print(f"  ! API调用失败（第{attempt + 1}次），{self.retry_delay}秒后重试: {error_msg}")
                time.sleep(self.retry_delay)
        
        return []
    
    def get_call_statistics(self):
        """获取API调用统计信息"""
        success_rate = ((self.total_calls - self.failed_calls) / self.total_calls * 100) if self.total_calls > 0 else 0
        return {
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'retried_calls': self.retried_calls,
            'success_rate': success_rate,
            'advanced_model_calls': self.advanced_model_calls,
            'basic_model_calls': self.basic_model_calls
        }
    
    def print_call_statistics(self):
        """打印API调用统计信息"""
        stats = self.get_call_statistics()
        api_version = "新版API (zai)" if self.use_new_api else "旧版API (zhipuai)"
        print(f"  API版本: {api_version}")
        print(f"  总调用次数: {stats['total_calls']}")
        print(f"  - 高级模型({self.advanced_model}): {stats['advanced_model_calls']} 次")
        print(f"  - 低级模型({self.basic_model}): {stats['basic_model_calls']} 次")
        print(f"  失败次数: {stats['failed_calls']}")
        print(f"  重试成功次数: {stats['retried_calls']}")
        print(f"  成功率: {stats['success_rate']:.2f}%")
    
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
            entityList (list): 实体列表
            prompt_template (str): 事件识别的提示词模板
            
        Returns:
            list: 事件列表
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
