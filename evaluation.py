# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class SchemaEvaluator:
    """schema匹配评估工具类"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        初始化评估器
        Args:
            similarity_threshold: 余弦相似度阈值，超过此值认为两个词相似
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # 不使用停用词，因为可能影响专业术语
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            max_features=10000
        )
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""
        # 移除多余空格，转换为小写
        text = re.sub(r'\s+', ' ', text.strip().lower())
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（结合多种方法）"""
        if not text1 or not text2:
            return 0.0
        
        # 预处理文本
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        if text1 == text2:
            return 1.0
        
        # 1. 编辑距离相似度（适用于短文本）
        edit_similarity = self._edit_distance_similarity(text1, text2)
        
        # 2. 子串包含相似度
        substring_similarity = self._substring_similarity(text1, text2)
        
        # 3. TF-IDF余弦相似度（适用于长文本）
        tfidf_similarity = self._tfidf_similarity(text1, text2)
        
        # 4. 词根相似度
        stem_similarity = self._stem_similarity(text1, text2)
        
        # 综合相似度：取最大值，因为不同方法适用于不同情况
        max_similarity = max(edit_similarity, substring_similarity, tfidf_similarity, stem_similarity)
        
        return max_similarity
    
    def _edit_distance_similarity(self, text1: str, text2: str) -> float:
        """基于编辑距离的相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 简化的编辑距离计算
        m, n = len(text1), len(text2)
        if m == 0:
            return 0.0
        if n == 0:
            return 0.0
        
        # 创建距离矩阵
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # 转换为相似度分数
        max_len = max(m, n)
        similarity = 1 - (dp[m][n] / max_len)
        return max(0.0, similarity)
    
    def _substring_similarity(self, text1: str, text2: str) -> float:
        """基于子串包含的相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 检查一个文本是否包含另一个文本
        if text1 in text2 or text2 in text1:
            shorter = min(len(text1), len(text2))
            longer = max(len(text1), len(text2))
            return shorter / longer
        
        return 0.0
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """基于TF-IDF的余弦相似度"""
        try:
            # 使用TF-IDF向量化
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            return 0.0
    
    def _stem_similarity(self, text1: str, text2: str) -> float:
        """基于词根的相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的词根提取（移除常见后缀）
        def simple_stem(word):
            suffixes = ['s', 'es', 'ed', 'ing', 'ly', 'er', 'est', 'tion', 'sion']
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    return word[:-len(suffix)]
            return word
        
        stem1 = simple_stem(text1)
        stem2 = simple_stem(text2)
        
        if stem1 == stem2:
            return 0.9  # 词根相同但可能有不同后缀
        elif stem1 in stem2 or stem2 in stem1:
            return 0.7  # 一个词根包含另一个
        
        return 0.0
    
    def find_best_match(self, target_item: Dict, candidate_list: List[Dict], 
                       key_fields: List[str]) -> Tuple[Dict, float]:
        """
        在候选列表中找到与目标项最相似的项
        Args:
            target_item: 目标项
            candidate_list: 候选列表
            key_fields: 用于匹配的关键字段列表
        Returns:
            (最佳匹配项, 相似度分数)
        """
        best_match = None
        best_similarity = 0.0
        
        for candidate in candidate_list:
            # 计算所有关键字段的相似度
            field_similarities = []
            for field in key_fields:
                target_value = target_item.get(field, "")
                candidate_value = candidate.get(field, "")
                similarity = self.calculate_similarity(target_value, candidate_value)
                field_similarities.append(similarity)
            
            # 使用平均相似度作为整体相似度
            # avg_similarity = np.mean(field_similarities) if field_similarities else 0.0
            # 使用max相似度作为整体相似度
            avg_similarity = max(field_similarities) if field_similarities else 0.0

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = candidate
        
        return best_match, best_similarity

    @staticmethod
    def load_ground_truth(data_dir: str) -> Dict:
        """加载论文提供的标注数据（真实值）"""
        ground_truth = {
            "entities": [],
            "events": [],
            "relations": [],
            "temporal_relations": []
        }

        # 假设论文数据为JSON格式，根据实际数据结构调整
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # 根据实际数据字段调整
                        if "entities" in data:
                            ground_truth["entities"].extend(data["entities"])
                        if "events" in data:
                            ground_truth["events"].extend(data["events"])
                        if "relations" in data:
                            ground_truth["relations"].extend(data["relations"])
                        if "temporal_relations" in data:
                            ground_truth["temporal_relations"].extend(data["temporal_relations"])

        return ground_truth

    def evaluate_entities(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
        """评估实体识别结果（使用词相似度匹配）"""
        if not predicted and not ground_truth:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth)}
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predicted), "fn": 0}
        
        true_positive = 0
        matched_truth_indices = set()
        
        # 对每个预测实体，在真实标签中找最佳匹配
        for pred_entity in predicted:
            best_match, similarity = self.find_best_match(
                pred_entity, ground_truth, ["实体文本", "类型"]
            )
            
            if best_match and similarity >= self.similarity_threshold:
                # 找到匹配的真实标签索引
                for i, truth_entity in enumerate(ground_truth):
                    if (truth_entity.get("实体文本") == best_match.get("实体文本") and 
                        truth_entity.get("类型") == best_match.get("类型")):
                        if i not in matched_truth_indices:
                            true_positive += 1
                            matched_truth_indices.add(i)
                        break
        
        precision = true_positive / len(predicted)
        recall = true_positive / len(ground_truth)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positive,
            "fp": len(predicted) - true_positive,
            "fn": len(ground_truth) - true_positive
        }

    def evaluate_events(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
        """评估事件识别结果（使用词相似度匹配）"""
        if not predicted and not ground_truth:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth)}
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predicted), "fn": 0}
        
        true_positive = 0
        matched_truth_indices = set()
        
        # 对每个预测事件，在真实标签中找最佳匹配
        for pred_event in predicted:
            best_match, similarity = self.find_best_match(
                pred_event, ground_truth, ["触发词", "事件类型"]
            )
            
            if best_match and similarity >= self.similarity_threshold:
                # 找到匹配的真实标签索引
                for i, truth_event in enumerate(ground_truth):
                    if (truth_event.get("触发词") == best_match.get("触发词") and 
                        truth_event.get("事件类型") == best_match.get("事件类型")):
                        if i not in matched_truth_indices:
                            true_positive += 1
                            matched_truth_indices.add(i)
                        break
        
        precision = true_positive / len(predicted)
        recall = true_positive / len(ground_truth)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positive,
            "fp": len(predicted) - true_positive,
            "fn": len(ground_truth) - true_positive
        }

    def evaluate_relations(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
        """评估实体关系结果（使用词相似度匹配）"""
        if not predicted and not ground_truth:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth)}
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predicted), "fn": 0}
        
        true_positive = 0
        matched_truth_indices = set()
        
        # 对每个预测关系，在真实标签中找最佳匹配
        for pred_relation in predicted:
            best_match, similarity = self.find_best_match(
                pred_relation, ground_truth, ["实体1", "关系类型", "实体2"]
            )
            
            if best_match and similarity >= self.similarity_threshold:
                # 找到匹配的真实标签索引
                for i, truth_relation in enumerate(ground_truth):
                    if (truth_relation.get("实体1") == best_match.get("实体1") and 
                        truth_relation.get("关系类型") == best_match.get("关系类型") and
                        truth_relation.get("实体2") == best_match.get("实体2")):
                        if i not in matched_truth_indices:
                            true_positive += 1
                            matched_truth_indices.add(i)
                        break
        
        precision = true_positive / len(predicted)
        recall = true_positive / len(ground_truth)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positive,
            "fp": len(predicted) - true_positive,
            "fn": len(ground_truth) - true_positive
        }
    
    def evaluate_temporal_relations(self, predicted: List[Dict], ground_truth: List[Dict]) -> Dict:
        """评估时间关系结果（使用词相似度匹配）"""
        if not predicted and not ground_truth:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth)}
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predicted), "fn": 0}
        
        true_positive = 0
        matched_truth_indices = set()
        
        # 对每个预测时间关系，在真实标签中找最佳匹配
        for pred_temporal in predicted:
            best_match, similarity = self.find_best_match(
                pred_temporal, ground_truth, ["事件类型1", "时间关系", "事件类型2"]
            )
            
            if best_match and similarity >= self.similarity_threshold:
                # 找到匹配的真实标签索引
                for i, truth_temporal in enumerate(ground_truth):
                    if (truth_temporal.get("事件类型1") == best_match.get("事件类型1") and 
                        truth_temporal.get("时间关系") == best_match.get("时间关系") and
                        truth_temporal.get("事件类型2") == best_match.get("事件类型2")):
                        if i not in matched_truth_indices:
                            true_positive += 1
                            matched_truth_indices.add(i)
                        break
        
        precision = true_positive / len(predicted)
        recall = true_positive / len(ground_truth)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positive,
            "fp": len(predicted) - true_positive,
            "fn": len(ground_truth) - true_positive
        }
    
    def evaluate_all(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """综合评估所有任务"""
        results = {}
        
        # 评估实体识别
        if "entities" in predicted and "entities" in ground_truth:
            results["entities"] = self.evaluate_entities(
                predicted["entities"], ground_truth["entities"]
            )
        
        # 评估事件识别
        if "events" in predicted and "events" in ground_truth:
            results["events"] = self.evaluate_events(
                predicted["events"], ground_truth["events"]
            )
        
        # 评估实体关系
        if "relations" in predicted and "relations" in ground_truth:
            results["relations"] = self.evaluate_relations(
                predicted["relations"], ground_truth["relations"]
            )
        
        # 评估时间关系
        if "temporal_relations" in predicted and "temporal_relations" in ground_truth:
            results["temporal_relations"] = self.evaluate_temporal_relations(
                predicted["temporal_relations"], ground_truth["temporal_relations"]
            )
        
        # 计算总体F1值
        f1_scores = [result["f1"] for result in results.values()]
        overall_f1 = np.mean(f1_scores) if f1_scores else 0.0
        results["overall"] = {"f1": overall_f1}
        
        return results