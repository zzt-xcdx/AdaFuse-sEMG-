#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门分析受试者7和8的问题
深入分析他们的具体表现和问题类别
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Subject7And8Analyzer:
    """受试者7和8分析器"""
    
    def __init__(self, results_dir='results_report_finegate_DB4_full'):
        self.results_dir = results_dir
        self.subject_data = {}
        
    def load_subject_reports(self):
        """加载受试者7和8的分类报告"""
        print("=== 加载受试者7和8的分类报告 ===")
        
        for subject_id in [7, 8]:
            file_path = os.path.join(self.results_dir, f'classification_report_subject_{subject_id}.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析分类报告
                class_performance = self.parse_classification_report(content)
                
                self.subject_data[subject_id] = {
                    'content': content,
                    'class_performance': class_performance
                }
                
                print(f"成功加载受试者 {subject_id} 的报告")
            else:
                print(f"未找到受试者 {subject_id} 的报告文件")
    
    def parse_classification_report(self, content):
        """解析分类报告"""
        lines = content.split('\n')
        class_performance = {}
        
        for line in lines:
            if re.match(r'\s*\d+\s+', line):  # 匹配类别行
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1_score = float(parts[3])
                    support = int(parts[4])
                    
                    class_performance[class_id] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score,
                        'support': support
                    }
        
        return class_performance
    
    def analyze_problematic_classes(self):
        """分析问题类别"""
        print("\n=== 分析问题类别 ===")
        
        for subject_id in [7, 8]:
            if subject_id not in self.subject_data:
                continue
                
            print(f"\n受试者 {subject_id} 的问题分析:")
            class_performance = self.subject_data[subject_id]['class_performance']
            
            # 识别问题类别
            problems = {
                'very_low_precision': [],  # 精确率 < 0.5
                'low_precision': [],       # 精确率 < 0.7
                'very_low_recall': [],     # 召回率 < 0.5
                'low_recall': [],          # 召回率 < 0.7
                'very_low_f1': [],         # F1 < 0.5
                'low_f1': [],              # F1 < 0.7
                'small_samples': []        # 样本数 < 50
            }
            
            for class_id, metrics in class_performance.items():
                if metrics['precision'] < 0.5:
                    problems['very_low_precision'].append(class_id)
                elif metrics['precision'] < 0.7:
                    problems['low_precision'].append(class_id)
                    
                if metrics['recall'] < 0.5:
                    problems['very_low_recall'].append(class_id)
                elif metrics['recall'] < 0.7:
                    problems['low_recall'].append(class_id)
                    
                if metrics['f1_score'] < 0.5:
                    problems['very_low_f1'].append(class_id)
                elif metrics['f1_score'] < 0.7:
                    problems['low_f1'].append(class_id)
                    
                if metrics['support'] < 50:
                    problems['small_samples'].append(class_id)
            
            # 打印问题统计
            print(f"  严重问题类别 (精确率<0.5): {problems['very_low_precision']}")
            print(f"  一般问题类别 (精确率<0.7): {problems['low_precision']}")
            print(f"  严重问题类别 (召回率<0.5): {problems['very_low_recall']}")
            print(f"  一般问题类别 (召回率<0.7): {problems['low_recall']}")
            print(f"  严重问题类别 (F1<0.5): {problems['very_low_f1']}")
            print(f"  样本稀少类别 (<50样本): {problems['small_samples']}")
            
            # 找出最严重的问题类别
            critical_problems = set(problems['very_low_precision']) | set(problems['very_low_recall']) | set(problems['very_low_f1'])
            print(f"  最严重问题类别: {sorted(list(critical_problems))}")
    
    def compare_with_better_subjects(self):
        """与表现较好的受试者比较"""
        print("\n=== 与表现较好的受试者比较 ===")
        
        # 加载表现较好的受试者（比如受试者1，准确率84%）
        better_subject_id = 1
        better_file = os.path.join(self.results_dir, f'classification_report_subject_{better_subject_id}.txt')
        
        if os.path.exists(better_file):
            with open(better_file, 'r', encoding='utf-8') as f:
                better_content = f.read()
            
            better_performance = self.parse_classification_report(better_content)
            
            print(f"\n与受试者 {better_subject_id} (准确率84%) 的比较:")
            
            for subject_id in [7, 8]:
                if subject_id not in self.subject_data:
                    continue
                    
                print(f"\n受试者 {subject_id} vs 受试者 {better_subject_id}:")
                
                subject_performance = self.subject_data[subject_id]['class_performance']
                
                # 比较每个类别的表现
                worse_classes = []
                for class_id in subject_performance.keys():
                    if class_id in better_performance:
                        subject_f1 = subject_performance[class_id]['f1_score']
                        better_f1 = better_performance[class_id]['f1_score']
                        
                        if subject_f1 < better_f1 * 0.8:  # 表现差20%以上
                            worse_classes.append((class_id, subject_f1, better_f1))
                
                # 按差异排序
                worse_classes.sort(key=lambda x: x[2] - x[1], reverse=True)
                
                print(f"  表现明显较差的类别 (前10个):")
                for class_id, subject_f1, better_f1 in worse_classes[:10]:
                    print(f"    类别 {class_id}: {subject_f1:.3f} vs {better_f1:.3f} (差异: {better_f1-subject_f1:.3f})")
    
    def generate_visualization(self):
        """生成可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, subject_id in enumerate([7, 8]):
            if subject_id not in self.subject_data:
                continue
                
            class_performance = self.subject_data[subject_id]['class_performance']
            
            # 提取数据
            class_ids = list(class_performance.keys())
            precisions = [class_performance[cid]['precision'] for cid in class_ids]
            recalls = [class_performance[cid]['recall'] for cid in class_ids]
            f1_scores = [class_performance[cid]['f1_score'] for cid in class_ids]
            
            # 绘制精确率、召回率、F1分数
            x = np.arange(len(class_ids))
            width = 0.25
            
            axes[i, 0].bar(x - width, precisions, width, label='精确率', alpha=0.8)
            axes[i, 0].bar(x, recalls, width, label='召回率', alpha=0.8)
            axes[i, 0].bar(x + width, f1_scores, width, label='F1分数', alpha=0.8)
            
            axes[i, 0].set_xlabel('类别编号')
            axes[i, 0].set_ylabel('分数')
            axes[i, 0].set_title(f'受试者 {subject_id} 各类别性能')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_xticks(x)
            axes[i, 0].set_xticklabels(class_ids, rotation=45)
            
            # 绘制问题类别热力图
            problem_matrix = np.zeros((len(class_ids), 3))
            for j, cid in enumerate(class_ids):
                metrics = class_performance[cid]
                problem_matrix[j, 0] = 1 if metrics['precision'] < 0.7 else 0
                problem_matrix[j, 1] = 1 if metrics['recall'] < 0.7 else 0
                problem_matrix[j, 2] = 1 if metrics['f1_score'] < 0.7 else 0
            
            im = axes[i, 1].imshow(problem_matrix.T, cmap='Reds', aspect='auto')
            axes[i, 1].set_xlabel('类别编号')
            axes[i, 1].set_ylabel('问题类型')
            axes[i, 1].set_title(f'受试者 {subject_id} 问题类别热力图')
            axes[i, 1].set_yticks([0, 1, 2])
            axes[i, 1].set_yticklabels(['精确率<0.7', '召回率<0.7', 'F1<0.7'])
            axes[i, 1].set_xticks(range(len(class_ids)))
            axes[i, 1].set_xticklabels(class_ids, rotation=45)
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i, 1])
        
        plt.tight_layout()
        plt.savefig('subjects_7_8_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化图表已保存到 subjects_7_8_analysis.png")
    
    def generate_improvement_suggestions(self):
        """生成改进建议"""
        print("\n=== 生成改进建议 ===")
        
        suggestions = {}
        
        for subject_id in [7, 8]:
            if subject_id not in self.subject_data:
                continue
                
            class_performance = self.subject_data[subject_id]['class_performance']
            
            # 分析问题模式
            precision_problems = [cid for cid, metrics in class_performance.items() if metrics['precision'] < 0.7]
            recall_problems = [cid for cid, metrics in class_performance.items() if metrics['recall'] < 0.7]
            f1_problems = [cid for cid, metrics in class_performance.items() if metrics['f1_score'] < 0.7]
            small_samples = [cid for cid, metrics in class_performance.items() if metrics['support'] < 50]
            
            subject_suggestions = []
            
            # 基于问题类型生成建议
            if len(precision_problems) > len(class_performance) * 0.3:
                subject_suggestions.append("精确率问题严重 - 建议增强特征区分度")
            
            if len(recall_problems) > len(class_performance) * 0.3:
                subject_suggestions.append("召回率问题严重 - 建议增加数据增强和正则化")
            
            if len(small_samples) > 0:
                subject_suggestions.append(f"样本稀少类别: {small_samples} - 建议数据增强或迁移学习")
            
            # 找出最需要改进的类别
            worst_classes = sorted(class_performance.items(), key=lambda x: x[1]['f1_score'])[:5]
            subject_suggestions.append(f"最需要改进的类别: {[cid for cid, _ in worst_classes]}")
            
            suggestions[subject_id] = subject_suggestions
            
            print(f"\n受试者 {subject_id} 改进建议:")
            for i, suggestion in enumerate(subject_suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        return suggestions

def main():
    """主函数"""
    print("受试者7和8问题分析")
    print("=" * 50)
    
    # 创建分析器
    analyzer = Subject7And8Analyzer()
    
    # 加载数据
    analyzer.load_subject_reports()
    
    # 分析问题
    analyzer.analyze_problematic_classes()
    
    # 与表现较好的受试者比较
    analyzer.compare_with_better_subjects()
    
    # 生成可视化
    analyzer.generate_visualization()
    
    # 生成改进建议
    suggestions = analyzer.generate_improvement_suggestions()
    
    print("\n=== 分析完成 ===")
    print("请查看 subjects_7_8_analysis.png 了解详细分析结果")

if __name__ == "__main__":
    main() 