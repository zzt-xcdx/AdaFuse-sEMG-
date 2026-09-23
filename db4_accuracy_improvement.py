#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DB4数据集准确率提升分析
专门针对results_report_finegate_DB4_full目录中的结果进行分析和改进
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
import re
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DB4AccuracyAnalyzer:
    """DB4数据集准确率分析器"""
    
    def __init__(self, results_dir='results_report_finegate_DB4_full'):
        self.results_dir = results_dir
        self.subject_accuracies = {}
        self.subject_details = {}
        
    def load_all_results(self):
        """加载所有受试者的结果"""
        print("=== 加载DB4数据集结果 ===")
        
        # 加载准确率数据
        accuracies_file = os.path.join(self.results_dir, 'subject_accuracies.npy')
        if os.path.exists(accuracies_file):
            try:
                results = np.load(accuracies_file, allow_pickle=True).item()
                self.subject_accuracies = results
                print(f"成功加载准确率数据，包含 {len(results.get('subject_accuracies', []))} 个受试者")
            except Exception as e:
                print(f"加载准确率文件时出错: {e}")
        
        # 加载分类报告
        self.load_classification_reports()
        
    def load_classification_reports(self):
        """加载分类报告"""
        print("加载分类报告...")
        
        for file in os.listdir(self.results_dir):
            if file.startswith('classification_report_subject_') and file.endswith('.txt'):
                # 修复文件名解析
                try:
                    subject_id = int(file.split('_')[3].split('.')[0])
                    file_path = os.path.join(self.results_dir, file)
                except (IndexError, ValueError):
                    print(f"无法解析文件名: {file}")
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # 提取准确率
                    accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content)
                    if accuracy_match:
                        accuracy = float(accuracy_match.group(1)) * 100  # 转换为百分比
                        self.subject_details[subject_id] = {
                            'accuracy': accuracy,
                            'report_content': content
                        }
                        
                except Exception as e:
                    print(f"加载受试者 {subject_id} 报告时出错: {e}")
        
        print(f"成功加载 {len(self.subject_details)} 个受试者的分类报告")
    
    def analyze_performance(self):
        """分析性能"""
        print("\n=== DB4数据集性能分析 ===")
        
        if not self.subject_details:
            print("没有找到分类报告数据")
            return None
        
        # 提取准确率
        accuracies = [details['accuracy'] for details in self.subject_details.values()]
        subject_ids = list(self.subject_details.keys())
        
        # 统计分析
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        print(f"DB4数据集总体准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"最高准确率: {max_acc:.2f}%")
        print(f"最低准确率: {min_acc:.2f}%")
        
        # 识别低性能受试者
        low_performance_threshold = mean_acc - std_acc
        low_performance_subjects = [sid for sid, acc in zip(subject_ids, accuracies) if acc < low_performance_threshold]
        
        print(f"低性能受试者 (< {low_performance_threshold:.2f}%): {low_performance_subjects}")
        
        # 性能分布
        performance_distribution = {
            'excellent': [sid for sid, acc in zip(subject_ids, accuracies) if acc >= 90],
            'good': [sid for sid, acc in zip(subject_ids, accuracies) if 80 <= acc < 90],
            'fair': [sid for sid, acc in zip(subject_ids, accuracies) if 70 <= acc < 80],
            'poor': [sid for sid, acc in zip(subject_ids, accuracies) if acc < 70]
        }
        
        print(f"\n性能分布:")
        print(f"优秀 (≥90%): {len(performance_distribution['excellent'])} 个受试者")
        print(f"良好 (80-90%): {len(performance_distribution['good'])} 个受试者")
        print(f"一般 (70-80%): {len(performance_distribution['fair'])} 个受试者")
        print(f"较差 (<70%): {len(performance_distribution['poor'])} 个受试者")
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc,
            'low_performance_subjects': low_performance_subjects,
            'performance_distribution': performance_distribution,
            'all_accuracies': accuracies,
            'subject_ids': subject_ids
        }
    
    def analyze_class_specific_performance(self, subject_id):
        """分析特定受试者的类别性能"""
        if subject_id not in self.subject_details:
            print(f"未找到受试者 {subject_id} 的数据")
            return None
        
        content = self.subject_details[subject_id]['report_content']
        
        # 解析分类报告
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
    
    def identify_problematic_classes(self, subject_id):
        """识别问题类别"""
        class_performance = self.analyze_class_specific_performance(subject_id)
        if not class_performance:
            return None
        
        problematic_classes = {
            'low_precision': [],
            'low_recall': [],
            'low_f1': [],
            'low_support': []
        }
        
        for class_id, metrics in class_performance.items():
            if metrics['precision'] < 0.6:
                problematic_classes['low_precision'].append(class_id)
            if metrics['recall'] < 0.6:
                problematic_classes['low_recall'].append(class_id)
            if metrics['f1_score'] < 0.6:
                problematic_classes['low_f1'].append(class_id)
            if metrics['support'] < 50:  # 样本数量少
                problematic_classes['low_support'].append(class_id)
        
        return problematic_classes
    
    def generate_visualizations(self, performance_data):
        """生成可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        accuracies = performance_data['all_accuracies']
        subject_ids = performance_data['subject_ids']
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 准确率分布直方图
        axes[0, 0].hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(performance_data['mean_accuracy'], color='red', linestyle='--', 
                          label=f'平均值: {performance_data["mean_accuracy"]:.2f}%')
        axes[0, 0].axvline(performance_data['mean_accuracy'] - performance_data['std_accuracy'], 
                          color='orange', linestyle='--', 
                          label=f'低性能阈值: {performance_data["mean_accuracy"] - performance_data["std_accuracy"]:.2f}%')
        axes[0, 0].set_xlabel('准确率 (%)')
        axes[0, 0].set_ylabel('受试者数量')
        axes[0, 0].set_title('DB4数据集准确率分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 准确率排序图
        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]
        sorted_subjects = [subject_ids[i] for i in sorted_indices]
        
        axes[0, 1].plot(range(1, len(sorted_accuracies)+1), sorted_accuracies, 'o-', alpha=0.7)
        axes[0, 1].axhline(performance_data['mean_accuracy'], color='red', linestyle='--', 
                          label=f'平均值: {performance_data["mean_accuracy"]:.2f}%')
        axes[0, 1].set_xlabel('受试者排名')
        axes[0, 1].set_ylabel('准确率 (%)')
        axes[0, 1].set_title('受试者准确率排序')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 性能分布饼图
        distribution = performance_data['performance_distribution']
        labels = ['优秀 (≥90%)', '良好 (80-90%)', '一般 (70-80%)', '较差 (<70%)']
        sizes = [len(distribution['excellent']), len(distribution['good']), 
                len(distribution['fair']), len(distribution['poor'])]
        colors = ['#2E8B57', '#87CEEB', '#FFD700', '#FF6347']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('性能分布')
        
        # 4. 低性能受试者详细分析
        low_perf_subjects = performance_data['low_performance_subjects']
        if low_perf_subjects:
            low_perf_accuracies = [self.subject_details[sid]['accuracy'] for sid in low_perf_subjects]
            axes[1, 1].bar(low_perf_subjects, low_perf_accuracies, alpha=0.7, color='red')
            axes[1, 1].axhline(performance_data['mean_accuracy'] - performance_data['std_accuracy'], 
                              color='orange', linestyle='--', label='低性能阈值')
            axes[1, 1].set_xlabel('受试者编号')
            axes[1, 1].set_ylabel('准确率 (%)')
            axes[1, 1].set_title('低性能受试者详细分析')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '没有低性能受试者', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('低性能受试者分析')
        
        plt.tight_layout()
        plt.savefig('db4_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化图表已保存到 db4_accuracy_analysis.png")
    
    def generate_improvement_recommendations(self, performance_data):
        """生成改进建议"""
        print("\n=== 生成改进建议 ===")
        
        mean_acc = performance_data['mean_accuracy']
        low_perf_subjects = performance_data['low_performance_subjects']
        distribution = performance_data['performance_distribution']
        
        recommendations = []
        
        # 基于总体性能的建议
        if mean_acc < 75:
            recommendations.append("🔴 高优先级：总体准确率较低，需要系统性改进")
        elif mean_acc < 80:
            recommendations.append("🟡 中优先级：准确率有提升空间，建议针对性优化")
        else:
            recommendations.append("🟢 低优先级：准确率较高，可进行精细调优")
        
        # 基于低性能受试者的建议
        if len(low_perf_subjects) > 3:
            recommendations.append("🔴 重点关注：存在多个低性能受试者，需要个性化处理")
        elif len(low_perf_subjects) > 0:
            recommendations.append("🟡 部分优化：针对低性能受试者进行专门优化")
        
        # 基于性能分布的建议
        if len(distribution['poor']) > 0:
            recommendations.append("🔴 需要改进：存在表现较差的受试者")
        if len(distribution['excellent']) < len(distribution['good']):
            recommendations.append("🟡 提升空间：优秀受试者比例较低")
        
        # 具体改进方向
        recommendations.extend([
            "📊 信号质量增强：自适应滤波、噪声抑制",
            "🔧 特征工程改进：多尺度特征、时频域特征",
            "🧠 模型架构优化：注意力机制、残差连接",
            "⚡ 训练策略增强：学习率调度、数据增强",
            "🔄 集成方法：多模型融合、特征集成"
        ])
        
        return recommendations
    
    def create_detailed_report(self, performance_data, recommendations):
        """创建详细报告"""
        print("\n=== 创建详细报告 ===")
        
        report = f"""
# DB4数据集准确率提升分析报告

## 数据集概况
- 数据集：DB4 (finegate版本)
- 受试者数量：{len(performance_data['subject_ids'])}
- 总体准确率：{performance_data['mean_accuracy']:.2f}% ± {performance_data['std_accuracy']:.2f}%
- 最高准确率：{performance_data['max_accuracy']:.2f}%
- 最低准确率：{performance_data['min_accuracy']:.2f}%

## 性能分布
- 优秀 (≥90%)：{len(performance_data['performance_distribution']['excellent'])} 个受试者
- 良好 (80-90%)：{len(performance_data['performance_distribution']['good'])} 个受试者
- 一般 (70-80%)：{len(performance_data['performance_distribution']['fair'])} 个受试者
- 较差 (<70%)：{len(performance_data['performance_distribution']['poor'])} 个受试者

## 低性能受试者分析
低性能受试者 (< {performance_data['mean_accuracy'] - performance_data['std_accuracy']:.2f}%)：{performance_data['low_performance_subjects']}

"""
        
        # 添加低性能受试者的详细分析
        if performance_data['low_performance_subjects']:
            report += "\n### 低性能受试者详细分析\n"
            for subject_id in performance_data['low_performance_subjects']:
                accuracy = self.subject_details[subject_id]['accuracy']
                problematic_classes = self.identify_problematic_classes(subject_id)
                
                report += f"\n**受试者 {subject_id}** (准确率: {accuracy:.2f}%)\n"
                if problematic_classes:
                    report += "- 问题类别分析：\n"
                    if problematic_classes['low_precision']:
                        report += f"  - 低精确率类别: {problematic_classes['low_precision']}\n"
                    if problematic_classes['low_recall']:
                        report += f"  - 低召回率类别: {problematic_classes['low_recall']}\n"
                    if problematic_classes['low_f1']:
                        report += f"  - 低F1分数类别: {problematic_classes['low_f1']}\n"
                    if problematic_classes['low_support']:
                        report += f"  - 样本稀少类别: {problematic_classes['low_support']}\n"
        
        # 添加改进建议
        report += "\n## 改进建议\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        # 添加实施计划
        report += """
## 实施计划

### 第一阶段：快速改进 (1-2周)
1. 针对低性能受试者进行个性化优化
2. 实施自适应信号增强
3. 优化特征选择策略

### 第二阶段：深度优化 (2-4周)
1. 改进模型架构
2. 实施集成学习方法
3. 超参数优化

### 第三阶段：精细调优 (1-2月)
1. 跨受试者泛化优化
2. 鲁棒性提升
3. 实时性能优化

## 预期效果
- 总体准确率提升 3-5%
- 低性能受试者准确率提升 5-10%
- 减少受试者间性能差异
- 提高模型鲁棒性

---
报告生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        with open('db4_improvement_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("详细报告已保存到 db4_improvement_report.md")

def main():
    """主函数"""
    print("DB4数据集准确率提升分析")
    print("=" * 50)
    
    # 创建分析器
    analyzer = DB4AccuracyAnalyzer()
    
    # 加载结果
    analyzer.load_all_results()
    
    # 分析性能
    performance_data = analyzer.analyze_performance()
    
    if performance_data:
        # 生成可视化
        analyzer.generate_visualizations(performance_data)
        
        # 生成改进建议
        recommendations = analyzer.generate_improvement_recommendations(performance_data)
        
        # 打印建议
        print("\n改进建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # 创建详细报告
        analyzer.create_detailed_report(performance_data, recommendations)
        
        print("\n=== 分析完成 ===")
        print("请查看以下文件：")
        print("1. db4_accuracy_analysis.png - 准确率分析图")
        print("2. db4_improvement_report.md - 详细改进报告")
    else:
        print("无法完成分析，请检查数据文件")

if __name__ == "__main__":
    main() 