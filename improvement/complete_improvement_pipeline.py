#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整改进流程
整合信号增强和特征选择，全面提升低准确率受试者的性能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入自定义模块
from improvement.adaptive_signal_enhancement import AdaptiveSignalEnhancer
from improvement.robust_feature_selection import RobustFeatureSelector, load_subject_features

class CompleteImprovementPipeline:
    """完整改进流程"""
    
    def __init__(self, dataset='DB4', root_data='D:/DB4'):
        self.dataset = dataset
        self.root_data = root_data
        self.signal_enhancer = AdaptiveSignalEnhancer(fs=2000)
        self.feature_selector = RobustFeatureSelector(n_features=50)
        
    def run_complete_pipeline(self, low_perf_subjects=[7, 8], high_perf_subjects=[10]):
        """运行完整改进流程"""
        print("="*80)
        print("完整改进流程开始")
        print("="*80)
        
        # 阶段1: 信号增强
        print("\n🔧 阶段1: 自适应信号增强")
        print("-" * 50)
        enhanced_data = self.enhance_signal_quality(low_perf_subjects)
        
        # 阶段2: 特征选择
        print("\n🎯 阶段2: 鲁棒特征选择")
        print("-" * 50)
        selected_features = self.select_robust_features(low_perf_subjects + high_perf_subjects)
        
        # 阶段3: 性能评估
        print("\n📊 阶段3: 改进效果评估")
        print("-" * 50)
        self.evaluate_improvements(low_perf_subjects, enhanced_data, selected_features)
        
        # 阶段4: 生成改进报告
        print("\n📋 阶段4: 生成改进报告")
        print("-" * 50)
        self.generate_improvement_report(low_perf_subjects, enhanced_data, selected_features)
        
        print("\n✅ 完整改进流程完成！")
    
    def enhance_signal_quality(self, subjects):
        """增强信号质量"""
        enhanced_data = {}
        
        for subject_id in subjects:
            print(f"\n处理受试者 {subject_id} 的信号增强...")
            
            # 增强数据
            enhanced_df, original_quality, enhanced_quality = self.signal_enhancer.enhance_subject_data(
                subject_id, self.dataset, self.root_data
            )
            
            if enhanced_df is not None:
                # 保存增强后的数据
                output_file = os.path.join(self.root_data, 
                                         f'data/restimulus/refilter_enhanced/{self.dataset}_s{subject_id}filter_enhanced.h5')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                enhanced_df.to_hdf(output_file, format='table', key='df', mode='w', complevel=9, complib='blosc')
                print(f"增强数据已保存: {output_file}")
                
                enhanced_data[subject_id] = {
                    'enhanced_df': enhanced_df,
                    'original_quality': original_quality,
                    'enhanced_quality': enhanced_quality
                }
                
                # 可视化增强效果
                raw_file = os.path.join(self.root_data, f'data/restimulus/reraw/{self.dataset}_s{subject_id}raw.h5')
                raw_df = pd.read_hdf(raw_file, 'df')
                emg_columns = [col for col in raw_df.columns if col not in ['restimulus', 'rerepetition']]
                raw_emg = raw_df.iloc[:, :len(emg_columns)].values
                
                self.signal_enhancer.visualize_enhancement(
                    subject_id, raw_emg, enhanced_df.iloc[:, :len(emg_columns)].values,
                    original_quality, enhanced_quality,
                    save_path=f'improvement/subject_{subject_id}_enhancement.png'
                )
        
        return enhanced_data
    
    def select_robust_features(self, subjects):
        """选择鲁棒特征"""
        print("加载受试者特征数据...")
        all_features = []
        all_labels = []
        
        for subject_id in subjects:
            features, labels = load_subject_features(subject_id, self.dataset, self.root_data)
            if features is not None:
                all_features.append(features)
                all_labels.append(labels)
                print(f"受试者 {subject_id}: {features.shape}")
        
        if not all_features:
            print("没有可用的特征数据")
            return None
        
        # 合并所有受试者的数据
        combined_features = np.vstack(all_features)
        combined_labels = np.concatenate(all_labels)
        
        print(f"\n合并后数据形状: {combined_features.shape}")
        print(f"标签分布: {np.bincount(combined_labels)}")
        
        # 执行特征选择
        selected_indices = self.feature_selector.select_features(combined_features, combined_labels)
        
        # 生成报告
        self.feature_selector.generate_feature_report()
        
        # 可视化结果
        self.feature_selector.visualize_feature_selection(
            features=combined_features, 
            save_path='improvement/robust_feature_selection.png'
        )
        
        # 保存选中的特征索引
        np.save('improvement/selected_feature_indices.npy', selected_indices)
        print(f"选中特征索引已保存: improvement/selected_feature_indices.npy")
        
        return selected_indices
    
    def evaluate_improvements(self, subjects, enhanced_data, selected_features):
        """评估改进效果"""
        print("评估改进效果...")
        
        for subject_id in subjects:
            if subject_id in enhanced_data:
                print(f"\n受试者 {subject_id} 改进效果:")
                
                original_quality = enhanced_data[subject_id]['original_quality']
                enhanced_quality = enhanced_data[subject_id]['enhanced_quality']
                
                # 计算质量提升
                if original_quality and enhanced_quality:
                    original_scores = [metrics['quality_score'] for metrics in original_quality.values()]
                    enhanced_scores = [metrics['quality_score'] for metrics in enhanced_quality.values()]
                    
                    avg_original = np.mean(original_scores)
                    avg_enhanced = np.mean(enhanced_scores)
                    improvement = (avg_enhanced / avg_original - 1) * 100
                    
                    print(f"  信号质量评分: {avg_original:.3f} → {avg_enhanced:.3f} (提升 {improvement:+.1f}%)")
                    
                    # SNR提升
                    original_snr = [metrics['snr'] for metrics in original_quality.values()]
                    enhanced_snr = [metrics['snr'] for metrics in enhanced_quality.values()]
                    
                    avg_original_snr = np.mean(original_snr)
                    avg_enhanced_snr = np.mean(enhanced_snr)
                    snr_improvement = (avg_enhanced_snr / avg_original_snr - 1) * 100
                    
                    print(f"  平均SNR: {avg_original_snr:.1f}dB → {avg_enhanced_snr:.1f}dB (提升 {snr_improvement:+.1f}%)")
    
    def generate_improvement_report(self, subjects, enhanced_data, selected_features):
        """生成改进报告"""
        report_file = 'improvement/improvement_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# EMG信号改进报告\n\n")
            f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📋 改进概述\n\n")
            f.write("本报告总结了针对低准确率受试者的EMG信号改进措施和效果。\n\n")
            
            f.write("## 🔧 改进措施\n\n")
            f.write("### 1. 自适应信号增强\n\n")
            f.write("- **目标**: 提升信号质量，减少噪声干扰\n")
            f.write("- **方法**: 根据信号质量自动选择滤波策略\n")
            f.write("- **特点**: 强滤波、中等滤波、轻微滤波三种策略\n\n")
            
            f.write("### 2. 鲁棒特征选择\n\n")
            f.write("- **目标**: 选择最稳定的特征子集\n")
            f.write("- **方法**: 多维度特征评估（稳定性、区分度、冗余度、鲁棒性）\n")
            f.write("- **特点**: 综合考虑特征的各种性能指标\n\n")
            
            f.write("## 📊 改进效果\n\n")
            
            for subject_id in subjects:
                f.write(f"### 受试者 {subject_id}\n\n")
                
                if subject_id in enhanced_data:
                    original_quality = enhanced_data[subject_id]['original_quality']
                    enhanced_quality = enhanced_data[subject_id]['enhanced_quality']
                    
                    if original_quality and enhanced_quality:
                        # 计算改进效果
                        original_scores = [metrics['quality_score'] for metrics in original_quality.values()]
                        enhanced_scores = [metrics['quality_score'] for metrics in enhanced_quality.values()]
                        
                        avg_original = np.mean(original_scores)
                        avg_enhanced = np.mean(enhanced_scores)
                        improvement = (avg_enhanced / avg_original - 1) * 100
                        
                        f.write(f"- **信号质量评分**: {avg_original:.3f} → {avg_enhanced:.3f} (提升 {improvement:+.1f}%)\n")
                        
                        # SNR提升
                        original_snr = [metrics['snr'] for metrics in original_quality.values()]
                        enhanced_snr = [metrics['snr'] for metrics in enhanced_quality.values()]
                        
                        avg_original_snr = np.mean(original_snr)
                        avg_enhanced_snr = np.mean(enhanced_snr)
                        snr_improvement = (avg_enhanced_snr / avg_original_snr - 1) * 100
                        
                        f.write(f"- **平均SNR**: {avg_original_snr:.1f}dB → {avg_enhanced_snr:.1f}dB (提升 {snr_improvement:+.1f}%)\n")
                
                f.write("\n")
            
            f.write("## 🎯 特征选择结果\n\n")
            if selected_features is not None:
                f.write(f"- **选中特征数量**: {len(selected_features)}\n")
                f.write(f"- **特征索引**: {selected_features.tolist()}\n")
                f.write("- **选择标准**: 稳定性、区分度、冗余度、鲁棒性综合评分\n\n")
            
            f.write("## 📈 预期效果\n\n")
            f.write("通过以上改进措施，预期能够：\n\n")
            f.write("1. **提升信号质量**: 减少噪声干扰，提高信噪比\n")
            f.write("2. **增强特征稳定性**: 选择最稳定的特征，减少变异性\n")
            f.write("3. **提高模型鲁棒性**: 对不稳定信号有更好的适应性\n")
            f.write("4. **改善分类准确率**: 特别是低准确率受试者的性能\n\n")
            
            f.write("## 🔄 后续步骤\n\n")
            f.write("1. 使用增强后的信号重新进行特征提取\n")
            f.write("2. 使用选中的特征子集重新训练模型\n")
            f.write("3. 评估改进后的模型性能\n")
            f.write("4. 根据结果进一步优化参数\n\n")
            
            f.write("---\n")
            f.write("*报告生成完成*\n")
        
        print(f"改进报告已生成: {report_file}")
    
    def create_improvement_summary(self):
        """创建改进总结"""
        summary_file = 'improvement/improvement_summary.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("EMG信号改进总结\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("改进目标:\n")
            f.write("- 提升低准确率受试者的信号质量\n")
            f.write("- 选择最稳定的特征子集\n")
            f.write("- 提高模型对不稳定信号的鲁棒性\n\n")
            
            f.write("改进措施:\n")
            f.write("1. 自适应信号增强\n")
            f.write("   - 根据信号质量自动选择滤波策略\n")
            f.write("   - 强滤波、中等滤波、轻微滤波\n")
            f.write("   - 异常值检测和去除\n\n")
            
            f.write("2. 鲁棒特征选择\n")
            f.write("   - 多维度特征评估\n")
            f.write("   - 稳定性、区分度、冗余度、鲁棒性\n")
            f.write("   - 综合评分选择最优特征\n\n")
            
            f.write("预期效果:\n")
            f.write("- 信号质量提升 20-50%\n")
            f.write("- 特征稳定性提升 30-60%\n")
            f.write("- 模型准确率提升 5-15%\n\n")
            
            f.write("文件输出:\n")
            f.write("- 增强后的信号数据: D:/DB4/data/restimulus/refilter_enhanced/\n")
            f.write("- 选中特征索引: improvement/selected_feature_indices.npy\n")
            f.write("- 可视化结果: improvement/*.png\n")
            f.write("- 详细报告: improvement/improvement_report.md\n")
        
        print(f"改进总结已生成: {summary_file}")

def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path("improvement")
    output_dir.mkdir(exist_ok=True)
    
    # 创建改进流程
    pipeline = CompleteImprovementPipeline(dataset='DB4', root_data='D:/DB4')
    
    # 运行完整改进流程
    pipeline.run_complete_pipeline(low_perf_subjects=[7, 8], high_perf_subjects=[10])
    
    # 创建改进总结
    pipeline.create_improvement_summary()
    
    print("\n🎉 所有改进措施已完成！")
    print("\n📁 生成的文件:")
    print("  - 增强信号数据: D:/DB4/data/restimulus/refilter_enhanced/")
    print("  - 特征选择结果: improvement/selected_feature_indices.npy")
    print("  - 可视化图表: improvement/*.png")
    print("  - 详细报告: improvement/improvement_report.md")
    print("  - 改进总结: improvement/improvement_summary.txt")
    
    print("\n🚀 下一步建议:")
    print("  1. 使用增强后的信号重新运行特征提取")
    print("  2. 使用选中的特征子集重新训练模型")
    print("  3. 评估改进后的模型性能")

if __name__ == "__main__":
    main() 