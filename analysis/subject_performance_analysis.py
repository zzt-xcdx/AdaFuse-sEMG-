#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
受试者性能分析脚本
分析受试者7和8准确率低的原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
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
import nina_funcs as nf

class SubjectPerformanceAnalyzer:
    """受试者性能分析器"""
    
    def __init__(self, dataset='DB4', root_data='D:/DB4'):
        self.dataset = dataset
        self.root_data = root_data
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def load_subject_data(self, subject_id):
        """加载受试者数据"""
        print(f"加载受试者 {subject_id} 的数据...")
        
        # 加载原始数据
        raw_file = os.path.join(self.root_data, f'data/restimulus/reraw/{self.dataset}_s{subject_id}raw.h5')
        if os.path.exists(raw_file):
            raw_df = pd.read_hdf(raw_file, 'df')
            print(f"原始数据形状: {raw_df.shape}")
        else:
            raw_df = None
            print("原始数据文件不存在")
        
        # 加载滤波后数据
        filtered_file = os.path.join(self.root_data, f'data/restimulus/refilter/{self.dataset}_s{subject_id}filter.h5')
        if os.path.exists(filtered_file):
            filtered_df = pd.read_hdf(filtered_file, 'df')
            print(f"滤波后数据形状: {filtered_df.shape}")
        else:
            filtered_df = None
            print("滤波后数据文件不存在")
        
        # 加载特征数据
        feature_file = os.path.join(self.root_data, f'data/restimulus/Fea/{self.dataset}_s{subject_id}feaEnhance.h5')
        if os.path.exists(feature_file):
            with h5py.File(feature_file, 'r') as f:
                features = f['features'][:]
                labels = f['label'][:]
            print(f"特征数据形状: {features.shape}")
        else:
            features = None
            labels = None
            print("特征数据文件不存在")
        
        return raw_df, filtered_df, features, labels
    
    def analyze_signal_quality(self, raw_df, filtered_df, subject_id):
        """分析信号质量"""
        print(f"\n分析受试者 {subject_id} 的信号质量...")
        
        if raw_df is None or filtered_df is None:
            print("缺少原始或滤波数据，跳过信号质量分析")
            return
        
        # 获取EMG通道数量
        emg_columns = [col for col in raw_df.columns if col not in ['restimulus', 'rerepetition']]
        emg_channel_count = len(emg_columns)
        
        # 选择几个通道进行分析
        channels_to_analyze = min(4, emg_channel_count)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'受试者 {subject_id} 信号质量分析', fontsize=16, fontweight='bold')
        
        for i in range(channels_to_analyze):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # 获取原始和滤波信号
            raw_signal = raw_df.iloc[:2000, i].values  # 取前2000个采样点
            filtered_signal = filtered_df.iloc[:2000, i].values
            
            # 计算信号质量指标
            raw_rms = np.sqrt(np.mean(raw_signal**2))
            filtered_rms = np.sqrt(np.mean(filtered_signal**2))
            raw_snr = 20 * np.log10(np.std(raw_signal) / np.std(raw_signal - np.mean(raw_signal)))
            
            # 绘制时域信号
            time = np.arange(len(raw_signal)) / 2000  # 采样率2000Hz
            ax.plot(time, raw_signal, alpha=0.7, label=f'原始信号 (RMS: {raw_rms:.3f})')
            ax.plot(time, filtered_signal, alpha=0.7, label=f'滤波信号 (RMS: {filtered_rms:.3f})')
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('幅度')
            ax.set_title(f'通道 {i+1} (SNR: {raw_snr:.1f} dB)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'analysis/subject_{subject_id}_signal_quality.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算整体信号质量指标
        quality_metrics = {}
        for i in range(emg_channel_count):
            raw_signal = raw_df.iloc[:, i].values
            filtered_signal = filtered_df.iloc[:, i].values
            
            # RMS值
            quality_metrics[f'ch{i+1}_raw_rms'] = np.sqrt(np.mean(raw_signal**2))
            quality_metrics[f'ch{i+1}_filtered_rms'] = np.sqrt(np.mean(filtered_signal**2))
            
            # 信噪比
            quality_metrics[f'ch{i+1}_snr'] = 20 * np.log10(np.std(raw_signal) / np.std(raw_signal - np.mean(raw_signal)))
            
            # 基线漂移
            baseline = np.mean(raw_signal)
            quality_metrics[f'ch{i+1}_baseline'] = baseline
        
        return quality_metrics
    
    def analyze_gesture_performance(self, features, labels, subject_id):
        """分析手势表现"""
        print(f"\n分析受试者 {subject_id} 的手势表现...")
        
        if features is None or labels is None:
            print("缺少特征或标签数据，跳过手势表现分析")
            return
        
        # 统计每个手势的样本数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # 创建手势表现分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'受试者 {subject_id} 手势表现分析', fontsize=16, fontweight='bold')
        
        # 1. 手势样本分布
        ax1 = axes[0, 0]
        ax1.bar(unique_labels, counts, color=self.colors[0], alpha=0.7)
        ax1.set_xlabel('手势编号')
        ax1.set_ylabel('样本数量')
        ax1.set_title('手势样本分布')
        ax1.grid(True, alpha=0.3)
        
        # 2. 特征分布
        ax2 = axes[0, 1]
        # 选择前几个主要特征进行可视化
        n_features_to_show = min(10, features.shape[1])
        feature_means = np.mean(features[:, :n_features_to_show], axis=0)
        feature_stds = np.std(features[:, :n_features_to_show], axis=0)
        
        ax2.errorbar(range(n_features_to_show), feature_means, yerr=feature_stds, 
                    fmt='o-', color=self.colors[1], alpha=0.7)
        ax2.set_xlabel('特征编号')
        ax2.set_ylabel('特征值')
        ax2.set_title('主要特征分布')
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征相关性热力图
        ax3 = axes[1, 0]
        # 计算特征间的相关性
        feature_corr = np.corrcoef(features[:, :n_features_to_show].T)
        im = ax3.imshow(feature_corr, cmap='coolwarm', aspect='auto')
        ax3.set_title('特征相关性热力图')
        plt.colorbar(im, ax=ax3)
        
        # 4. 手势复杂度分析
        ax4 = axes[1, 1]
        # 计算每个手势的特征复杂度（标准差）
        gesture_complexity = []
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                gesture_features = features[mask]
                complexity = np.mean(np.std(gesture_features, axis=0))
                gesture_complexity.append(complexity)
            else:
                gesture_complexity.append(0)
        
        ax4.bar(unique_labels, gesture_complexity, color=self.colors[2], alpha=0.7)
        ax4.set_xlabel('手势编号')
        ax4.set_ylabel('特征复杂度')
        ax4.set_title('手势复杂度分析')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'analysis/subject_{subject_id}_gesture_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'gesture_counts': dict(zip(unique_labels, counts)),
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'gesture_complexity': dict(zip(unique_labels, gesture_complexity))
        }
    
    def compare_subjects(self, subject_ids=[7, 8]):
        """比较受试者表现"""
        print(f"\n比较受试者 {subject_ids} 的表现...")
        
        comparison_data = {}
        
        for subject_id in subject_ids:
            print(f"\n{'='*50}")
            print(f"分析受试者 {subject_id}")
            print(f"{'='*50}")
            
            # 加载数据
            raw_df, filtered_df, features, labels = self.load_subject_data(subject_id)
            
            # 分析信号质量
            quality_metrics = self.analyze_signal_quality(raw_df, filtered_df, subject_id)
            
            # 分析手势表现
            gesture_metrics = self.analyze_gesture_performance(features, labels, subject_id)
            
            comparison_data[subject_id] = {
                'quality_metrics': quality_metrics,
                'gesture_metrics': gesture_metrics
            }
        
        # 创建比较图
        self.create_comparison_plots(comparison_data, subject_ids)
        
        return comparison_data
    
    def create_comparison_plots(self, comparison_data, subject_ids):
        """创建比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('受试者表现比较', fontsize=16, fontweight='bold')
        
        # 1. 信号质量比较
        ax1 = axes[0, 0]
        avg_snr_values = []
        for subject_id in subject_ids:
            quality_metrics = comparison_data[subject_id]['quality_metrics']
            snr_values = [v for k, v in quality_metrics.items() if 'snr' in k]
            avg_snr = np.mean(snr_values) if snr_values else 0
            avg_snr_values.append(avg_snr)
        
        ax1.bar(subject_ids, avg_snr_values, color=self.colors[:len(subject_ids)], alpha=0.7)
        ax1.set_xlabel('受试者编号')
        ax1.set_ylabel('平均信噪比 (dB)')
        ax1.set_title('信号质量比较')
        ax1.grid(True, alpha=0.3)
        
        # 2. 手势复杂度比较
        ax2 = axes[0, 1]
        for i, subject_id in enumerate(subject_ids):
            gesture_metrics = comparison_data[subject_id]['gesture_metrics']
            if 'gesture_complexity' in gesture_metrics:
                complexities = list(gesture_metrics['gesture_complexity'].values())
                ax2.plot(range(len(complexities)), complexities, 
                        marker='o', label=f'受试者{subject_id}', 
                        color=self.colors[i], alpha=0.7)
        
        ax2.set_xlabel('手势编号')
        ax2.set_ylabel('复杂度')
        ax2.set_title('手势复杂度比较')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 样本分布比较
        ax3 = axes[1, 0]
        for i, subject_id in enumerate(subject_ids):
            gesture_metrics = comparison_data[subject_id]['gesture_metrics']
            if 'gesture_counts' in gesture_metrics:
                counts = list(gesture_metrics['gesture_counts'].values())
                ax3.plot(range(len(counts)), counts, 
                        marker='s', label=f'受试者{subject_id}', 
                        color=self.colors[i], alpha=0.7)
        
        ax3.set_xlabel('手势编号')
        ax3.set_ylabel('样本数量')
        ax3.set_title('样本分布比较')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征稳定性比较
        ax4 = axes[1, 1]
        for i, subject_id in enumerate(subject_ids):
            gesture_metrics = comparison_data[subject_id]['gesture_metrics']
            if 'feature_stds' in gesture_metrics:
                feature_stds = gesture_metrics['feature_stds']
                ax4.plot(range(len(feature_stds)), feature_stds, 
                        marker='^', label=f'受试者{subject_id}', 
                        color=self.colors[i], alpha=0.7)
        
        ax4.set_xlabel('特征编号')
        ax4.set_ylabel('特征标准差')
        ax4.set_title('特征稳定性比较')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/subjects_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, comparison_data, subject_ids):
        """生成分析报告"""
        print("\n" + "="*60)
        print("受试者性能分析报告")
        print("="*60)
        
        for subject_id in subject_ids:
            print(f"\n受试者 {subject_id} 分析结果:")
            print("-" * 40)
            
            quality_metrics = comparison_data[subject_id]['quality_metrics']
            gesture_metrics = comparison_data[subject_id]['gesture_metrics']
            
            # 信号质量总结
            snr_values = [v for k, v in quality_metrics.items() if 'snr' in k]
            avg_snr = np.mean(snr_values) if snr_values else 0
            print(f"平均信噪比: {avg_snr:.2f} dB")
            
            # 手势表现总结
            if 'gesture_counts' in gesture_metrics:
                total_samples = sum(gesture_metrics['gesture_counts'].values())
                print(f"总样本数: {total_samples}")
                
                # 找出样本最少的手势
                min_samples_gesture = min(gesture_metrics['gesture_counts'].items(), key=lambda x: x[1])
                print(f"样本最少的手势: {min_samples_gesture[0]} (仅{min_samples_gesture[1]}个样本)")
            
            # 特征稳定性
            if 'feature_stds' in gesture_metrics:
                avg_feature_std = np.mean(gesture_metrics['feature_stds'])
                print(f"平均特征标准差: {avg_feature_std:.4f}")
        
        print("\n" + "="*60)
        print("建议改进措施:")
        print("="*60)
        print("1. 检查信号采集质量，确保电极接触良好")
        print("2. 增加样本数量不足的手势的训练数据")
        print("3. 考虑使用数据增强技术")
        print("4. 调整模型参数或使用更适合的模型架构")
        print("5. 检查是否存在手势执行不规范的问题")

def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 创建分析器
    analyzer = SubjectPerformanceAnalyzer(dataset='DB4', root_data='D:/DB4')
    
    # 分析受试者7和8
    comparison_data = analyzer.compare_subjects(subject_ids=[7, 8])
    
    # 生成报告
    analyzer.generate_report(comparison_data, [7, 8])
    
    print("\n分析完成！请查看生成的图片和报告。")

if __name__ == "__main__":
    main() 