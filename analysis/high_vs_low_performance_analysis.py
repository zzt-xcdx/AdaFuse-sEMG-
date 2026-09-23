#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高准确率 vs 低准确率受试者对比分析
对比受试者10（高准确率）和受试者7、8（低准确率）的差异
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

class PerformanceComparisonAnalyzer:
    """性能对比分析器"""
    
    def __init__(self, dataset='DB4', root_data='D:/DB4'):
        self.dataset = dataset
        self.root_data = root_data
        self.colors = {
            'high_perf': '#2ca02c',    # 绿色 - 高准确率
            'low_perf': '#d62728',     # 红色 - 低准确率
            'medium_perf': '#ff7f0e'   # 橙色 - 中等准确率
        }
        
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
    
    def calculate_signal_metrics(self, raw_df, filtered_df, subject_id):
        """计算信号质量指标"""
        if raw_df is None or filtered_df is None:
            return None
        
        # 获取EMG通道数量
        emg_columns = [col for col in raw_df.columns if col not in ['restimulus', 'rerepetition']]
        emg_channel_count = len(emg_columns)
        
        metrics = {}
        
        for i in range(emg_channel_count):
            raw_signal = raw_df.iloc[:, i].values
            filtered_signal = filtered_df.iloc[:, i].values
            
            # RMS值
            metrics[f'ch{i+1}_raw_rms'] = np.sqrt(np.mean(raw_signal**2))
            metrics[f'ch{i+1}_filtered_rms'] = np.sqrt(np.mean(filtered_signal**2))
            
            # 信噪比（改进计算）
            signal_power = np.var(raw_signal)
            noise_power = np.var(raw_signal - np.mean(raw_signal))
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 0
            metrics[f'ch{i+1}_snr'] = snr
            
            # 基线漂移
            baseline = np.mean(raw_signal)
            metrics[f'ch{i+1}_baseline'] = baseline
            
            # 信号稳定性（变异系数）
            cv = np.std(raw_signal) / np.abs(np.mean(raw_signal)) if np.mean(raw_signal) != 0 else 0
            metrics[f'ch{i+1}_cv'] = cv
            
            # 峰值检测
            peaks = np.max(np.abs(raw_signal))
            metrics[f'ch{i+1}_peak'] = peaks
        
        return metrics
    
    def calculate_feature_metrics(self, features, labels, subject_id):
        """计算特征质量指标"""
        if features is None or labels is None:
            return None
        
        metrics = {}
        
        # 特征统计
        metrics['feature_mean'] = np.mean(features, axis=0)
        metrics['feature_std'] = np.std(features, axis=0)
        metrics['feature_cv'] = np.std(features, axis=0) / np.abs(np.mean(features, axis=0))
        
        # 手势统计
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['gesture_counts'] = dict(zip(unique_labels, counts))
        metrics['total_samples'] = len(labels)
        
        # 特征相关性
        feature_corr = np.corrcoef(features.T)
        metrics['feature_correlation'] = feature_corr
        
        # 手势复杂度
        gesture_complexity = []
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                gesture_features = features[mask]
                complexity = np.mean(np.std(gesture_features, axis=0))
                gesture_complexity.append(complexity)
            else:
                gesture_complexity.append(0)
        metrics['gesture_complexity'] = dict(zip(unique_labels, gesture_complexity))
        
        return metrics
    
    def create_comprehensive_comparison(self, high_perf_subject=10, low_perf_subjects=[7, 8]):
        """创建综合对比分析"""
        print("="*60)
        print("高准确率 vs 低准确率受试者对比分析")
        print("="*60)
        
        all_subjects = [high_perf_subject] + low_perf_subjects
        all_data = {}
        
        # 加载所有受试者数据
        for subject_id in all_subjects:
            print(f"\n处理受试者 {subject_id}...")
            raw_df, filtered_df, features, labels = self.load_subject_data(subject_id)
            
            # 计算信号指标
            signal_metrics = self.calculate_signal_metrics(raw_df, filtered_df, subject_id)
            
            # 计算特征指标
            feature_metrics = self.calculate_feature_metrics(features, labels, subject_id)
            
            all_data[subject_id] = {
                'raw_df': raw_df,
                'filtered_df': filtered_df,
                'features': features,
                'labels': labels,
                'signal_metrics': signal_metrics,
                'feature_metrics': feature_metrics
            }
        
        # 创建对比图表
        self.create_comparison_plots(all_data, high_perf_subject, low_perf_subjects)
        
        # 生成详细报告
        self.generate_detailed_report(all_data, high_perf_subject, low_perf_subjects)
        
        return all_data
    
    def create_comparison_plots(self, all_data, high_perf_subject, low_perf_subjects):
        """创建对比图表"""
        
        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('高准确率 vs 低准确率受试者综合对比', fontsize=18, fontweight='bold')
        
        # 设置子图布局
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. 信号质量对比
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_signal_quality_comparison(ax1, all_data, high_perf_subject, low_perf_subjects)
        
        # 2. 特征稳定性对比
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_feature_stability_comparison(ax2, all_data, high_perf_subject, low_perf_subjects)
        
        # 3. 手势复杂度对比
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_gesture_complexity_comparison(ax3, all_data, high_perf_subject, low_perf_subjects)
        
        # 4. 样本分布对比
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_sample_distribution_comparison(ax4, all_data, high_perf_subject, low_perf_subjects)
        
        # 5. 特征相关性对比
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_feature_correlation_comparison(ax5, all_data, high_perf_subject, low_perf_subjects)
        
        # 6. 信号时域对比
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_time_domain_comparison(ax6, all_data, high_perf_subject, low_perf_subjects)
        
        # 7. 特征分布对比
        ax7 = fig.add_subplot(gs[2, :])
        self.plot_feature_distribution_comparison(ax7, all_data, high_perf_subject, low_perf_subjects)
        
        # 8. 关键指标雷达图
        ax8 = fig.add_subplot(gs[3, :])
        self.plot_radar_comparison(ax8, all_data, high_perf_subject, low_perf_subjects)
        
        plt.tight_layout()
        plt.savefig('analysis/high_vs_low_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_signal_quality_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """信号质量对比"""
        subjects = [high_perf_subject] + low_perf_subjects
        avg_snr_values = []
        avg_rms_values = []
        
        for subject_id in subjects:
            signal_metrics = all_data[subject_id]['signal_metrics']
            if signal_metrics:
                snr_values = [v for k, v in signal_metrics.items() if 'snr' in k]
                rms_values = [v for k, v in signal_metrics.items() if 'raw_rms' in k]
                avg_snr_values.append(np.mean(snr_values) if snr_values else 0)
                avg_rms_values.append(np.mean(rms_values) if rms_values else 0)
            else:
                avg_snr_values.append(0)
                avg_rms_values.append(0)
        
        x = np.arange(len(subjects))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, avg_snr_values, width, label='平均信噪比', 
                      color=self.colors['high_perf'], alpha=0.7)
        bars2 = ax.bar(x + width/2, avg_rms_values, width, label='平均RMS', 
                      color=self.colors['low_perf'], alpha=0.7)
        
        ax.set_xlabel('受试者编号')
        ax.set_ylabel('数值')
        ax.set_title('信号质量对比')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    def plot_feature_stability_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """特征稳定性对比"""
        subjects = [high_perf_subject] + low_perf_subjects
        colors = [self.colors['high_perf']] + [self.colors['low_perf']] * len(low_perf_subjects)
        
        for i, subject_id in enumerate(subjects):
            feature_metrics = all_data[subject_id]['feature_metrics']
            if feature_metrics and 'feature_std' in feature_metrics:
                feature_stds = feature_metrics['feature_std']
                ax.plot(range(len(feature_stds)), feature_stds, 
                       marker='o', label=f'受试者{subject_id}', 
                       color=colors[i], alpha=0.7)
        
        ax.set_xlabel('特征编号')
        ax.set_ylabel('特征标准差')
        ax.set_title('特征稳定性对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_gesture_complexity_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """手势复杂度对比"""
        subjects = [high_perf_subject] + low_perf_subjects
        colors = [self.colors['high_perf']] + [self.colors['low_perf']] * len(low_perf_subjects)
        
        for i, subject_id in enumerate(subjects):
            feature_metrics = all_data[subject_id]['feature_metrics']
            if feature_metrics and 'gesture_complexity' in feature_metrics:
                complexities = list(feature_metrics['gesture_complexity'].values())
                ax.plot(range(len(complexities)), complexities, 
                       marker='s', label=f'受试者{subject_id}', 
                       color=colors[i], alpha=0.7)
        
        ax.set_xlabel('手势编号')
        ax.set_ylabel('复杂度')
        ax.set_title('手势复杂度对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_sample_distribution_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """样本分布对比"""
        subjects = [high_perf_subject] + low_perf_subjects
        colors = [self.colors['high_perf']] + [self.colors['low_perf']] * len(low_perf_subjects)
        
        for i, subject_id in enumerate(subjects):
            feature_metrics = all_data[subject_id]['feature_metrics']
            if feature_metrics and 'gesture_counts' in feature_metrics:
                counts = list(feature_metrics['gesture_counts'].values())
                ax.plot(range(len(counts)), counts, 
                       marker='^', label=f'受试者{subject_id}', 
                       color=colors[i], alpha=0.7)
        
        ax.set_xlabel('手势编号')
        ax.set_ylabel('样本数量')
        ax.set_title('样本分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_feature_correlation_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """特征相关性对比"""
        # 选择第一个低准确率受试者进行对比
        low_perf_subject = low_perf_subjects[0]
        
        high_corr = all_data[high_perf_subject]['feature_metrics']['feature_correlation']
        low_corr = all_data[low_perf_subject]['feature_metrics']['feature_correlation']
        
        # 计算相关性差异
        corr_diff = np.abs(high_corr - low_corr)
        
        im = ax.imshow(corr_diff, cmap='hot', aspect='auto')
        ax.set_title(f'特征相关性差异\n(受试者{high_perf_subject} vs {low_perf_subject})')
        plt.colorbar(im, ax=ax)
    
    def plot_time_domain_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """时域信号对比"""
        # 选择第一个低准确率受试者进行对比
        low_perf_subject = low_perf_subjects[0]
        
        # 获取第一个通道的信号
        high_signal = all_data[high_perf_subject]['filtered_df'].iloc[:1000, 0].values
        low_signal = all_data[low_perf_subject]['filtered_df'].iloc[:1000, 0].values
        
        time = np.arange(1000) / 2000  # 采样率2000Hz
        
        ax.plot(time, high_signal, label=f'受试者{high_perf_subject}', 
               color=self.colors['high_perf'], alpha=0.7)
        ax.plot(time, low_signal, label=f'受试者{low_perf_subject}', 
               color=self.colors['low_perf'], alpha=0.7)
        
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('幅度')
        ax.set_title('时域信号对比 (通道1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_feature_distribution_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """特征分布对比"""
        subjects = [high_perf_subject] + low_perf_subjects
        colors = [self.colors['high_perf']] + [self.colors['low_perf']] * len(low_perf_subjects)
        
        # 选择前几个主要特征
        n_features = 5
        
        for i, subject_id in enumerate(subjects):
            features = all_data[subject_id]['features']
            if features is not None:
                for j in range(n_features):
                    ax.hist(features[:, j], bins=30, alpha=0.3, 
                           label=f'受试者{subject_id}-特征{j+1}', 
                           color=colors[i])
        
        ax.set_xlabel('特征值')
        ax.set_ylabel('频次')
        ax.set_title('特征分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_radar_comparison(self, ax, all_data, high_perf_subject, low_perf_subjects):
        """雷达图对比"""
        # 计算关键指标
        categories = ['信号质量', '特征稳定性', '手势一致性', '样本平衡性', '特征区分度']
        
        # 计算各指标
        high_metrics = self.calculate_radar_metrics(all_data[high_perf_subject])
        low_metrics = []
        for subject_id in low_perf_subjects:
            low_metrics.append(self.calculate_radar_metrics(all_data[subject_id]))
        
        # 计算低准确率组的平均值
        avg_low_metrics = np.mean(low_metrics, axis=0)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        high_metrics = np.concatenate((high_metrics, [high_metrics[0]]))
        avg_low_metrics = np.concatenate((avg_low_metrics, [avg_low_metrics[0]]))
        
        ax.plot(angles, high_metrics, 'o-', linewidth=2, label=f'受试者{high_perf_subject}', 
               color=self.colors['high_perf'])
        ax.fill(angles, high_metrics, alpha=0.25, color=self.colors['high_perf'])
        
        ax.plot(angles, avg_low_metrics, 'o-', linewidth=2, label='低准确率组平均', 
               color=self.colors['low_perf'])
        ax.fill(angles, avg_low_metrics, alpha=0.25, color=self.colors['low_perf'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('关键指标雷达图对比')
        ax.legend()
        ax.grid(True)
    
    def calculate_radar_metrics(self, subject_data):
        """计算雷达图指标"""
        metrics = []
        
        # 1. 信号质量 (基于SNR)
        signal_metrics = subject_data['signal_metrics']
        if signal_metrics:
            snr_values = [v for k, v in signal_metrics.items() if 'snr' in k]
            avg_snr = np.mean(snr_values) if snr_values else 0
            # 归一化到0-1
            signal_quality = min(1.0, max(0.0, (avg_snr + 20) / 40))  # 假设SNR范围-20到20dB
        else:
            signal_quality = 0
        metrics.append(signal_quality)
        
        # 2. 特征稳定性 (基于特征标准差)
        feature_metrics = subject_data['feature_metrics']
        if feature_metrics and 'feature_std' in feature_metrics:
            feature_stds = feature_metrics['feature_std']
            avg_std = np.mean(feature_stds)
            # 归一化到0-1 (标准差越小越稳定)
            feature_stability = max(0.0, min(1.0, 1.0 - avg_std / 10))
        else:
            feature_stability = 0
        metrics.append(feature_stability)
        
        # 3. 手势一致性 (基于手势复杂度)
        if feature_metrics and 'gesture_complexity' in feature_metrics:
            complexities = list(feature_metrics['gesture_complexity'].values())
            avg_complexity = np.mean(complexities)
            # 归一化到0-1 (复杂度越低越一致)
            gesture_consistency = max(0.0, min(1.0, 1.0 - avg_complexity / 15))
        else:
            gesture_consistency = 0
        metrics.append(gesture_consistency)
        
        # 4. 样本平衡性 (基于样本分布)
        if feature_metrics and 'gesture_counts' in feature_metrics:
            counts = list(feature_metrics['gesture_counts'].values())
            # 计算变异系数
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
            # 归一化到0-1 (变异系数越小越平衡)
            sample_balance = max(0.0, min(1.0, 1.0 - cv))
        else:
            sample_balance = 0
        metrics.append(sample_balance)
        
        # 5. 特征区分度 (基于特征相关性)
        if feature_metrics and 'feature_correlation' in feature_metrics:
            corr_matrix = feature_metrics['feature_correlation']
            # 计算平均相关性
            n_features = corr_matrix.shape[0]
            total_corr = np.sum(np.abs(corr_matrix)) - n_features  # 减去对角线
            avg_corr = total_corr / (n_features * (n_features - 1))
            # 归一化到0-1 (相关性越低区分度越高)
            feature_discrimination = max(0.0, min(1.0, 1.0 - avg_corr))
        else:
            feature_discrimination = 0
        metrics.append(feature_discrimination)
        
        return np.array(metrics)
    
    def generate_detailed_report(self, all_data, high_perf_subject, low_perf_subjects):
        """生成详细报告"""
        print("\n" + "="*80)
        print("高准确率 vs 低准确率受试者详细对比报告")
        print("="*80)
        
        # 高准确率受试者分析
        print(f"\n📈 高准确率受试者 {high_perf_subject} 特点:")
        print("-" * 50)
        self.analyze_subject_performance(all_data[high_perf_subject], high_perf_subject)
        
        # 低准确率受试者分析
        for subject_id in low_perf_subjects:
            print(f"\n📉 低准确率受试者 {subject_id} 特点:")
            print("-" * 50)
            self.analyze_subject_performance(all_data[subject_id], subject_id)
        
        # 关键差异总结
        print(f"\n🔍 关键差异总结:")
        print("-" * 50)
        self.summarize_key_differences(all_data, high_perf_subject, low_perf_subjects)
        
        # 改进建议
        print(f"\n💡 针对低准确率受试者的改进建议:")
        print("-" * 50)
        self.provide_improvement_suggestions(all_data, low_perf_subjects)
    
    def analyze_subject_performance(self, subject_data, subject_id):
        """分析单个受试者的表现"""
        signal_metrics = subject_data['signal_metrics']
        feature_metrics = subject_data['feature_metrics']
        
        if signal_metrics:
            # 信号质量分析
            snr_values = [v for k, v in signal_metrics.items() if 'snr' in k]
            rms_values = [v for k, v in signal_metrics.items() if 'raw_rms' in k]
            peak_values = [v for k, v in signal_metrics.items() if 'peak' in k]
            
            avg_snr = np.mean(snr_values) if snr_values else 0
            avg_rms = np.mean(rms_values) if rms_values else 0
            max_peak = np.max(peak_values) if peak_values else 0
            
            print(f"信号质量:")
            print(f"  平均信噪比: {avg_snr:.2f} dB")
            print(f"  平均RMS: {avg_rms:.2f}")
            print(f"  最大峰值: {max_peak:.2f}")
        
        if feature_metrics:
            # 特征质量分析
            if 'feature_std' in feature_metrics:
                avg_feature_std = np.mean(feature_metrics['feature_std'])
                print(f"特征稳定性:")
                print(f"  平均特征标准差: {avg_feature_std:.4f}")
            
            if 'gesture_complexity' in feature_metrics:
                complexities = list(feature_metrics['gesture_complexity'].values())
                avg_complexity = np.mean(complexities)
                max_complexity = np.max(complexities)
                print(f"手势复杂度:")
                print(f"  平均复杂度: {avg_complexity:.2f}")
                print(f"  最大复杂度: {max_complexity:.2f}")
            
            if 'gesture_counts' in feature_metrics:
                counts = list(feature_metrics['gesture_counts'].values())
                total_samples = sum(counts)
                min_samples = min(counts)
                max_samples = max(counts)
                cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
                print(f"样本分布:")
                print(f"  总样本数: {total_samples}")
                print(f"  最少样本: {min_samples}")
                print(f"  最多样本: {max_samples}")
                print(f"  变异系数: {cv:.3f}")
    
    def summarize_key_differences(self, all_data, high_perf_subject, low_perf_subjects):
        """总结关键差异"""
        high_data = all_data[high_perf_subject]
        low_data_list = [all_data[subject_id] for subject_id in low_perf_subjects]
        
        # 信号质量差异
        high_snr = self.get_avg_snr(high_data)
        low_snr_list = [self.get_avg_snr(data) for data in low_data_list]
        avg_low_snr = np.mean(low_snr_list)
        
        print(f"1. 信号质量差异:")
        print(f"   高准确率受试者平均SNR: {high_snr:.2f} dB")
        print(f"   低准确率受试者平均SNR: {avg_low_snr:.2f} dB")
        print(f"   差异: {high_snr - avg_low_snr:+.2f} dB")
        
        # 特征稳定性差异
        high_stability = self.get_feature_stability(high_data)
        low_stability_list = [self.get_feature_stability(data) for data in low_data_list]
        avg_low_stability = np.mean(low_stability_list)
        
        print(f"\n2. 特征稳定性差异:")
        print(f"   高准确率受试者特征稳定性: {high_stability:.4f}")
        print(f"   低准确率受试者特征稳定性: {avg_low_stability:.4f}")
        print(f"   差异: {high_stability - avg_low_stability:+.4f}")
        
        # 手势复杂度差异
        high_complexity = self.get_gesture_complexity(high_data)
        low_complexity_list = [self.get_gesture_complexity(data) for data in low_data_list]
        avg_low_complexity = np.mean(low_complexity_list)
        
        print(f"\n3. 手势复杂度差异:")
        print(f"   高准确率受试者平均复杂度: {high_complexity:.2f}")
        print(f"   低准确率受试者平均复杂度: {avg_low_complexity:.2f}")
        print(f"   差异: {high_complexity - avg_low_complexity:+.2f}")
    
    def get_avg_snr(self, subject_data):
        """获取平均SNR"""
        signal_metrics = subject_data['signal_metrics']
        if signal_metrics:
            snr_values = [v for k, v in signal_metrics.items() if 'snr' in k]
            return np.mean(snr_values) if snr_values else 0
        return 0
    
    def get_feature_stability(self, subject_data):
        """获取特征稳定性"""
        feature_metrics = subject_data['feature_metrics']
        if feature_metrics and 'feature_std' in feature_metrics:
            return 1.0 - np.mean(feature_metrics['feature_std']) / 10
        return 0
    
    def get_gesture_complexity(self, subject_data):
        """获取手势复杂度"""
        feature_metrics = subject_data['feature_metrics']
        if feature_metrics and 'gesture_complexity' in feature_metrics:
            complexities = list(feature_metrics['gesture_complexity'].values())
            return np.mean(complexities)
        return 0
    
    def provide_improvement_suggestions(self, all_data, low_perf_subjects):
        """提供改进建议"""
        print("1. 信号质量改进:")
        print("   - 检查是否存在电极接触不良问题")
        print("   - 考虑使用更强的滤波方案")
        print("   - 增加信号预处理步骤")
        
        print("\n2. 特征提取改进:")
        print("   - 使用更稳定的特征提取方法")
        print("   - 考虑添加特征选择步骤")
        print("   - 尝试不同的特征组合")
        
        print("\n3. 数据质量改进:")
        print("   - 增加数据增强技术")
        print("   - 平衡不同手势的样本数量")
        print("   - 考虑重新采集数据")
        
        print("\n4. 模型改进:")
        print("   - 使用更适合不稳定信号的模型架构")
        print("   - 调整模型参数")
        print("   - 考虑使用集成学习方法")

def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 创建分析器
    analyzer = PerformanceComparisonAnalyzer(dataset='DB4', root_data='D:/DB4')
    
    # 进行对比分析
    all_data = analyzer.create_comprehensive_comparison(high_perf_subject=10, low_perf_subjects=[7, 8])
    
    print("\n分析完成！请查看生成的对比图表和详细报告。")

if __name__ == "__main__":
    main() 