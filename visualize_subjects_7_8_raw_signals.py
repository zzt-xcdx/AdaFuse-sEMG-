#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化受试者7和8的原始EMG信号
分析信号质量、噪声水平、信号强度等问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import warnings
from scipy import signal
from scipy.stats import stats
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RawSignalAnalyzer:
    """原始信号分析器"""
    
    def __init__(self, data_dir='D:/DB4'):
        self.data_dir = data_dir
        self.sampling_rate = 2000  # Hz
        self.subjects = [7, 8]
        
    def load_raw_data(self, subject_id, exercise=1):
        """加载原始EMG数据"""
        file_path = os.path.join(self.data_dir, f's{subject_id}', f'S{subject_id}_E{exercise}_A1.mat')
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
            
        try:
            mat = scio.loadmat(file_path)
            emg_data = mat['emg']  # 原始EMG信号
            stimulus = mat['stimulus'].flatten()  # 手势标签
            repetition = mat['repetition'].flatten()  # 重复次数
            
            print(f"受试者{subject_id} Exercise{exercise} 数据加载成功:")
            print(f"  EMG数据形状: {emg_data.shape}")
            print(f"  标签数量: {len(stimulus)}")
            print(f"  唯一标签: {np.unique(stimulus[stimulus > 0])}")
            print(f"  重复次数: {np.unique(repetition[repetition > 0])}")
            
            return {
                'emg': emg_data,
                'stimulus': stimulus,
                'repetition': repetition
            }
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def analyze_signal_quality(self, emg_data, subject_id, exercise):
        """分析信号质量指标"""
        print(f"\n=== 受试者{subject_id} Exercise{exercise} 信号质量分析 ===")
        
        # 计算每个通道的信号质量指标
        channels = emg_data.shape[1]
        quality_metrics = {}
        
        for ch in range(channels):
            signal_ch = emg_data[:, ch]
            
            # 信号强度指标
            rms = np.sqrt(np.mean(signal_ch**2))
            mean_abs = np.mean(np.abs(signal_ch))
            peak_to_peak = np.max(signal_ch) - np.min(signal_ch)
            
            # 噪声指标
            # 使用滑动窗口计算局部标准差作为噪声估计
            window_size = 100  # 50ms窗口
            local_std = []
            for i in range(0, len(signal_ch) - window_size, window_size//2):
                window = signal_ch[i:i+window_size]
                local_std.append(np.std(window))
            noise_level = np.mean(local_std)
            
            # 信噪比估计 (RMS / 噪声水平)
            snr_estimate = rms / (noise_level + 1e-10)
            
            # 信号复杂度 (近似熵)
            complexity = self.approximate_entropy(signal_ch[:1000], m=2, r=0.2*np.std(signal_ch))
            
            quality_metrics[ch] = {
                'rms': rms,
                'mean_abs': mean_abs,
                'peak_to_peak': peak_to_peak,
                'noise_level': noise_level,
                'snr_estimate': snr_estimate,
                'complexity': complexity
            }
            
            print(f"  通道{ch+1}: RMS={rms:.3f}, 噪声={noise_level:.3f}, SNR≈{snr_estimate:.2f}")
        
        return quality_metrics
    
    def approximate_entropy(self, data, m=2, r=0.2):
        """计算近似熵"""
        N = len(data)
        if N < m + 2:
            return 0
            
        def count_matches(template, data, r):
            count = 0
            for i in range(len(data) - len(template) + 1):
                if np.all(np.abs(data[i:i+len(template)] - template) <= r):
                    count += 1
            return count
        
        # 计算phi(m)和phi(m+1)
        phi_m = 0
        phi_m1 = 0
        
        for i in range(N - m + 1):
            template_m = data[i:i+m]
            template_m1 = data[i:i+m+1]
            
            matches_m = count_matches(template_m, data, r)
            matches_m1 = count_matches(template_m1, data, r)
            
            phi_m += np.log(matches_m / (N - m + 1))
            phi_m1 += np.log(matches_m1 / (N - m + 1))
        
        phi_m /= (N - m + 1)
        phi_m1 /= (N - m + 1)
        
        return phi_m - phi_m1
    
    def visualize_raw_signals(self, subject_id, exercise=1):
        """可视化原始信号"""
        data = self.load_raw_data(subject_id, exercise)
        if data is None:
            return
            
        emg_data = data['emg']
        stimulus = data['stimulus']
        repetition = data['repetition']
        
        # 分析信号质量
        quality_metrics = self.analyze_signal_quality(emg_data, subject_id, exercise)
        
        # 创建可视化
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f'受试者{subject_id} Exercise{exercise} 原始EMG信号分析', fontsize=16)
        
        # 绘制前16个通道的原始信号
        channels_to_plot = min(16, emg_data.shape[1])
        
        for ch in range(channels_to_plot):
            row = ch // 4
            col = ch % 4
            
            # 绘制原始信号
            time_axis = np.arange(len(emg_data[:, ch])) / self.sampling_rate
            axes[row, col].plot(time_axis, emg_data[:, ch], linewidth=0.5, alpha=0.8)
            
            # 标记手势区域
            gesture_starts = np.where(np.diff(stimulus) > 0)[0]
            gesture_ends = np.where(np.diff(stimulus) < 0)[0]
            
            for start, end in zip(gesture_starts, gesture_ends):
                if end > start:
                    axes[row, col].axvspan(start/self.sampling_rate, end/self.sampling_rate, 
                                          alpha=0.2, color='red')
            
            # 设置标题和标签
            metrics = quality_metrics[ch]
            axes[row, col].set_title(f'通道{ch+1}\nRMS: {metrics["rms"]:.2f}, SNR: {metrics["snr_estimate"]:.1f}')
            axes[row, col].set_xlabel('时间 (s)')
            axes[row, col].set_ylabel('幅值 (μV)')
            axes[row, col].grid(True, alpha=0.3)
            
            # 设置y轴范围
            y_range = np.percentile(emg_data[:, ch], [1, 99])
            axes[row, col].set_ylim(y_range[0] * 1.1, y_range[1] * 1.1)
        
        plt.tight_layout()
        plt.savefig(f'subject_{subject_id}_exercise_{exercise}_raw_signals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return quality_metrics
    
    def compare_subjects_signal_quality(self):
        """比较两个受试者的信号质量"""
        print("\n=== 比较受试者7和8的信号质量 ===")
        
        all_metrics = {}
        
        for subject_id in self.subjects:
            print(f"\n分析受试者{subject_id}:")
            subject_metrics = {}
            
            for exercise in [1, 2, 3]:
                data = self.load_raw_data(subject_id, exercise)
                if data is not None:
                    metrics = self.analyze_signal_quality(data['emg'], subject_id, exercise)
                    subject_metrics[exercise] = metrics
                    
                    # 可视化原始信号
                    self.visualize_raw_signals(subject_id, exercise)
            
            all_metrics[subject_id] = subject_metrics
        
        # 创建比较图表
        self.create_comparison_charts(all_metrics)
        
        return all_metrics
    
    def create_comparison_charts(self, all_metrics):
        """创建比较图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('受试者7和8信号质量比较', fontsize=16)
        
        # 准备数据
        subjects = list(all_metrics.keys())
        exercises = [1, 2, 3]
        
        # 1. RMS值比较
        rms_data = {subject: [] for subject in subjects}
        for subject in subjects:
            for exercise in exercises:
                if exercise in all_metrics[subject]:
                    avg_rms = np.mean([ch['rms'] for ch in all_metrics[subject][exercise].values()])
                    rms_data[subject].append(avg_rms)
        
        x = np.arange(len(exercises))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, rms_data[subjects[0]], width, label=f'受试者{subjects[0]}', alpha=0.8)
        axes[0, 0].bar(x + width/2, rms_data[subjects[1]], width, label=f'受试者{subjects[1]}', alpha=0.8)
        axes[0, 0].set_xlabel('Exercise')
        axes[0, 0].set_ylabel('平均RMS值')
        axes[0, 0].set_title('信号强度比较')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'E{ex}' for ex in exercises])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 信噪比比较
        snr_data = {subject: [] for subject in subjects}
        for subject in subjects:
            for exercise in exercises:
                if exercise in all_metrics[subject]:
                    avg_snr = np.mean([ch['snr_estimate'] for ch in all_metrics[subject][exercise].values()])
                    snr_data[subject].append(avg_snr)
        
        axes[0, 1].bar(x - width/2, snr_data[subjects[0]], width, label=f'受试者{subjects[0]}', alpha=0.8)
        axes[0, 1].bar(x + width/2, snr_data[subjects[1]], width, label=f'受试者{subjects[1]}', alpha=0.8)
        axes[0, 1].set_xlabel('Exercise')
        axes[0, 1].set_ylabel('平均信噪比')
        axes[0, 1].set_title('信噪比比较')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'E{ex}' for ex in exercises])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 噪声水平比较
        noise_data = {subject: [] for subject in subjects}
        for subject in subjects:
            for exercise in exercises:
                if exercise in all_metrics[subject]:
                    avg_noise = np.mean([ch['noise_level'] for ch in all_metrics[subject][exercise].values()])
                    noise_data[subject].append(avg_noise)
        
        axes[1, 0].bar(x - width/2, noise_data[subjects[0]], width, label=f'受试者{subjects[0]}', alpha=0.8)
        axes[1, 0].bar(x + width/2, noise_data[subjects[1]], width, label=f'受试者{subjects[1]}', alpha=0.8)
        axes[1, 0].set_xlabel('Exercise')
        axes[1, 0].set_ylabel('平均噪声水平')
        axes[1, 0].set_title('噪声水平比较')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'E{ex}' for ex in exercises])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 信号复杂度比较
        complexity_data = {subject: [] for subject in subjects}
        for subject in subjects:
            for exercise in exercises:
                if exercise in all_metrics[subject]:
                    avg_complexity = np.mean([ch['complexity'] for ch in all_metrics[subject][exercise].values()])
                    complexity_data[subject].append(avg_complexity)
        
        axes[1, 1].bar(x - width/2, complexity_data[subjects[0]], width, label=f'受试者{subjects[0]}', alpha=0.8)
        axes[1, 1].bar(x + width/2, complexity_data[subjects[1]], width, label=f'受试者{subjects[1]}', alpha=0.8)
        axes[1, 1].set_xlabel('Exercise')
        axes[1, 1].set_ylabel('平均信号复杂度')
        axes[1, 1].set_title('信号复杂度比较')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'E{ex}' for ex in exercises])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('subjects_7_8_signal_quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_specific_problems(self, all_metrics):
        """分析具体问题"""
        print("\n=== 问题诊断报告 ===")
        
        for subject_id in self.subjects:
            print(f"\n受试者{subject_id} 问题分析:")
            
            if subject_id not in all_metrics:
                continue
                
            for exercise in [1, 2, 3]:
                if exercise not in all_metrics[subject_id]:
                    continue
                    
                print(f"  Exercise {exercise}:")
                metrics = all_metrics[subject_id][exercise]
                
                # 检查每个通道的问题
                for ch, ch_metrics in metrics.items():
                    problems = []
                    
                    if ch_metrics['rms'] < 50:  # 信号太弱
                        problems.append("信号强度过低")
                    if ch_metrics['snr_estimate'] < 3:  # 信噪比太低
                        problems.append("信噪比过低")
                    if ch_metrics['noise_level'] > 30:  # 噪声太高
                        problems.append("噪声水平过高")
                    if ch_metrics['complexity'] < 0.5:  # 信号过于简单
                        problems.append("信号过于简单")
                    
                    if problems:
                        print(f"    通道{ch+1}: {', '.join(problems)}")
                        print(f"      RMS: {ch_metrics['rms']:.2f}, SNR: {ch_metrics['snr_estimate']:.2f}")
                        print(f"      噪声: {ch_metrics['noise_level']:.2f}, 复杂度: {ch_metrics['complexity']:.3f}")

def main():
    """主函数"""
    print("=== 受试者7和8原始信号分析 ===")
    
    # 检查数据目录
    data_dir = 'D:/DB4'
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请检查数据路径或修改脚本中的data_dir变量")
        return
    
    # 创建分析器
    analyzer = RawSignalAnalyzer(data_dir)
    
    # 分析两个受试者的信号质量
    all_metrics = analyzer.compare_subjects_signal_quality()
    
    # 生成问题诊断报告
    analyzer.analyze_specific_problems(all_metrics)
    
    print("\n=== 分析完成 ===")
    print("生成的文件:")
    print("- subject_7_exercise_*.png: 受试者7各Exercise的原始信号")
    print("- subject_8_exercise_*.png: 受试者8各Exercise的原始信号")
    print("- subjects_7_8_signal_quality_comparison.png: 信号质量比较图")

if __name__ == "__main__":
    main() 