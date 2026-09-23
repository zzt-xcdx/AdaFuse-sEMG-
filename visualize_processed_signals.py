#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化增强预处理后的信号对比
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os
from scipy.signal import butter, filtfilt, wiener

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SignalProcessor:
    """信号处理器"""
    
    def __init__(self, sampling_rate=2000):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def adaptive_bandpass_filter(self, signal_data, low_freq=10, high_freq=90, order=4):
        """自适应带通滤波"""
        signal_std = np.std(signal_data)
        
        # 如果信号变化很大，使用更严格的滤波
        if signal_std > 200:
            low_freq = 20
            high_freq = 70
            order = 6
        
        low = low_freq / self.nyquist
        high = high_freq / self.nyquist
        
        if low >= high:
            low = 0.01
            high = 0.45
        
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_data)
        
        return filtered
    
    def adaptive_noise_reduction(self, signal_data):
        """自适应噪声抑制"""
        # 计算局部信号变化
        window_size = min(1000, len(signal_data) // 10)
        variation_levels = []
        
        for i in range(0, len(signal_data) - window_size, window_size // 2):
            window = signal_data[i:i+window_size]
            variation_levels.append(np.std(window))
        
        avg_variation = np.mean(variation_levels)
        
        # 根据信号变化水平选择处理策略
        if avg_variation > 200:
            # 强滤波
            filtered = self.adaptive_bandpass_filter(signal_data, 25, 65, 8)
            filtered = wiener(filtered, (101,))
        elif avg_variation > 100:
            # 中等滤波
            filtered = self.adaptive_bandpass_filter(signal_data, 20, 75, 6)
            filtered = wiener(filtered, (51,))
        else:
            # 轻微滤波
            filtered = self.adaptive_bandpass_filter(signal_data, 15, 85, 4)
        
        return filtered
    
    def normalize_signal(self, signal_data):
        """鲁棒标准化"""
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median))
        return (signal_data - median) / (mad * 1.4826 + 1e-10)
    
    def process_signal(self, signal_data):
        """处理单个信号"""
        # 1. 自适应噪声抑制
        processed = self.adaptive_noise_reduction(signal_data)
        
        # 2. 鲁棒标准化
        processed = self.normalize_signal(processed)
        
        return processed

def load_and_process_data(data_dir, subject_id, exercise):
    """加载并处理数据"""
    file_path = os.path.join(data_dir, f's{subject_id}', f'S{subject_id}_E{exercise}_A1.mat')
    
    if not os.path.exists(file_path):
        return None
    
    try:
        mat = scio.loadmat(file_path)
        emg = mat['emg']
        stimulus = mat['stimulus'].flatten()
        
        # 处理信号
        processor = SignalProcessor()
        processed_emg = np.zeros_like(emg)
        
        for ch in range(emg.shape[1]):
            processed_emg[:, ch] = processor.process_signal(emg[:, ch])
        
        return {
            'original': emg,
            'processed': processed_emg,
            'stimulus': stimulus
        }
    except Exception as e:
        print(f"处理受试者{subject_id} Exercise{exercise}失败: {e}")
        return None

def visualize_processed_comparison(data_dir):
    """可视化处理后的信号对比"""
    subjects = [7, 8, 10]
    exercises = [1, 2, 3]
    
    for exercise in exercises:
        print(f"正在处理Exercise {exercise}...")
        
        # 创建大图
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Exercise {exercise} - 增强预处理后信号对比', fontsize=16)
        
        for i, subject_id in enumerate(subjects):
            data = load_and_process_data(data_dir, subject_id, exercise)
            
            if data is None:
                axes[i, 0].text(0.5, 0.5, '数据加载失败', ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_title(f'受试者{subject_id} - 加载失败')
                continue
            
            # 选择前3个通道进行可视化
            for ch in range(3):
                original_signal = data['original'][:, ch]
                processed_signal = data['processed'][:, ch]
                stimulus = data['stimulus']
                
                # 绘制处理后的信号
                time_axis = np.arange(len(processed_signal)) / 2000
                axes[i, ch].plot(time_axis, processed_signal, linewidth=0.5, alpha=0.8, color='blue', label='处理后')
                
                # 标记手势区域
                gesture_starts = np.where(np.diff(stimulus) > 0)[0]
                gesture_ends = np.where(np.diff(stimulus) < 0)[0]
                
                for start, end in zip(gesture_starts, gesture_ends):
                    if end > start:
                        axes[i, ch].axvspan(start/2000, end/2000, alpha=0.2, color='red')
                
                # 设置标题和标签
                original_rms = np.sqrt(np.mean(original_signal**2))
                original_std = np.std(original_signal)
                processed_rms = np.sqrt(np.mean(processed_signal**2))
                processed_std = np.std(processed_signal)
                
                improvement = (original_std - processed_std) / original_std * 100
                
                axes[i, ch].set_title(f'受试者{subject_id} 通道{ch+1}\n'
                                     f'原始: RMS={original_rms:.1f}, STD={original_std:.1f}\n'
                                     f'处理后: RMS={processed_rms:.1f}, STD={processed_std:.1f}\n'
                                     f'改善: {improvement:.1f}%')
                axes[i, ch].set_xlabel('时间 (s)')
                axes[i, ch].set_ylabel('幅值 (标准化)')
                axes[i, ch].grid(True, alpha=0.3)
                
                # 设置y轴范围
                y_range = np.percentile(processed_signal, [1, 99])
                axes[i, ch].set_ylim(y_range[0] * 1.1, y_range[1] * 1.1)
        
        plt.tight_layout()
        plt.savefig(f'exercise_{exercise}_processed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Exercise {exercise} 可视化完成，已保存为 exercise_{exercise}_processed_comparison.png")

def visualize_improvement_summary(data_dir):
    """可视化改善效果总结"""
    subjects = [7, 8, 10]
    exercises = [1, 2, 3]
    
    # 收集改善数据
    improvement_data = {}
    
    for subject_id in subjects:
        improvement_data[subject_id] = {}
        for exercise in exercises:
            data = load_and_process_data(data_dir, subject_id, exercise)
            
            if data is not None:
                improvements = []
                for ch in range(data['original'].shape[1]):
                    original_std = np.std(data['original'][:, ch])
                    processed_std = np.std(data['processed'][:, ch])
                    improvement = (original_std - processed_std) / original_std * 100
                    improvements.append(improvement)
                
                improvement_data[subject_id][exercise] = np.mean(improvements)
    
    # 创建改善效果对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('增强预处理改善效果对比', fontsize=16)
    
    for i, exercise in enumerate(exercises):
        exercise_data = []
        labels = []
        
        for subject_id in subjects:
            if exercise in improvement_data[subject_id]:
                exercise_data.append(improvement_data[subject_id][exercise])
                labels.append(f'受试者{subject_id}')
        
        if exercise_data:
            bars = axes[i].bar(labels, exercise_data, alpha=0.8)
            axes[i].set_title(f'Exercise {exercise}')
            axes[i].set_ylabel('平均改善率 (%)')
            axes[i].grid(True, alpha=0.3)
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, exercise_data):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('improvement_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("改善效果总结图已保存为 improvement_summary.png")

def main():
    """主函数"""
    data_dir = 'D:/DB4'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    print("=== 开始可视化增强预处理后的信号对比 ===")
    
    # 可视化处理后的信号对比
    visualize_processed_comparison(data_dir)
    
    # 可视化改善效果总结
    visualize_improvement_summary(data_dir)
    
    print("\n=== 可视化完成 ===")
    print("生成的文件:")
    print("- exercise_*_processed_comparison.png: 各Exercise处理后的信号对比图")
    print("- improvement_summary.png: 改善效果总结图")

if __name__ == "__main__":
    main() 