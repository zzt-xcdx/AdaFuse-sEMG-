#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对受试者7和8的增强预处理脚本
专门处理噪声水平过高的问题
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import warnings
from scipy import signal
from scipy.signal import butter, filtfilt, wiener
from scipy.stats import stats
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedPreprocessor:
    """增强的EMG信号预处理器"""
    
    def __init__(self, sampling_rate=2000):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def load_data(self, data_dir, subject_id, exercise):
        """加载数据"""
        file_path = os.path.join(data_dir, f's{subject_id}', f'S{subject_id}_E{exercise}_A1.mat')
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
            
        try:
            mat = scio.loadmat(file_path)
            return {
                'emg': mat['emg'],
                'stimulus': mat['stimulus'].flatten(),
                'repetition': mat['repetition'].flatten()
            }
        except Exception as e:
            print(f"加载失败: {e}")
            return None
    
    def adaptive_bandpass_filter(self, signal_data, low_freq=10, high_freq=90, order=4):
        """自适应带通滤波"""
        # 根据信号特性调整滤波参数
        signal_std = np.std(signal_data)
        
        # 如果噪声很高，使用更严格的滤波
        if signal_std > 100:
            low_freq = 15  # 提高低频截止
            high_freq = 80  # 降低高频截止
            order = 6  # 增加滤波阶数
        
        # 设计滤波器
        low = low_freq / self.nyquist
        high = high_freq / self.nyquist
        
        if low >= high:
            low = 0.01
            high = 0.45
        
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_data)
        
        return filtered
    
    def wiener_filter(self, signal_data, window_size=51):
        """维纳滤波去噪"""
        # 自适应窗口大小
        if len(signal_data) < window_size:
            window_size = len(signal_data) // 4
            if window_size % 2 == 0:
                window_size += 1
        
        return wiener(signal_data, (window_size,))
    
    def wavelet_denoising(self, signal_data, wavelet='db4', level=3):
        """小波去噪"""
        try:
            import pywt
        except ImportError:
            print("pywt未安装，跳过小波去噪")
            return signal_data
        
        # 小波分解
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # 阈值处理
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal_data)))
        
        # 软阈值
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # 重构信号
        denoised = pywt.waverec(coeffs, wavelet)
        
        # 确保长度一致
        if len(denoised) != len(signal_data):
            denoised = denoised[:len(signal_data)]
        
        return denoised
    
    def adaptive_noise_reduction(self, signal_data):
        """自适应噪声抑制"""
        # 计算局部噪声水平
        window_size = min(1000, len(signal_data) // 10)
        noise_levels = []
        
        for i in range(0, len(signal_data) - window_size, window_size // 2):
            window = signal_data[i:i+window_size]
            noise_levels.append(np.std(window))
        
        avg_noise = np.mean(noise_levels)
        
        # 如果噪声水平很高，应用多重滤波
        if avg_noise > 100:
            # 第一层：强带通滤波
            filtered = self.adaptive_bandpass_filter(signal_data, 20, 70, 8)
            
            # 第二层：维纳滤波
            filtered = self.wiener_filter(filtered, 101)
            
            # 第三层：小波去噪
            filtered = self.wavelet_denoising(filtered, 'db4', 4)
            
        elif avg_noise > 50:
            # 中等噪声：标准滤波
            filtered = self.adaptive_bandpass_filter(signal_data, 15, 85, 6)
            filtered = self.wiener_filter(filtered, 51)
            
        else:
            # 低噪声：轻微滤波
            filtered = self.adaptive_bandpass_filter(signal_data, 10, 90, 4)
        
        return filtered
    
    def normalize_signal(self, signal_data, method='zscore'):
        """信号标准化"""
        if method == 'zscore':
            return (signal_data - np.mean(signal_data)) / np.std(signal_data)
        elif method == 'minmax':
            return (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
        elif method == 'robust':
            # 使用中位数和MAD进行鲁棒标准化
            median = np.median(signal_data)
            mad = np.median(np.abs(signal_data - median))
            return (signal_data - median) / (mad * 1.4826)
        else:
            return signal_data
    
    def process_channel(self, signal_data, channel_id):
        """处理单个通道"""
        print(f"  处理通道{channel_id+1}...")
        
        # 计算原始信号质量
        original_rms = np.sqrt(np.mean(signal_data**2))
        original_std = np.std(signal_data)
        
        # 应用增强预处理
        processed = signal_data.copy()
        
        # 1. 自适应噪声抑制
        processed = self.adaptive_noise_reduction(processed)
        
        # 2. 鲁棒标准化
        processed = self.normalize_signal(processed, 'robust')
        
        # 计算处理后信号质量
        processed_rms = np.sqrt(np.mean(processed**2))
        processed_std = np.std(processed)
        
        # 计算改善程度
        noise_reduction = (original_std - processed_std) / original_std * 100
        
        print(f"    原始: RMS={original_rms:.1f}, STD={original_std:.1f}")
        print(f"    处理后: RMS={processed_rms:.1f}, STD={processed_std:.1f}")
        print(f"    噪声减少: {noise_reduction:.1f}%")
        
        return processed, {
            'original_rms': original_rms,
            'original_std': original_std,
            'processed_rms': processed_rms,
            'processed_std': processed_std,
            'noise_reduction': noise_reduction
        }
    
    def process_subject_data(self, data_dir, subject_id, exercise):
        """处理受试者的完整数据"""
        print(f"\n=== 处理受试者{subject_id} Exercise{exercise} ===")
        
        # 加载数据
        data = self.load_data(data_dir, subject_id, exercise)
        if data is None:
            return None
        
        emg_data = data['emg']
        channels = emg_data.shape[1]
        
        # 处理每个通道
        processed_emg = np.zeros_like(emg_data)
        channel_metrics = {}
        
        for ch in range(channels):
            processed_ch, metrics = self.process_channel(emg_data[:, ch], ch)
            processed_emg[:, ch] = processed_ch
            channel_metrics[ch] = metrics
        
        # 计算整体改善效果
        avg_noise_reduction = np.mean([m['noise_reduction'] for m in channel_metrics.values()])
        print(f"\n整体噪声减少: {avg_noise_reduction:.1f}%")
        
        return {
            'original_emg': emg_data,
            'processed_emg': processed_emg,
            'stimulus': data['stimulus'],
            'repetition': data['repetition'],
            'channel_metrics': channel_metrics,
            'overall_improvement': avg_noise_reduction
        }
    
    def visualize_improvement(self, result, subject_id, exercise, save_path=None):
        """可视化改善效果"""
        if result is None:
            return
        
        original_emg = result['original_emg']
        processed_emg = result['processed_emg']
        channel_metrics = result['channel_metrics']
        
        # 选择前8个通道进行可视化
        channels_to_plot = min(8, original_emg.shape[1])
        
        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle(f'受试者{subject_id} Exercise{exercise} 预处理效果对比', fontsize=16)
        
        for ch in range(channels_to_plot):
            row = ch // 2
            col = ch % 2
            
            # 绘制原始信号和处理后信号
            time_axis = np.arange(len(original_emg[:, ch])) / self.sampling_rate
            
            axes[row, col].plot(time_axis, original_emg[:, ch], 
                               linewidth=0.5, alpha=0.7, label='原始信号', color='red')
            axes[row, col].plot(time_axis, processed_emg[:, ch], 
                               linewidth=0.5, alpha=0.7, label='处理后信号', color='blue')
            
            # 设置标题和标签
            metrics = channel_metrics[ch]
            axes[row, col].set_title(f'通道{ch+1}\n噪声减少: {metrics["noise_reduction"]:.1f}%')
            axes[row, col].set_xlabel('时间 (s)')
            axes[row, col].set_ylabel('幅值')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
            # 设置y轴范围
            y_min = min(np.min(original_emg[:, ch]), np.min(processed_emg[:, ch]))
            y_max = max(np.max(original_emg[:, ch]), np.max(processed_emg[:, ch]))
            axes[row, col].set_ylim(y_min * 1.1, y_max * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_processed_data(self, result, output_dir, subject_id, exercise):
        """保存处理后的数据"""
        if result is None:
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为MAT文件
        output_file = os.path.join(output_dir, f'S{subject_id}_E{exercise}_processed.mat')
        
        # 准备保存的数据
        save_data = {
            'emg': result['processed_emg'],
            'stimulus': result['stimulus'],
            'repetition': result['repetition'],
            'original_emg': result['original_emg'],
            'channel_metrics': result['channel_metrics'],
            'overall_improvement': result['overall_improvement']
        }
        
        try:
            scio.savemat(output_file, save_data)
            print(f"数据已保存到: {output_file}")
        except Exception as e:
            print(f"保存失败: {e}")

def main():
    """主函数"""
    print("=== 受试者7和8增强预处理 ===")
    
    # 配置
    data_dir = 'D:/DB4'
    output_dir = 'processed_data_subjects_7_8'
    subjects = [7, 8]
    exercises = [1, 2, 3]
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请检查数据路径或修改脚本中的data_dir变量")
        return
    
    # 创建预处理器
    preprocessor = EnhancedPreprocessor()
    
    # 处理所有受试者和Exercise
    all_results = {}
    
    for subject_id in subjects:
        all_results[subject_id] = {}
        
        for exercise in exercises:
            print(f"\n{'='*60}")
            print(f"处理受试者{subject_id} Exercise{exercise}")
            print(f"{'='*60}")
            
            # 处理数据
            result = preprocessor.process_subject_data(data_dir, subject_id, exercise)
            
            if result is not None:
                all_results[subject_id][exercise] = result
                
                # 可视化改善效果
                preprocessor.visualize_improvement(
                    result, subject_id, exercise,
                    f'subject_{subject_id}_exercise_{exercise}_improvement.png'
                )
                
                # 保存处理后的数据
                preprocessor.save_processed_data(
                    result, output_dir, subject_id, exercise
                )
    
    # 生成总结报告
    print(f"\n{'='*60}")
    print("预处理完成总结")
    print(f"{'='*60}")
    
    for subject_id in subjects:
        if subject_id in all_results:
            print(f"\n受试者{subject_id}:")
            for exercise in exercises:
                if exercise in all_results[subject_id]:
                    improvement = all_results[subject_id][exercise]['overall_improvement']
                    print(f"  Exercise{exercise}: 平均噪声减少 {improvement:.1f}%")
    
    print(f"\n处理后的数据保存在: {output_dir}")
    print("可视化结果已保存为PNG文件")

if __name__ == "__main__":
    main() 