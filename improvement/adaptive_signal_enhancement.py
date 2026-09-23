#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应信号增强模块
针对低准确率受试者的信号质量问题进行改进
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import zscore
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

class AdaptiveSignalEnhancer:
    """自适应信号增强器"""
    
    def __init__(self, fs=2000):
        self.fs = fs
        self.quality_thresholds = {
            'snr_min': 15,      # 最小信噪比阈值
            'rms_min': 50,      # 最小RMS阈值
            'peak_max': 2000,   # 最大峰值阈值
            'baseline_max': 100 # 最大基线漂移阈值
        }
    
    def assess_signal_quality(self, emg_data):
        """评估信号质量"""
        quality_metrics = {}
        
        for ch in range(emg_data.shape[1]):
            signal_ch = emg_data[:, ch]
            
            # 计算RMS
            rms = np.sqrt(np.mean(signal_ch**2))
            
            # 计算信噪比
            signal_power = np.var(signal_ch)
            noise_power = np.var(signal_ch - np.mean(signal_ch))
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
            
            # 计算峰值
            peak = np.max(np.abs(signal_ch))
            
            # 计算基线漂移
            baseline = np.abs(np.mean(signal_ch))
            
            quality_metrics[f'ch{ch+1}'] = {
                'rms': rms,
                'snr': snr,
                'peak': peak,
                'baseline': baseline,
                'quality_score': self.calculate_quality_score(rms, snr, peak, baseline)
            }
        
        return quality_metrics
    
    def calculate_quality_score(self, rms, snr, peak, baseline):
        """计算质量评分 (0-1, 越高越好)"""
        score = 0
        
        # RMS评分 (50-500为理想范围)
        if 50 <= rms <= 500:
            score += 0.25
        elif 20 <= rms <= 1000:
            score += 0.15
        else:
            score += 0.05
        
        # SNR评分 (15dB以上为良好)
        if snr >= 15:
            score += 0.25
        elif snr >= 10:
            score += 0.15
        elif snr >= 5:
            score += 0.10
        else:
            score += 0.05
        
        # 峰值评分 (避免过大峰值)
        if peak <= 2000:
            score += 0.25
        elif peak <= 5000:
            score += 0.15
        else:
            score += 0.05
        
        # 基线评分 (避免过大漂移)
        if baseline <= 50:
            score += 0.25
        elif baseline <= 100:
            score += 0.15
        else:
            score += 0.05
        
        return score
    
    def adaptive_filter(self, emg_data, quality_metrics):
        """自适应滤波"""
        enhanced_data = np.copy(emg_data)
        
        for ch in range(emg_data.shape[1]):
            signal_ch = emg_data[:, ch]
            quality = quality_metrics[f'ch{ch+1}']
            
            # 根据质量评分选择滤波策略
            if quality['quality_score'] < 0.5:
                # 低质量信号：使用强滤波
                enhanced_data[:, ch] = self.strong_filter(signal_ch)
            elif quality['quality_score'] < 0.8:
                # 中等质量信号：使用中等滤波
                enhanced_data[:, ch] = self.medium_filter(signal_ch)
            else:
                # 高质量信号：使用轻微滤波
                enhanced_data[:, ch] = self.light_filter(signal_ch)
        
        return enhanced_data
    
    def strong_filter(self, signal_ch):
        """强滤波方案（针对低质量信号）"""
        # 1. 异常值检测和去除
        signal_clean = self.remove_outliers(signal_ch)
        
        # 2. 强高通滤波 (30Hz)
        sos_hp = signal.butter(6, 30, btype='highpass', fs=self.fs, output='sos')
        signal_hp = signal.sosfiltfilt(sos_hp, signal_clean)
        
        # 3. 强陷波滤波 (50Hz, 100Hz)
        sos_n50 = signal.iirnotch(50, 50, fs=self.fs)
        sos_n50 = signal.tf2sos(*sos_n50)
        signal_n50 = signal.sosfiltfilt(sos_n50, signal_hp)
        
        sos_n100 = signal.iirnotch(100, 50, fs=self.fs)
        sos_n100 = signal.tf2sos(*sos_n100)
        signal_n100 = signal.sosfiltfilt(sos_n100, signal_n50)
        
        # 4. 强低通滤波 (400Hz)
        sos_lp = signal.butter(6, 400, btype='lowpass', fs=self.fs, output='sos')
        signal_lp = signal.sosfiltfilt(sos_lp, signal_n100)
        
        # 5. 中值滤波去噪
        signal_median = signal.medfilt(signal_lp, kernel_size=5)
        
        return signal_median.astype(np.float32)
    
    def medium_filter(self, signal_ch):
        """中等滤波方案"""
        # 1. 轻微异常值处理
        signal_clean = self.remove_outliers(signal_ch, threshold=3.0)
        
        # 2. 标准高通滤波 (20Hz)
        sos_hp = signal.butter(4, 20, btype='highpass', fs=self.fs, output='sos')
        signal_hp = signal.sosfiltfilt(sos_hp, signal_clean)
        
        # 3. 标准陷波滤波
        sos_n50 = signal.iirnotch(50, 30, fs=self.fs)
        sos_n50 = signal.tf2sos(*sos_n50)
        signal_n50 = signal.sosfiltfilt(sos_n50, signal_hp)
        
        sos_n100 = signal.iirnotch(100, 30, fs=self.fs)
        sos_n100 = signal.tf2sos(*sos_n100)
        signal_n100 = signal.sosfiltfilt(sos_n100, signal_n50)
        
        # 4. 标准低通滤波 (450Hz)
        sos_lp = signal.butter(4, 450, btype='lowpass', fs=self.fs, output='sos')
        signal_lp = signal.sosfiltfilt(sos_lp, signal_n100)
        
        return signal_lp.astype(np.float32)
    
    def light_filter(self, signal_ch):
        """轻微滤波方案（针对高质量信号）"""
        # 1. 标准滤波（与原来相同）
        return nf.advanced_emg_filter(signal_ch.reshape(-1, 1), fs=self.fs, region='EU').flatten()
    
    def remove_outliers(self, signal_ch, threshold=4.0):
        """去除异常值"""
        # 使用Z-score方法检测异常值
        z_scores = np.abs(zscore(signal_ch))
        outlier_mask = z_scores > threshold
        
        # 将异常值替换为邻近值的平均值
        signal_clean = signal_ch.copy()
        for i in np.where(outlier_mask)[0]:
            # 取前后5个点的平均值
            start_idx = max(0, i-5)
            end_idx = min(len(signal_ch), i+6)
            valid_indices = np.arange(start_idx, end_idx)[~outlier_mask[start_idx:end_idx]]
            
            if len(valid_indices) > 0:
                signal_clean[i] = np.mean(signal_ch[valid_indices])
            else:
                signal_clean[i] = 0
        
        return signal_clean
    
    def enhance_subject_data(self, subject_id, dataset='DB4', root_data='D:/DB4'):
        """增强单个受试者的数据"""
        print(f"增强受试者 {subject_id} 的数据...")
        
        # 加载原始数据
        raw_file = os.path.join(root_data, f'data/restimulus/reraw/{dataset}_s{subject_id}raw.h5')
        if not os.path.exists(raw_file):
            print(f"原始数据文件不存在: {raw_file}")
            return None
        
        df_raw = pd.read_hdf(raw_file, 'df')
        print(f"原始数据形状: {df_raw.shape}")
        
        # 获取EMG通道
        emg_columns = [col for col in df_raw.columns if col not in ['restimulus', 'rerepetition']]
        emg_channel_count = len(emg_columns)
        emg_data = df_raw.iloc[:, :emg_channel_count].values
        
        # 评估信号质量
        print("评估信号质量...")
        quality_metrics = self.assess_signal_quality(emg_data)
        
        # 打印质量评估结果
        print("\n信号质量评估:")
        for ch, metrics in quality_metrics.items():
            print(f"  {ch}: RMS={metrics['rms']:.1f}, SNR={metrics['snr']:.1f}dB, "
                  f"Peak={metrics['peak']:.1f}, Score={metrics['quality_score']:.3f}")
        
        # 自适应滤波
        print("\n应用自适应滤波...")
        enhanced_emg = self.adaptive_filter(emg_data, quality_metrics)
        
        # 重新评估滤波后质量
        enhanced_quality = self.assess_signal_quality(enhanced_emg)
        
        print("\n滤波后质量评估:")
        for ch, metrics in enhanced_quality.items():
            print(f"  {ch}: RMS={metrics['rms']:.1f}, SNR={metrics['snr']:.1f}dB, "
                  f"Peak={metrics['peak']:.1f}, Score={metrics['quality_score']:.3f}")
        
        # 创建输出DataFrame
        df_enhanced = pd.DataFrame(enhanced_emg, columns=range(emg_channel_count))
        df_enhanced['restimulus'] = df_raw['restimulus'].values
        df_enhanced['rerepetition'] = df_raw['rerepetition'].values
        
        return df_enhanced, quality_metrics, enhanced_quality
    
    def visualize_enhancement(self, subject_id, raw_data, enhanced_data, 
                            quality_metrics, enhanced_quality, save_path=None):
        """可视化增强效果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'受试者{subject_id} 信号增强效果对比', fontsize=16, fontweight='bold')
        
        # 1. 时域信号对比
        ax1 = axes[0, 0]
        time = np.arange(2000) / self.fs  # 显示前1秒
        ax1.plot(time, raw_data[:2000, 0], 'b-', alpha=0.7, label='原始信号')
        ax1.plot(time, enhanced_data[:2000, 0], 'r-', alpha=0.7, label='增强信号')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('幅度')
        ax1.set_title('时域信号对比 (通道1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 质量评分对比
        ax2 = axes[0, 1]
        channels = list(quality_metrics.keys())
        original_scores = [quality_metrics[ch]['quality_score'] for ch in channels]
        enhanced_scores = [enhanced_quality[ch]['quality_score'] for ch in channels]
        
        x = np.arange(len(channels))
        width = 0.35
        ax2.bar(x - width/2, original_scores, width, label='原始质量', alpha=0.7)
        ax2.bar(x + width/2, enhanced_scores, width, label='增强质量', alpha=0.7)
        ax2.set_xlabel('通道')
        ax2.set_ylabel('质量评分')
        ax2.set_title('质量评分对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(channels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. SNR对比
        ax3 = axes[1, 0]
        original_snr = [quality_metrics[ch]['snr'] for ch in channels]
        enhanced_snr = [enhanced_quality[ch]['snr'] for ch in channels]
        
        ax3.bar(x - width/2, original_snr, width, label='原始SNR', alpha=0.7)
        ax3.bar(x + width/2, enhanced_snr, width, label='增强SNR', alpha=0.7)
        ax3.set_xlabel('通道')
        ax3.set_ylabel('SNR (dB)')
        ax3.set_title('信噪比对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(channels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 功率谱对比
        ax4 = axes[1, 1]
        f_raw, Pxx_raw = signal.welch(raw_data[:10000, 0], self.fs, nperseg=1024)
        f_enh, Pxx_enh = signal.welch(enhanced_data[:10000, 0], self.fs, nperseg=1024)
        
        ax4.semilogy(f_raw, Pxx_raw, 'b-', alpha=0.7, label='原始信号')
        ax4.semilogy(f_enh, Pxx_enh, 'r-', alpha=0.7, label='增强信号')
        ax4.set_xlabel('频率 (Hz)')
        ax4.set_ylabel('功率谱密度')
        ax4.set_title('功率谱对比')
        ax4.set_xlim(0, 500)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path("improvement")
    output_dir.mkdir(exist_ok=True)
    
    # 创建增强器
    enhancer = AdaptiveSignalEnhancer(fs=2000)
    
    # 处理低准确率受试者
    low_perf_subjects = [7, 8]
    
    for subject_id in low_perf_subjects:
        print(f"\n{'='*60}")
        print(f"处理受试者 {subject_id}")
        print(f"{'='*60}")
        
        # 增强数据
        enhanced_df, original_quality, enhanced_quality = enhancer.enhance_subject_data(
            subject_id, dataset='DB4', root_data='D:/DB4'
        )
        
        if enhanced_df is not None:
            # 保存增强后的数据
            output_file = os.path.join('D:/DB4', f'data/restimulus/refilter_enhanced/DB4_s{subject_id}filter_enhanced.h5')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            enhanced_df.to_hdf(output_file, format='table', key='df', mode='w', complevel=9, complib='blosc')
            print(f"增强数据已保存: {output_file}")
            
            # 可视化增强效果
            raw_file = os.path.join('D:/DB4', f'data/restimulus/reraw/DB4_s{subject_id}raw.h5')
            raw_df = pd.read_hdf(raw_file, 'df')
            emg_columns = [col for col in raw_df.columns if col not in ['restimulus', 'rerepetition']]
            raw_emg = raw_df.iloc[:, :len(emg_columns)].values
            
            enhancer.visualize_enhancement(
                subject_id, raw_emg, enhanced_df.iloc[:, :len(emg_columns)].values,
                original_quality, enhanced_quality,
                save_path=f'improvement/subject_{subject_id}_enhancement.png'
            )
    
    print("\n自适应信号增强完成！")

if __name__ == "__main__":
    main() 