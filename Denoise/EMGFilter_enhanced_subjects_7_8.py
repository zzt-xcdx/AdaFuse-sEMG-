#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门针对受试者7和8的增强滤波方案
解决SNR为0.0dB、运动伪影、特征不稳定等问题
"""

import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats
from scipy.ndimage import median_filter

# 将项目根目录添加到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nina_funcs as nf

class EnhancedFilterForSubjects7And8:
    """专门针对受试者7和8的增强滤波器"""
    
    def __init__(self, fs=2000):
        self.fs = fs
        
    def detect_motion_artifacts(self, signal_ch, threshold_factor=5.0):
        """检测运动伪影"""
        # 计算信号的统计特性
        mean_val = np.mean(signal_ch)
        std_val = np.std(signal_ch)
        
        # 检测异常值（运动伪影）
        upper_threshold = mean_val + threshold_factor * std_val
        lower_threshold = mean_val - threshold_factor * std_val
        
        # 标记异常点
        artifacts_mask = (signal_ch > upper_threshold) | (signal_ch < lower_threshold)
        
        return artifacts_mask, upper_threshold, lower_threshold
    
    def remove_motion_artifacts(self, signal_ch, window_size=100):
        """去除运动伪影"""
        # 检测运动伪影
        artifacts_mask, upper_th, lower_th = self.detect_motion_artifacts(signal_ch)
        
        # 创建清理后的信号
        cleaned_signal = signal_ch.copy()
        
        # 对异常点进行插值处理
        if np.any(artifacts_mask):
            # 找到异常点的索引
            artifact_indices = np.where(artifacts_mask)[0]
            
            # 对每个异常点进行局部中值滤波
            for idx in artifact_indices:
                start_idx = max(0, idx - window_size // 2)
                end_idx = min(len(signal_ch), idx + window_size // 2)
                
                # 获取局部窗口（排除异常点）
                local_window = signal_ch[start_idx:end_idx]
                local_mask = artifacts_mask[start_idx:end_idx]
                
                if np.sum(~local_mask) > 0:  # 如果有非异常点
                    # 使用非异常点的中值替换
                    median_val = np.median(local_window[~local_mask])
                    cleaned_signal[idx] = median_val
                else:
                    # 如果局部窗口都是异常点，使用全局中值
                    cleaned_signal[idx] = np.median(signal_ch[~artifacts_mask])
        
        return cleaned_signal
    
    def adaptive_noise_suppression(self, signal_ch, quality_score):
        """自适应噪声抑制"""
        if quality_score < 0.3:  # 极低质量信号
            # 强噪声抑制
            return self.strong_noise_suppression(signal_ch)
        elif quality_score < 0.6:  # 低质量信号
            # 中等噪声抑制
            return self.medium_noise_suppression(signal_ch)
        else:  # 较高质量信号
            # 轻微噪声抑制
            return self.light_noise_suppression(signal_ch)
    
    def strong_noise_suppression(self, signal_ch):
        """强噪声抑制（针对SNR=0.0dB的情况）"""
        # 1. 强高通滤波 (40Hz)
        sos_hp = signal.butter(8, 40, btype='highpass', fs=self.fs, output='sos')
        signal_hp = signal.sosfiltfilt(sos_hp, signal_ch)
        
        # 2. 强陷波滤波 (50Hz, 100Hz)
        sos_n50 = signal.iirnotch(50, 60, fs=self.fs)
        sos_n50 = signal.tf2sos(*sos_n50)
        signal_n50 = signal.sosfiltfilt(sos_n50, signal_hp)
        
        sos_n100 = signal.iirnotch(100, 60, fs=self.fs)
        sos_n100 = signal.tf2sos(*sos_n100)
        signal_n100 = signal.sosfiltfilt(sos_n100, signal_n50)
        
        # 3. 强低通滤波 (350Hz)
        sos_lp = signal.butter(8, 350, btype='lowpass', fs=self.fs, output='sos')
        signal_lp = signal.sosfiltfilt(sos_lp, signal_n100)
        
        # 4. 中值滤波去噪
        signal_median = median_filter(signal_lp, size=7)
        
        # 5. 小波去噪（简化版）
        signal_denoised = self.wavelet_denoise(signal_median)
        
        return signal_denoised
    
    def medium_noise_suppression(self, signal_ch):
        """中等噪声抑制"""
        # 1. 标准高通滤波 (25Hz)
        sos_hp = signal.butter(6, 25, btype='highpass', fs=self.fs, output='sos')
        signal_hp = signal.sosfiltfilt(sos_hp, signal_ch)
        
        # 2. 标准陷波滤波
        sos_n50 = signal.iirnotch(50, 40, fs=self.fs)
        sos_n50 = signal.tf2sos(*sos_n50)
        signal_n50 = signal.sosfiltfilt(sos_n50, signal_hp)
        
        sos_n100 = signal.iirnotch(100, 40, fs=self.fs)
        sos_n100 = signal.tf2sos(*sos_n100)
        signal_n100 = signal.sosfiltfilt(sos_n100, signal_n50)
        
        # 3. 标准低通滤波 (400Hz)
        sos_lp = signal.butter(6, 400, btype='lowpass', fs=self.fs, output='sos')
        signal_lp = signal.sosfiltfilt(sos_lp, signal_n100)
        
        # 4. 轻微中值滤波
        signal_median = median_filter(signal_lp, size=5)
        
        return signal_median
    
    def light_noise_suppression(self, signal_ch):
        """轻微噪声抑制"""
        # 使用标准滤波方案
        return nf.advanced_emg_filter(signal_ch.reshape(-1, 1), fs=self.fs, region='EU').flatten()
    
    def wavelet_denoise(self, signal_ch, wavelet='db4', level=3):
        """小波去噪（简化版）"""
        try:
            import pywt
            # 小波分解
            coeffs = pywt.wavedec(signal_ch, wavelet, level=level)
            
            # 阈值处理
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal_ch)))
            
            # 软阈值处理
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            
            # 小波重构
            denoised = pywt.waverec(coeffs, wavelet)
            
            # 确保长度一致
            if len(denoised) != len(signal_ch):
                denoised = denoised[:len(signal_ch)]
            
            return denoised
        except ImportError:
            # 如果没有pywt，使用中值滤波替代
            return median_filter(signal_ch, size=5)
    
    def calculate_signal_quality(self, signal_ch):
        """计算信号质量评分"""
        # 计算RMS
        rms = np.sqrt(np.mean(signal_ch**2))
        
        # 计算峰值
        peak = np.max(np.abs(signal_ch))
        
        # 计算信噪比（简化版）
        # 假设信号在20-450Hz范围内，噪声在其他频率
        freqs, psd = signal.welch(signal_ch, fs=self.fs)
        
        # 信号频带 (20-450Hz)
        signal_mask = (freqs >= 20) & (freqs <= 450)
        noise_mask = (freqs < 20) | (freqs > 450)
        
        if np.any(signal_mask) and np.any(noise_mask):
            signal_power = np.mean(psd[signal_mask])
            noise_power = np.mean(psd[noise_mask])
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 0
        else:
            snr_db = 0
        
        # 计算质量评分 (0-1)
        quality_score = 0
        
        # RMS评分 (理想范围50-500)
        if 50 <= rms <= 500:
            quality_score += 0.4
        elif 20 <= rms <= 1000:
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # SNR评分
        if snr_db >= 10:
            quality_score += 0.4
        elif snr_db >= 5:
            quality_score += 0.2
        elif snr_db >= 0:
            quality_score += 0.1
        else:
            quality_score += 0.05
        
        # 峰值评分
        if peak <= 1000:
            quality_score += 0.2
        elif peak <= 3000:
            quality_score += 0.1
        else:
            quality_score += 0.05
        
        return {
            'rms': rms,
            'peak': peak,
            'snr_db': snr_db,
            'quality_score': quality_score
        }
    
    def enhance_channel(self, signal_ch, channel_idx):
        """增强单个通道"""
        print(f"  处理通道 {channel_idx + 1}...")
        
        # 1. 计算原始信号质量
        original_quality = self.calculate_signal_quality(signal_ch)
        print(f"    原始质量: RMS={original_quality['rms']:.1f}, "
              f"SNR={original_quality['snr_db']:.1f}dB, "
              f"Peak={original_quality['peak']:.1f}, "
              f"Score={original_quality['quality_score']:.3f}")
        
        # 2. 去除运动伪影
        signal_no_artifacts = self.remove_motion_artifacts(signal_ch)
        
        # 3. 自适应噪声抑制
        enhanced_signal = self.adaptive_noise_suppression(
            signal_no_artifacts, 
            original_quality['quality_score']
        )
        
        # 4. 计算增强后质量
        enhanced_quality = self.calculate_signal_quality(enhanced_signal)
        print(f"    增强后质量: RMS={enhanced_quality['rms']:.1f}, "
              f"SNR={enhanced_quality['snr_db']:.1f}dB, "
              f"Peak={enhanced_quality['peak']:.1f}, "
              f"Score={enhanced_quality['quality_score']:.3f}")
        
        return enhanced_signal
    
    def enhance_subject_data(self, subject_id, dataset='DB4', root_data='D:/DB4'):
        """增强受试者数据"""
        print(f"\n=== 增强受试者 {subject_id} 的数据 ===")
        
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
        
        print(f"EMG通道数量: {emg_channel_count}")
        
        # 逐通道增强
        enhanced_emg = np.zeros_like(emg_data)
        for ch in range(emg_channel_count):
            enhanced_emg[:, ch] = self.enhance_channel(emg_data[:, ch], ch)
        
        # 创建输出DataFrame
        df_enhanced = pd.DataFrame(enhanced_emg, columns=range(emg_channel_count))
        df_enhanced['restimulus'] = df_raw['restimulus'].values
        df_enhanced['rerepetition'] = df_raw['rerepetition'].values
        
        return df_enhanced

def main():
    """主函数"""
    print("受试者7和8专用增强滤波方案")
    print("=" * 50)
    
    # 配置
    dataset = 'DB4'
    root_data = 'D:/DB4'
    target_subjects = [7, 8]  # 只处理受试者7和8
    
    # 创建增强滤波器
    enhancer = EnhancedFilterForSubjects7And8(fs=2000)
    
    # 处理目标受试者
    for subject_id in target_subjects:
        print(f"\n处理受试者 {subject_id}...")
        
        # 增强数据
        df_enhanced = enhancer.enhance_subject_data(subject_id, dataset, root_data)
        
        if df_enhanced is not None:
            # 保存增强后的数据
            output_dir = os.path.join(root_data, 'data/restimulus/refilter_enhanced')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f'{dataset}_s{subject_id}filter_enhanced.h5')
            df_enhanced.to_hdf(output_file, format='table', key='df', mode='w', complevel=9, complib='blosc')
            
            print(f"增强数据已保存到: {output_file}")
            print(f"数据形状: {df_enhanced.shape}")
        else:
            print(f"受试者 {subject_id} 数据处理失败")
    
    print("\n=== 处理完成 ===")
    print("增强后的数据已保存到 refilter_enhanced 目录")
    print("现在可以运行 EMGFilter.py 来使用这些增强数据")

if __name__ == "__main__":
    main() 