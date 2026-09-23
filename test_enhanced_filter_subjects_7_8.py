#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试针对受试者7和8的增强滤波方案
验证运动伪影去除、噪声抑制和信号质量提升效果
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入增强滤波器
from Denoise.EMGFilter_enhanced_subjects_7_8 import EnhancedFilterForSubjects7And8

def test_enhanced_filter():
    """测试增强滤波方案"""
    print("测试针对受试者7和8的增强滤波方案")
    print("=" * 60)
    
    # 创建增强滤波器
    enhancer = EnhancedFilterForSubjects7And8(fs=2000)
    
    # 生成模拟的受试者7和8的噪声信号
    print("生成模拟噪声信号...")
    
    # 模拟受试者7的信号（SNR=0.0dB，特征0方差大）
    fs = 2000
    duration = 2.0  # 2秒
    n_samples = int(fs * duration)
    
    # 基础EMG信号
    t = np.linspace(0, duration, n_samples)
    emg_base = np.random.normal(0, 1, n_samples)
    
    # 添加基线漂移
    baseline_drift = 0.3 * np.sin(2 * np.pi * 0.5 * t)
    
    # 添加工频干扰
    power_line_50 = 1.5 * np.sin(2 * np.pi * 50 * t)
    power_line_100 = 1.0 * np.sin(2 * np.pi * 100 * t)
    
    # 添加高频噪声
    high_freq_noise = 0.5 * np.random.normal(0, 1, n_samples)
    
    # 模拟受试者7的信号（低质量）
    subject_7_signal = emg_base + baseline_drift + power_line_50 + power_line_100 + high_freq_noise
    
    # 模拟受试者8的信号（包含运动伪影）
    subject_8_signal = emg_base + baseline_drift + power_line_50 + power_line_100 + high_freq_noise
    
    # 添加运动伪影（极大尖峰）
    artifact_indices = [int(0.5 * fs), int(1.2 * fs), int(1.8 * fs)]  # 在0.5s, 1.2s, 1.8s处添加伪影
    for idx in artifact_indices:
        if idx < len(subject_8_signal):
            subject_8_signal[idx] += 2000 * np.random.choice([-1, 1])  # 添加±2000的尖峰
    
    # 测试增强滤波
    print("\n测试受试者7的信号增强...")
    enhanced_7 = enhancer.enhance_channel(subject_7_signal, 0)
    
    print("\n测试受试者8的信号增强...")
    enhanced_8 = enhancer.enhance_channel(subject_8_signal, 0)
    
    # 可视化结果
    print("\n生成可视化结果...")
    create_comparison_plots(t, subject_7_signal, enhanced_7, subject_8_signal, enhanced_8)
    
    print("\n测试完成！请查看生成的图片文件。")

def create_comparison_plots(t, signal_7_raw, signal_7_enhanced, signal_8_raw, signal_8_enhanced):
    """创建对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 受试者7对比
    axes[0, 0].plot(t, signal_7_raw, 'b-', alpha=0.7, label='原始信号')
    axes[0, 0].plot(t, signal_7_enhanced, 'r-', alpha=0.8, label='增强信号')
    axes[0, 0].set_title('受试者7: 原始信号 vs 增强信号')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('幅度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 受试者8对比
    axes[0, 1].plot(t, signal_8_raw, 'b-', alpha=0.7, label='原始信号')
    axes[0, 1].plot(t, signal_8_enhanced, 'r-', alpha=0.8, label='增强信号')
    axes[0, 1].set_title('受试者8: 原始信号 vs 增强信号')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('幅度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 频谱对比 - 受试者7
    freqs_7_raw, psd_7_raw = plt.mlab.psd(signal_7_raw, Fs=2000, NFFT=1024)
    freqs_7_enh, psd_7_enh = plt.mlab.psd(signal_7_enhanced, Fs=2000, NFFT=1024)
    
    axes[1, 0].semilogy(freqs_7_raw, psd_7_raw, 'b-', alpha=0.7, label='原始信号')
    axes[1, 0].semilogy(freqs_7_enh, psd_7_enh, 'r-', alpha=0.8, label='增强信号')
    axes[1, 0].set_title('受试者7: 功率谱密度对比')
    axes[1, 0].set_xlabel('频率 (Hz)')
    axes[1, 0].set_ylabel('功率谱密度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 500)
    
    # 频谱对比 - 受试者8
    freqs_8_raw, psd_8_raw = plt.mlab.psd(signal_8_raw, Fs=2000, NFFT=1024)
    freqs_8_enh, psd_8_enh = plt.mlab.psd(signal_8_enhanced, Fs=2000, NFFT=1024)
    
    axes[1, 1].semilogy(freqs_8_raw, psd_8_raw, 'b-', alpha=0.7, label='原始信号')
    axes[1, 1].semilogy(freqs_8_enh, psd_8_enh, 'r-', alpha=0.8, label='增强信号')
    axes[1, 1].set_title('受试者8: 功率谱密度对比')
    axes[1, 1].set_xlabel('频率 (Hz)')
    axes[1, 1].set_ylabel('功率谱密度')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 500)
    
    plt.tight_layout()
    plt.savefig('enhanced_filter_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("对比图已保存到 enhanced_filter_test_results.png")

def test_on_real_data():
    """在真实数据上测试（如果数据存在）"""
    print("\n尝试在真实数据上测试...")
    
    dataset = 'DB4'
    root_data = 'D:/DB4'
    
    # 检查是否有真实数据
    for subject_id in [7, 8]:
        raw_file = os.path.join(root_data, f'data/restimulus/reraw/{dataset}_s{subject_id}raw.h5')
        
        if os.path.exists(raw_file):
            print(f"\n找到受试者 {subject_id} 的真实数据，进行测试...")
            
            try:
                # 加载真实数据
                df_raw = pd.read_hdf(raw_file, 'df')
                emg_columns = [col for col in df_raw.columns if col not in ['restimulus', 'rerepetition']]
                emg_data = df_raw.iloc[:, :len(emg_columns)].values
                
                # 创建增强滤波器
                enhancer = EnhancedFilterForSubjects7And8(fs=2000)
                
                # 测试第一个通道
                channel_0 = emg_data[:, 0]
                print(f"受试者 {subject_id} 通道1数据形状: {channel_0.shape}")
                
                # 增强处理
                enhanced_channel = enhancer.enhance_channel(channel_0, 0)
                
                # 保存对比图
                create_real_data_comparison(channel_0, enhanced_channel, subject_id)
                
            except Exception as e:
                print(f"处理受试者 {subject_id} 数据时出错: {e}")
        else:
            print(f"未找到受试者 {subject_id} 的真实数据文件")

def create_real_data_comparison(raw_signal, enhanced_signal, subject_id):
    """创建真实数据的对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 时域对比
    t = np.arange(len(raw_signal)) / 2000  # 采样频率2000Hz
    
    axes[0].plot(t, raw_signal, 'b-', alpha=0.7, label='原始信号')
    axes[0].plot(t, enhanced_signal, 'r-', alpha=0.8, label='增强信号')
    axes[0].set_title(f'受试者 {subject_id}: 真实数据对比')
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('幅度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 频谱对比
    freqs_raw, psd_raw = plt.mlab.psd(raw_signal, Fs=2000, NFFT=1024)
    freqs_enh, psd_enh = plt.mlab.psd(enhanced_signal, Fs=2000, NFFT=1024)
    
    axes[1].semilogy(freqs_raw, psd_raw, 'b-', alpha=0.7, label='原始信号')
    axes[1].semilogy(freqs_enh, psd_enh, 'r-', alpha=0.8, label='增强信号')
    axes[1].set_title(f'受试者 {subject_id}: 功率谱密度对比')
    axes[1].set_xlabel('频率 (Hz)')
    axes[1].set_ylabel('功率谱密度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 500)
    
    plt.tight_layout()
    plt.savefig(f'subject_{subject_id}_real_data_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"受试者 {subject_id} 的真实数据对比图已保存")

def main():
    """主函数"""
    print("受试者7和8增强滤波方案测试")
    print("=" * 60)
    
    # 测试模拟数据
    test_enhanced_filter()
    
    # 测试真实数据（如果存在）
    test_on_real_data()
    
    print("\n=== 测试完成 ===")
    print("请查看生成的图片文件来评估增强效果")

if __name__ == "__main__":
    main() 