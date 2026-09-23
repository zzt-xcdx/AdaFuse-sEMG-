#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进的EMG滤波方案
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入新的滤波函数
from nina_funcs import advanced_emg_filter, check_filter_quality, adaptive_emg_filter

def generate_test_signal(fs=2000, duration=2.0):
    """
    生成测试EMG信号，包含各种干扰
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # 基础EMG信号 (20-450Hz)
    emg_base = np.random.normal(0, 1, len(t))
    
    # 添加基线漂移 (低频干扰)
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 2 * t)
    
    # 添加工频干扰
    power_line_50 = 2.0 * np.sin(2 * np.pi * 50 * t)
    power_line_100 = 1.5 * np.sin(2 * np.pi * 100 * t)
    
    # 添加高频噪声
    high_freq_noise = 0.3 * np.random.normal(0, 1, len(t))
    
    # 组合信号
    raw_signal = emg_base + baseline_drift + power_line_50 + power_line_100 + high_freq_noise
    
    return raw_signal, t

def test_filter_performance():
    """
    测试滤波性能
    """
    print("=" * 60)
    print("测试改进的EMG滤波方案")
    print("=" * 60)
    
    # 生成测试信号
    fs = 2000
    raw_signal, t = generate_test_signal(fs, duration=2.0)
    
    print(f"测试信号长度: {len(raw_signal)} 样本")
    print(f"采样频率: {fs} Hz")
    print(f"信号持续时间: {len(raw_signal)/fs:.1f} 秒")
    
    # 应用改进的滤波
    print("\n应用改进的滤波方案...")
    try:
        filtered_signal = advanced_emg_filter(raw_signal, fs=fs, region='EU')
        print("✓ 滤波成功完成")
    except Exception as e:
        print(f"✗ 滤波失败: {e}")
        return
    
    # 检查滤波质量
    print("\n检查滤波质量...")
    quality_ok = check_filter_quality(raw_signal, filtered_signal, fs=fs)
    
    # 可视化结果
    visualize_results(raw_signal, filtered_signal, t, fs)
    
    return quality_ok

def visualize_results(raw, filtered, t, fs):
    """
    可视化滤波结果
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(15, 10))
    
    # 时域信号
    plt.subplot(2, 2, 1)
    plt.plot(t, raw, 'b-', alpha=0.7, label='Raw Signal')
    plt.plot(t, filtered, 'r-', alpha=0.7, label='Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 频域信号
    plt.subplot(2, 2, 2)
    from scipy.signal import welch
    f_raw, P_raw = welch(raw, fs, nperseg=1024)
    f_filt, P_filt = welch(filtered, fs, nperseg=1024)
    
    plt.semilogy(f_raw, P_raw, 'b-', alpha=0.7, label='Raw Signal')
    plt.semilogy(f_filt, P_filt, 'r-', alpha=0.7, label='Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Frequency Domain Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    
    # 滤波前后对比 (放大显示)
    plt.subplot(2, 2, 3)
    start_idx = int(0.5 * fs)  # 从0.5秒开始
    end_idx = int(1.0 * fs)    # 到1.0秒结束
    plt.plot(t[start_idx:end_idx], raw[start_idx:end_idx], 'b-', alpha=0.7, label='Raw Signal')
    plt.plot(t[start_idx:end_idx], filtered[start_idx:end_idx], 'r-', alpha=0.7, label='Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Filtered vs Raw (0.5-1.0s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 频谱细节 (0-100Hz)
    plt.subplot(2, 2, 4)
    plt.semilogy(f_raw, P_raw, 'b-', alpha=0.7, label='Raw Signal')
    plt.semilogy(f_filt, P_filt, 'r-', alpha=0.7, label='Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Spectrum Detail (0-100Hz)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig('filter_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n可视化结果已保存为 'filter_test_results.png'")

def test_multi_channel():
    """
    测试多通道滤波
    """
    print("\n" + "=" * 60)
    print("测试多通道滤波")
    print("=" * 60)
    
    fs = 2000
    duration = 1.0
    n_channels = 4
    
    # 生成多通道测试信号
    t = np.linspace(0, duration, int(fs * duration))
    raw_multi = np.zeros((len(t), n_channels))
    
    for ch in range(n_channels):
        raw_multi[:, ch], _ = generate_test_signal(fs, duration)
    
    print(f"多通道信号形状: {raw_multi.shape}")
    
    # 应用滤波
    try:
        filtered_multi = advanced_emg_filter(raw_multi, fs=fs, region='EU')
        print("✓ 多通道滤波成功完成")
        
        # 检查每个通道的质量
        for ch in range(min(2, n_channels)):  # 只检查前2个通道
            print(f"\n检查通道 {ch} 的滤波质量:")
            quality_ok = check_filter_quality(raw_multi[:, ch], filtered_multi[:, ch], fs=fs)
            
    except Exception as e:
        print(f"✗ 多通道滤波失败: {e}")

def test_error_handling():
    """
    测试错误处理
    """
    print("\n" + "=" * 60)
    print("测试错误处理")
    print("=" * 60)
    
    # 测试空数组
    try:
        result = advanced_emg_filter(np.array([]), fs=2000)
        print("✓ 空数组处理正常")
    except Exception as e:
        print(f"✗ 空数组处理失败: {e}")
    
    # 测试无效参数
    try:
        result = advanced_emg_filter(np.random.randn(100), fs=0)  # 无效采样率
        print("✓ 无效参数处理正常")
    except Exception as e:
        print(f"✗ 无效参数处理失败: {e}")

if __name__ == "__main__":
    # 运行所有测试
    print("开始测试改进的EMG滤波方案...")
    
    # 基本功能测试
    quality_ok = test_filter_performance()
    
    # 多通道测试
    test_multi_channel()
    
    # 错误处理测试
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    if quality_ok:
        print("✓ 滤波方案工作正常，质量检查通过")
    else:
        print("⚠ 滤波方案工作正常，但质量检查未通过，请检查参数设置") 