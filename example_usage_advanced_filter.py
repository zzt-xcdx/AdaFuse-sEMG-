#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的EMG滤波方案使用示例
展示如何在现有预处理流程中集成新的滤波方案
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入新的滤波函数
from nina_funcs import advanced_emg_filter, adaptive_emg_filter, check_filter_quality

def example_1_basic_usage():
    """
    示例1: 基本使用方法
    """
    print("=" * 60)
    print("示例1: 基本使用方法")
    print("=" * 60)
    
    # 生成模拟EMG数据
    fs = 2000
    duration = 1.0  # 1秒
    n_samples = int(fs * duration)
    n_channels = 4
    
    # 创建多通道EMG数据
    emg_data = np.random.randn(n_samples, n_channels)
    
    print(f"原始EMG数据形状: {emg_data.shape}")
    print(f"采样频率: {fs} Hz")
    
    # 使用改进的滤波方案
    filtered_data = advanced_emg_filter(emg_data, fs=fs, region='EU')
    
    print(f"滤波后数据形状: {filtered_data.shape}")
    print("✓ 基本滤波完成")
    
    return emg_data, filtered_data

def example_2_with_quality_check():
    """
    示例2: 带质量检查的滤波
    """
    print("\n" + "=" * 60)
    print("示例2: 带质量检查的滤波")
    print("=" * 60)
    
    # 生成包含干扰的测试信号
    fs = 2000
    duration = 2.0
    n_samples = int(fs * duration)
    
    # 创建包含工频干扰的信号
    t = np.linspace(0, duration, n_samples)
    clean_signal = np.random.normal(0, 1, n_samples)
    power_line_50 = 2.0 * np.sin(2 * np.pi * 50 * t)
    power_line_100 = 1.5 * np.sin(2 * np.pi * 100 * t)
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    
    noisy_signal = clean_signal + power_line_50 + power_line_100 + baseline_drift
    
    print(f"含干扰信号长度: {len(noisy_signal)}")
    
    # 使用带质量检查的滤波
    filtered_signal = adaptive_emg_filter(
        noisy_signal, 
        fs=fs, 
        region='EU', 
        check_quality=True
    )
    
    print("✓ 带质量检查的滤波完成")
    
    return noisy_signal, filtered_signal

def example_3_integration_with_existing_code():
    """
    示例3: 与现有代码集成
    """
    print("\n" + "=" * 60)
    print("示例3: 与现有代码集成")
    print("=" * 60)
    
    # 模拟从现有预处理流程中获取的数据
    fs = 2000
    n_samples = 4000
    n_channels = 12  # 模拟DB2的12通道
    
    # 模拟原始EMG数据
    raw_emg = np.random.randn(n_samples, n_channels)
    
    print("步骤1: 加载原始EMG数据")
    print(f"  数据形状: {raw_emg.shape}")
    
    # 替换原有的简单滤波
    print("\n步骤2: 应用改进的滤波方案")
    print("  原方法: 简单的带通滤波")
    print("  新方法: 高通20Hz → 陷波50/100Hz → 低通450Hz")
    
    filtered_emg = advanced_emg_filter(raw_emg, fs=fs, region='EU')
    
    print(f"  滤波后数据形状: {filtered_emg.shape}")
    
    # 继续后续处理
    print("\n步骤3: 继续后续处理")
    print("  - 数据分割")
    print("  - 特征提取")
    print("  - 标准化")
    print("  - 模型训练")
    
    print("✓ 集成示例完成")
    
    return raw_emg, filtered_emg

def example_4_comparison_with_old_method():
    """
    示例4: 与旧方法对比
    """
    print("\n" + "=" * 60)
    print("示例4: 与旧方法对比")
    print("=" * 60)
    
    # 导入旧的滤波方法
    from nina_funcs import bandpass_filter
    
    fs = 2000
    duration = 1.0
    n_samples = int(fs * duration)
    
    # 创建测试信号
    t = np.linspace(0, duration, n_samples)
    clean_signal = np.random.normal(0, 1, n_samples)
    power_line_50 = 2.0 * np.sin(2 * np.pi * 50 * t)
    power_line_100 = 1.5 * np.sin(2 * np.pi * 100 * t)
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    
    test_signal = clean_signal + power_line_50 + power_line_100 + baseline_drift
    
    print("对比测试信号包含:")
    print("  - 基础EMG信号")
    print("  - 50Hz工频干扰")
    print("  - 100Hz工频干扰")
    print("  - 基线漂移")
    
    # 旧方法: 简单带通滤波
    print("\n旧方法: 简单带通滤波 (20-450Hz)")
    old_filtered = bandpass_filter(test_signal, 20, 450, fs)
    
    # 新方法: 改进的滤波方案
    print("新方法: 改进的滤波方案")
    new_filtered = advanced_emg_filter(test_signal, fs=fs, region='EU')
    
    # 质量对比
    print("\n质量对比:")
    print("旧方法质量检查:")
    old_quality = check_filter_quality(test_signal, old_filtered, fs=fs)
    
    print("\n新方法质量检查:")
    new_quality = check_filter_quality(test_signal, new_filtered, fs=fs)
    
    print(f"\n结论:")
    print(f"  旧方法质量: {'✓ 合格' if old_quality else '✗ 不合格'}")
    print(f"  新方法质量: {'✓ 合格' if new_quality else '✗ 不合格'}")
    
    return test_signal, old_filtered, new_filtered

def main():
    """
    主函数：运行所有示例
    """
    print("改进的EMG滤波方案使用示例")
    print("=" * 60)
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_with_quality_check()
    example_3_integration_with_existing_code()
    example_4_comparison_with_old_method()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
    
    print("\n使用建议:")
    print("1. 在现有预处理流程中，将简单的带通滤波替换为 advanced_emg_filter()")
    print("2. 如果需要质量检查，使用 adaptive_emg_filter() 并设置 check_quality=True")
    print("3. 根据地区设置 region 参数: 'EU' (50/100Hz) 或 'US' (60/120Hz)")
    print("4. 新方案完全向后兼容，不会破坏现有代码")

if __name__ == "__main__":
    main() 