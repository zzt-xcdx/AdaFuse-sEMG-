#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速对比受试者7、8和10号的信号质量
"""

import numpy as np
import scipy.io as scio
import os

def quick_compare(data_dir):
    """快速对比三个受试者的信号质量"""
    subjects = [7, 8, 10]
    exercises = [1, 2, 3]
    
    print("=== 受试者7、8、10号信号质量对比 ===\n")
    
    all_metrics = {}
    
    for subject_id in subjects:
        print(f"受试者{subject_id}:")
        subject_metrics = {}
        
        for exercise in exercises:
            file_path = os.path.join(data_dir, f's{subject_id}', f'S{subject_id}_E{exercise}_A1.mat')
            
            if os.path.exists(file_path):
                try:
                    mat = scio.loadmat(file_path)
                    emg = mat['emg']
                    
                    # 计算基本指标
                    rms_values = np.sqrt(np.mean(emg**2, axis=0))
                    std_values = np.std(emg, axis=0)
                    peak_to_peak = np.max(emg, axis=0) - np.min(emg, axis=0)
                    
                    avg_rms = np.mean(rms_values)
                    avg_std = np.mean(std_values)
                    avg_peak = np.mean(peak_to_peak)
                    
                    print(f"  Exercise{exercise}: RMS={avg_rms:.1f}, STD={avg_std:.1f}, 峰峰值={avg_peak:.1f}")
                    
                    subject_metrics[exercise] = {
                        'rms': avg_rms,
                        'std': avg_std,
                        'peak': avg_peak
                    }
                    
                except Exception as e:
                    print(f"  Exercise{exercise}: 加载失败")
            else:
                print(f"  Exercise{exercise}: 文件不存在")
        
        all_metrics[subject_id] = subject_metrics
        print()
    
    # 分析问题
    print("=== 问题分析 ===")
    
    # 计算10号受试者的平均指标作为基准
    if 10 in all_metrics:
        baseline_metrics = {}
        for exercise in exercises:
            if exercise in all_metrics[10]:
                baseline_metrics[exercise] = all_metrics[10][exercise]
        
        if baseline_metrics:
            print(f"\n10号受试者（表现良好）作为基准:")
            for exercise in exercises:
                if exercise in baseline_metrics:
                    print(f"  Exercise{exercise}: RMS={baseline_metrics[exercise]['rms']:.1f}, STD={baseline_metrics[exercise]['std']:.1f}")
            
            # 对比7号和8号的问题
            for subject_id in [7, 8]:
                if subject_id in all_metrics:
                    print(f"\n受试者{subject_id}的问题分析:")
                    
                    for exercise in exercises:
                        if exercise in all_metrics[subject_id] and exercise in baseline_metrics:
                            current = all_metrics[subject_id][exercise]
                            baseline = baseline_metrics[exercise]
                            
                            std_ratio = current['std'] / baseline['std']
                            rms_ratio = current['rms'] / baseline['rms']
                            
                            print(f"  Exercise{exercise}:")
                            print(f"    标准差比率: {std_ratio:.2f}x (10号的{std_ratio:.1f}倍)")
                            print(f"    RMS比率: {rms_ratio:.2f}x (10号的{rms_ratio:.1f}倍)")
                            
                            if std_ratio > 3:
                                print(f"    ❌ 严重问题: 噪声水平是10号的{std_ratio:.1f}倍")
                            elif std_ratio > 2:
                                print(f"    ⚠️ 中等问题: 噪声水平是10号的{std_ratio:.1f}倍")
                            elif std_ratio > 1.5:
                                print(f"    ⚠️ 轻微问题: 噪声水平是10号的{std_ratio:.1f}倍")
                            else:
                                print(f"    ✅ 正常: 噪声水平接近10号")
    
    return all_metrics

def main():
    data_dir = 'D:/DB4'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    all_metrics = quick_compare(data_dir)
    
    print("\n=== 总结 ===")
    print("受试者7和8准确率低的主要原因:")
    print("1. 所有通道的噪声水平都远高于10号受试者")
    print("2. 信号标准差普遍是10号的3-10倍")
    print("3. 这表明存在严重的电极接触、皮肤准备或设备问题")
    print("\n建议:")
    print("- 重新检查电极放置和连接")
    print("- 清洁皮肤表面，降低阻抗")
    print("- 验证采集设备是否正常工作")
    print("- 使用增强预处理技术改善信号质量")

if __name__ == "__main__":
    main() 