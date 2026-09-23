#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试信号分析，深入检查问题
"""

import numpy as np
import scipy.io as scio
import os

def debug_signal_analysis(data_dir):
    """调试信号分析"""
    subjects = [7, 8, 10]
    exercises = [1, 2, 3]
    
    print("=== 深入调试信号分析 ===\n")
    
    for subject_id in subjects:
        print(f"受试者{subject_id}:")
        
        for exercise in exercises:
            file_path = os.path.join(data_dir, f's{subject_id}', f'S{subject_id}_E{exercise}_A1.mat')
            
            if os.path.exists(file_path):
                try:
                    mat = scio.loadmat(file_path)
                    emg = mat['emg']
                    
                    print(f"  Exercise{exercise}:")
                    print(f"    数据形状: {emg.shape}")
                    
                    # 检查每个通道的详细情况
                    for ch in range(min(3, emg.shape[1])):  # 只看前3个通道
                        signal_ch = emg[:, ch]
                        
                        # 基本统计
                        rms = np.sqrt(np.mean(signal_ch**2))
                        std = np.std(signal_ch)
                        mean_val = np.mean(signal_ch)
                        min_val = np.min(signal_ch)
                        max_val = np.max(signal_ch)
                        
                        # 检查是否有异常值
                        q99 = np.percentile(signal_ch, 99)
                        q1 = np.percentile(signal_ch, 1)
                        
                        print(f"      通道{ch+1}: RMS={rms:.1f}, STD={std:.1f}, 均值={mean_val:.1f}")
                        print(f"        范围: [{min_val:.1f}, {max_val:.1f}], 99%分位数={q99:.1f}")
                        
                        # 检查是否有极端值
                        extreme_count = np.sum(np.abs(signal_ch) > 1000)
                        if extreme_count > 0:
                            print(f"        ⚠️ 发现{extreme_count}个极端值(>1000)")
                    
                    print()
                    
                except Exception as e:
                    print(f"  Exercise{exercise}: 加载失败 - {e}")
            else:
                print(f"  Exercise{exercise}: 文件不存在")
        
        print("-" * 50)

def main():
    data_dir = 'D:/DB4'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    debug_signal_analysis(data_dir)

if __name__ == "__main__":
    main() 