#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量预处理DB5数据集的Stockwell时频图
生成(N, 16, 32, 400)的时频图并保存为h5文件
"""

import os
import numpy as np
import h5py
from tqdm import tqdm
import sys
sys.path.append('D:/DB5/stockwell')
from stockwell import st

def compute_stockwell_batch(emg_data, freq_bins=32):
    """
    批量计算Stockwell变换
    Args:
        emg_data: (N, 400, 16) 原始EMG数据
        freq_bins: 频点数量，默认32
    Returns:
        stockwell_tf: (N, 16, 32, 400) 时频图
    """
    n_samples, time_len, n_channels = emg_data.shape
    print(f"处理数据形状: {emg_data.shape}")
    print(f"生成时频图形状: ({n_samples}, {n_channels}, {freq_bins}, {time_len})")
    
    # 预分配内存
    all_tf = np.zeros((n_samples, n_channels, freq_bins, time_len), dtype=np.float32)
    
    # 批量处理
    for i in tqdm(range(n_samples), desc="计算Stockwell变换"):
        for ch in range(n_channels):
            # 计算单个通道的Stockwell变换
            signal = emg_data[i, :, ch].astype(np.float32)
            st_result = st.st(signal)
            
            # 取幅度谱，只保留前freq_bins个频点
            st_magnitude = np.abs(st_result)[:freq_bins, :]
            all_tf[i, ch] = st_magnitude
    
    return all_tf

def process_subject(subject_id, input_dir, output_dir):
    """
    处理单个被试的数据
    """
    # 输入文件路径
    emg_file = os.path.join(input_dir, f'DB5_s{subject_id}SegEnhance.h5')
    
    # 输出文件路径
    os.makedirs(output_dir, exist_ok=True)
    stockwell_file = os.path.join(output_dir, f'DB5_s{subject_id}_stockwell.h5')
    
    print(f"\n{'='*50}")
    print(f"处理被试 {subject_id}")
    print(f"输入文件: {emg_file}")
    print(f"输出文件: {stockwell_file}")
    print(f"{'='*50}")
    
    # 检查输入文件是否存在
    if not os.path.exists(emg_file):
        print(f"错误: 输入文件不存在 {emg_file}")
        return False
    
    try:
        # 读取原始EMG数据
        print("读取EMG数据...")
        with h5py.File(emg_file, 'r') as f:
            emg = f['emg'][:]  # (N, 400, 16)
            label = f['label'][:]
            rep = f['rep'][:]
        
        print(f"EMG数据形状: {emg.shape}")
        print(f"标签数量: {len(label)}")
        print(f"重复次数: {len(rep)}")
        
        # 计算Stockwell变换
        print("计算Stockwell时频图...")
        stockwell_tf = compute_stockwell_batch(emg, freq_bins=32)
        
        # 保存结果
        print("保存时频图...")
        with h5py.File(stockwell_file, 'w') as f:
            f.create_dataset('stockwell', data=stockwell_tf, compression='gzip', compression_opts=9)
            f.create_dataset('label', data=label)
            f.create_dataset('rep', data=rep)
            
            # 添加元数据
            f.attrs['subject_id'] = subject_id
            f.attrs['shape'] = stockwell_tf.shape
            f.attrs['freq_bins'] = 32
            f.attrs['time_points'] = 400
            f.attrs['channels'] = 16
        
        # 验证保存的数据
        print("验证保存的数据...")
        with h5py.File(stockwell_file, 'r') as f:
            saved_tf = f['stockwell'][:]
            saved_label = f['label'][:]
            saved_rep = f['rep'][:]
        
        print(f"保存的时频图形状: {saved_tf.shape}")
        print(f"保存的标签数量: {len(saved_label)}")
        print(f"保存的重复次数: {len(saved_rep)}")
        
        # 计算文件大小
        file_size_mb = os.path.getsize(stockwell_file) / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")
        
        print(f"被试 {subject_id} 处理完成！")
        return True
        
    except Exception as e:
        print(f"处理被试 {subject_id} 时出错: {str(e)}")
        return False

def main():
    """
    主函数：批量处理所有被试
    """
    # 配置路径
    input_dir = "D:/DB5/data/restimulus/reSegEnhance"
    output_dir = "D:/DB5/data/restimulus/Stockwell"
    
    # 被试列表 (DB5有10个被试)
    subjects = list(range(1, 11))
    
    print("DB5 Stockwell时频图批量预处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理被试: {subjects}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    success_count = 0
    for subject_id in subjects:
        if process_subject(subject_id, input_dir, output_dir):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"批量预处理完成！")
    print(f"成功处理: {success_count}/{len(subjects)} 个被试")
    print(f"输出目录: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 