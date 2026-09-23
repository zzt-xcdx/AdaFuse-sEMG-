import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 将 nina_funcs.py 所在的目录添加到 sys.path
# 假设 nina_funcs.py 在项目根目录下
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 从 nina_funcs 导入处理函数和特征提取函数
try:
    # 尝试导入所有处理和特征提取相关的函数
    from nina_funcs import bandpass_filter, windowing, \
                         feature_extractor, frequency_features_extractor, \
                         mav, iemg, wl, rms, var, aac, zero_cross, ssc
    print("成功导入 nina_funcs 中的处理和特征提取函数")
except ImportError as e:
    print(f"Error importing nina_funcs functions: {e}")
    print("请检查 nina_funcs.py 文件及其依赖项是否存在错误。")
    sys.exit()

# 定义原始数据文件路径 (使用实际存在的原始数据文件路径)
raw_mat_file = 'D:/DB2/DB2_s1/S1_E1_A1.mat'

def visualize_steps():
    # 1. 加载原始数据
    print(f"加载原始数据: {raw_mat_file}")
    try:
        data = sio.loadmat(raw_mat_file)
        emg_data = data['emg']
        # 需要处理 restimulus 和 rerepetition 数据，以便 windowing 函数使用
        restimulus_data = data['restimulus'].flatten()
        rerepetition_data = data['rerepetition'].flatten()
        print(f"原始EMG数据形状: {emg_data.shape}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {raw_mat_file}")
        return
    except Exception as e:
        print(f"加载.mat文件时发生错误: {e}")
        return

    # 选择一个短时间段进行可视化
    sample_length = 2000 # 采样点数
    sample_emg = emg_data[:sample_length, :]

    plt.figure(figsize=(18, 15))
    plt.suptitle('EMG数据处理可视化', fontsize=16)

    # Plot 原始EMG信号
    plt.subplot(5, 1, 1)
    plt.plot(sample_emg[:, 0]) # 绘制第一个通道
    plt.title('1. 原始EMG信号 (通道 0)')
    plt.ylabel('幅值')

    # 2. 应用信号增强 (使用 bandpass_filter 和 Zscore)
    print("应用信号增强...")
    try:
        # 应用带通滤波
        lowcut = 20.0 # Hz
        highcut = 450.0 # Hz
        fs = 2000 # Hz (根据项目设置)
        
        # bandpass_filter 期望输入是一个一维信号和滤波参数
        # 我们需要对每个通道单独进行滤波
        
        sample_enhanced_emg = np.zeros_like(sample_emg)
        for i in range(sample_emg.shape[1]): # 遍历通道
            sample_enhanced_emg[:, i] = bandpass_filter(sample_emg[:, i], lowcut, highcut, fs)
        
        # 应用 Z-score 标准化 (对滤波后的数据)
        # 手动实现一个简单的Z-score标准化，用于可视化
        mean = np.mean(sample_enhanced_emg, axis=0)
        std = np.std(sample_enhanced_emg, axis=0)
        # 避免除以零
        std[std == 0] = 1e-8
        sample_enhanced_emg_normalized = (sample_enhanced_emg - mean) / std

    except Exception as e:
        print(f"信号增强时发生错误: {e}")
        return

    # Plot 信号增强后的EMG信号
    plt.subplot(5, 1, 2)
    plt.plot(sample_enhanced_emg_normalized[:, 0]) # 绘制第一个通道
    plt.title('2. 信号增强 (滤波+局部Zscore, 通道 0)')
    plt.ylabel('幅值')

    # 3. 应用数据分割 (窗口化)
    print("应用数据分割 (窗口化)...")
    window_size = 400 # 200ms窗口大小 (400个数据点)
    step_size = 80   # 40ms步长 (80个数据点)
    try:
        # windowing 函数期望的输入是：数据, 重复次数, 手势标签, 窗口大小, 步长
        # 并且数据、重复次数和手势标签应该是 Pandas DataFrame 的形式
        # 为了匹配 windowing 函数，我们需要先将 numpy 数组转换为 Pandas DataFrame
        
        # 创建一个包含emg, restimulus, rerepetition 的 DataFrame
        # 注意列名要匹配 windowing 函数内部的硬编码 ('stimulus'/'restimulus', 'repetition'/'rerepetition')
        # 假设 windowing 期望的列是 0-11 是 EMG, 12 是 stimulus/restimulus, 13 是 repetition/rerepetition
        # 根据 nina_funcs.py 中 windowing 函数的定义，它期望 data 是一个 DataFrame，EMG 数据在前12列，
        # 第12列是 stimulus/restimulus，第13列是 repetition/rerepetition
        
        emg_df = pd.DataFrame(emg_data)
        emg_df['restimulus'] = restimulus_data
        emg_df['rerepetition'] = rerepetition_data
        
        # windowing 函数的 reps 和 gestures 参数如果为 None 或 [] 会保留所有数据
        # 为了可视化，我们先不对重复次数和手势进行筛选
        segmented_data, labels, reps = windowing(emg_df, None, None, window_size, step_size)
        
        print(f"分割后数据形状: {segmented_data.shape}")
        # 选择前几个窗口进行可视化
        num_windows_to_show = 3
        # 确保 segmented_data 至少有 num_windows_to_show 个样本
        if segmented_data.shape[0] < num_windows_to_show:
             num_windows_to_show = segmented_data.shape[0]
        if num_windows_to_show == 0:
            print("错误：分割后没有生成任何窗口。")
            plt.subplot(5, 1, 3)
            plt.text(0.5, 0.5, "分割后没有生成任何窗口", ha='center', va='center')
            plt.title('3. 数据分割')
            plt.ylabel('幅值')
            # 在没有分割数据的情况下，后续步骤无法进行，直接返回
            plt.tight_layout()
            plt.savefig('emg_processing_visualization.png')
            plt.close()
            print("已生成 emg_processing_visualization.png (部分步骤)")
            return

        sample_segmented_data = segmented_data[:num_windows_to_show, :, :]

    except Exception as e:
        print(f"数据分割时发生错误: {e}")
        return

    # Plot 分割后的EMG信号片段
    plt.subplot(5, 1, 3)
    for i in range(num_windows_to_show):
        plt.plot(sample_segmented_data[i, :, 0] + i * np.max(np.abs(sample_segmented_data[i, :, 0]))) # 绘制第一个通道，并错开显示
    plt.title(f'3. 数据分割 (前 {num_windows_to_show} 个窗口, 通道 0)')
    plt.ylabel('幅值 (错开显示)')

    # 4. 应用特征提取
    print("应用特征提取...")
    try:
        # ----- 调用 nina_funcs 中的特征提取函数 -----
        # feature_extractor 期望输入: features (list), shape (tuple), data (numpy array)
        # frequency_features_extractor 期望输入: shape (tuple), data (numpy array), fs=2000
        
        # 时域特征函数列表 (与 GetFeature.py 中的定义一致)
        td_feature_functions = [mav, iemg, wl, rms, var, aac, zero_cross, ssc]
        
        # 对选取的样本数据 (sample_segmented_data) 应用特征提取
        # sample_segmented_data 的形状是 (num_windows_to_show, window_size, n_channels)
        # 这与 nina_funcs 中的函数期望的 (样本数, 时间步长, 通道数) 形状匹配
        
        sample_td_features = feature_extractor(td_feature_functions, sample_segmented_data.shape, sample_segmented_data)
        # 传递 sample_segmented_data.shape 作为 shape 参数
        sample_fd_features = frequency_features_extractor(sample_segmented_data.shape, sample_segmented_data, fs=2000)
        
        # 组合时域和频域特征
        sample_features = np.concatenate((sample_td_features, sample_fd_features), axis=1)
        
        print(f"提取的样本特征形状: {sample_features.shape}")

    except Exception as e:
        print(f"特征提取时发生错误: {e}")
        return

    # Plot 提取的特征向量 (展示第一个窗口的特征)
    plt.subplot(5, 1, 4)
    # 只需要展示第一个窗口的特征
    if sample_features.shape[0] > 0:
        plt.bar(range(sample_features.shape[1]), sample_features[0, :])
        plt.title('4. 提取的特征向量 (第一个窗口)')
        plt.xlabel('特征维度')
        plt.ylabel('特征值')
    else:
         plt.text(0.5, 0.5, "特征提取失败或无特征", ha='center', va='center')

    # 5. 数据标准化 (对提取的特征进行标准化)
    print("应用数据标准化...")
    try:
        # 对提取的样本特征进行局部Zscore标准化，用于可视化
        mean_features = np.mean(sample_features, axis=0)
        std_features = np.std(sample_features, axis=0)
        # 避免除以零
        std_features[std_features == 0] = 1e-8
        sample_features_normalized = (sample_features - mean_features) / std_features

    except Exception as e:
        print(f"数据标准化时发生错误: {e}")
        return

    # Plot 标准化后的特征向量
    plt.subplot(5, 1, 5)
    plt.bar(range(sample_features_normalized.shape[1]), sample_features_normalized[0, :])
    plt.title('5. 标准化后的特征向量 (第一个窗口)')
    plt.xlabel('特征维度')
    plt.ylabel('标准化特征值')

    plt.tight_layout()
    plt.savefig('emg_processing_visualization.png')
    plt.close()
    print("已生成 emg_processing_visualization.png")

if __name__ == '__main__':
    visualize_steps() 