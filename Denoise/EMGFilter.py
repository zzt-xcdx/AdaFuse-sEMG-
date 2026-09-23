import h5py
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import sys
import os
# 将项目根目录添加到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nina_funcs as nf

# 数据集配置
dataset = 'DB2'  #
root_data = 'D:/DB2'
subject_range = range(1, 41)  # D4有10个被试 (s1-s10)

## 滤波配置
ENABLE_QUALITY_CHECK = False  # 设置为True启用质量检查
# 移除 USE_ENHANCED_SIGNALS 和 TARGET_SUBJECTS，因为db2不需要

print(f"处理数据集: {dataset}")
print(f"被试范围: {list(subject_range)}")

for j in subject_range:
    print(f"\n处理被试 {j}...")

    # 直接使用原始滤波方案
    print("使用原始滤波方案...")
    df1 = pd.read_hdf(os.path.join(root_data, f'data/restimulus/reraw/{dataset}_s{j}raw.h5'), 'df')
    print("原始数据shape:", df1.shape, "列名:", df1.columns)

    # 自动检测EMG通道数量（排除标签列）
    emg_columns = [col for col in df1.columns if col not in ['restimulus', 'rerepetition']]
    emg_channel_count = len(emg_columns)
    print(f"检测到EMG通道数量: {emg_channel_count}")

    # 只对EMG通道做滤波，转为numpy array
    emg_data = df1.iloc[:, :emg_channel_count].values
    print(f"EMG数据形状: {emg_data.shape}")

    # 使用改进的专业滤波方案
    print("应用改进的EMG滤波方案...")
    print("  高通20Hz → 陷波50Hz → 陷波100Hz → 低通450Hz")

    if ENABLE_QUALITY_CHECK:
        # 带质量检查的滤波
        emg_filtered = nf.adaptive_emg_filter(emg_data, fs=2000, region='EU', check_quality=True)
    else:
        # 基本滤波
        emg_filtered = nf.advanced_emg_filter(emg_data, fs=2000, region='EU')

    # 拼回标签
    dfemg_filtered = pd.DataFrame(emg_filtered, columns=range(emg_channel_count))
    dfemg_filtered['restimulus'] = df1['restimulus'].values
    dfemg_filtered['rerepetition'] = df1['rerepetition'].values
    print("滤波后数据shape:", dfemg_filtered.shape, "列名:", dfemg_filtered.columns)

    # 存储为h5文件
    output_dir = os.path.join(root_data, 'data/restimulus/refilter')
    os.makedirs(output_dir, exist_ok=True)
    dfemg_filtered.to_hdf(os.path.join(output_dir, f'{dataset}_s{j}filter.h5'),
                          format='table', key='df', mode='w', complevel=9, complib='blosc')
    print(f'******************{dataset}_s{j}滤波完成***********************')