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
dataset = 'DB5'  # 适配DB5
root_data = 'D:/DB5'
subject_range = range(1, 11)  # DB5有10个被试 (s1-s10)

print(f"处理数据集: {dataset}")
print(f"被试范围: {list(subject_range)}")

for j in subject_range:
    print(f"\n处理被试 {j}...")
    df1 = pd.read_hdf(os.path.join(root_data, f'data/restimulus/reraw/{dataset}_s{j}raw.h5'), 'df')
    print("原始数据shape:", df1.shape, "列名:", df1.columns)
    
    # 自动检测EMG通道数量（排除标签列）
    emg_columns = [col for col in df1.columns if col not in ['restimulus', 'rerepetition']]
    emg_channel_count = len(emg_columns)
    print(f"检测到EMG通道数量: {emg_channel_count}")
    
    # 只对EMG通道做滤波，转为numpy array
    emg_data = df1.iloc[:, :emg_channel_count].values
    print(f"EMG数据形状: {emg_data.shape}")
    
    # ===================== 原来的滤波方案 =====================
    print("应用原来的滤波方案...")
    print("  步骤1: 带通滤波 (10-90Hz)")
    print("  步骤2: 陷波滤波 (50Hz)")
    
    # 步骤1: 带通滤波 - 使用refilter_data函数
    # 参数: f=(10, 90) - 10Hz到90Hz的带通滤波
    # butterworth_order=4 - 4阶Butterworth滤波器
    # btype='bandpass' - 带通类型
    emg_band = nf.refilter_data(data=emg_data, f=(10, 90), butterworth_order=4, btype='bandpass')
    
    # 步骤2: 陷波滤波 - 使用notch_refilter函数
    # 参数: f0=50 - 50Hz陷波频率
    # Q=30 - 品质因数
    # fs=200 - 采样频率(注意：这里采样频率设置错误，应该是2000)
    emg_notch = nf.notch_refilter(data=emg_band, f0=50, Q=30, fs=200)
    
    # ===================== 原来的滤波方案结束 =====================
    
    # 拼回标签
    dfemg_notch = pd.DataFrame(emg_notch, columns=range(emg_channel_count))
    dfemg_notch['restimulus'] = df1['restimulus'].values
    dfemg_notch['rerepetition'] = df1['rerepetition'].values
    print("滤波后数据shape:", dfemg_notch.shape, "列名:", dfemg_notch.columns)
    
    # 存储为h5文件
    output_dir = os.path.join(root_data, 'data/restimulus/refilter_original')
    os.makedirs(output_dir, exist_ok=True)
    dfemg_notch.to_hdf(os.path.join(output_dir, f'{dataset}_s{j}filter_original.h5'), format='table', key='df', mode='w', complevel=9, complib='blosc')
    print(f'******************{dataset}_s{j}原始滤波完成***********************') 