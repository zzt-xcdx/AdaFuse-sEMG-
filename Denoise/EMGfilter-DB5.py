import os
import shutil
import pandas as pd
import sys

# 将项目根目录添加到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nina_funcs as nf

# 数据集配置
dataset = 'DB5'
root_data = 'D:/DB5'
subject_range = range(1, 11)

# 滤波配置
ENABLE_QUALITY_CHECK = False
USE_ENHANCED_SIGNALS = True

for j in subject_range:
    enhanced_file = os.path.join(root_data, f'data/restimulus/refilter_enhanced/{dataset}_s{j}filter_enhanced.h5')

    # 如果存在增强信号文件，则复制到目标目录
    if USE_ENHANCED_SIGNALS and os.path.exists(enhanced_file):
        output_dir = os.path.join(root_data, 'data/restimulus/refilter')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{dataset}_s{j}filter.h5')
        shutil.copy2(enhanced_file, output_file)
        continue  # 跳过后续处理

    # 读取原始数据
    df1 = pd.read_hdf(os.path.join(root_data, f'data/restimulus/reraw/{dataset}_s{j}raw.h5'), 'df')

    # 提取EMG数据和检查通道数量
    emg_columns = [col for col in df1.columns if col not in ['restimulus', 'rerepetition']]
    emg_channel_count = len(emg_columns)

    if emg_channel_count != 16:
        print(f"警告: 检测到的通道数量为 {emg_channel_count}，而预期为 16。")

    emg_data = df1.iloc[:, :emg_channel_count].values

    # 应用改进的EMG滤波方案
    if ENABLE_QUALITY_CHECK:
        emg_filtered = nf.adaptive_emg_filter(emg_data, fs=200, region='EU', check_quality=True)
    else:
        emg_filtered = nf.advanced_emg_filter(emg_data, fs=2000, region='EU', hp_cut=20, notch_freqs=(50, 100), lp_cut=99)

    # 创建DataFrame并添加标签
    dfemg_filtered = pd.DataFrame(emg_filtered, columns=range(emg_channel_count))
    dfemg_filtered['restimulus'] = df1['restimulus'].values
    dfemg_filtered['rerepetition'] = df1['rerepetition'].values

    # 保存滤波后的数据
    output_dir = os.path.join(root_data, 'data/restimulus/refilter')
    os.makedirs(output_dir, exist_ok=True)
    dfemg_filtered.to_hdf(os.path.join(output_dir, f'{dataset}_s{j}filter.h5'), format='table', key='df', mode='w',
                          complevel=9, complib='blosc')