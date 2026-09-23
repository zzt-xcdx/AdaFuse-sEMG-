import h5py
import scipy.io as scio
import numpy as np
import pandas as pd
import os

# 数据集配置
dataset = 'DB5'
dir = 'D:/DB5'
subject_range = range(1, 11)  # DB5有10个被试
default_output_base = 'D:/DB5/data/restimulus/reraw'
output_base = default_output_base


def create_global_label_mapping_db5():
    """为DB5创建全局标签映射，确保52类手势唯一编号"""
    global_label_map = {}
    current_label = 1
    for subject_id in subject_range:
        for exercise in [1, 2, 3]:
            mat_file = f'{dir}/s{subject_id}/S{subject_id}_E{exercise}_A1.mat'
            if os.path.exists(mat_file):
                mat = scio.loadmat(mat_file)
                restimulus = mat['restimulus'].flatten()
                exercise_num = mat['exercise'][0, 0] if isinstance(mat['exercise'], np.ndarray) else mat['exercise']
                unique_labels = np.unique(restimulus[restimulus > 0])
                for label in unique_labels:
                    key = (exercise_num, label)
                    if key not in global_label_map:
                        global_label_map[key] = current_label
                        current_label += 1
    print(f"共找到 {len(global_label_map)} 个全局手势标签")
    if len(global_label_map) == 52:
        print("✅ 成功找到所有52个手势!")
    else:
        print(f"❌ 期望52个手势，实际找到{len(global_label_map)}个")
    return global_label_map


def relabel_dataframe_db5(df, exercise_num, global_label_map):
    """将restimulus标签重新映射为全局唯一标签"""
    df_copy = df.copy()
    restimulus_col = df_copy.columns[16]  # DB5: 第17列是restimulus
    old_labels = df_copy[restimulus_col].values
    new_labels = old_labels.copy()
    for (ex_num, old_label), new_label in global_label_map.items():
        if ex_num == exercise_num:
            mask = (old_labels == old_label) & (old_labels > 0)
            new_labels[mask] = new_label
    df_copy[restimulus_col] = new_labels
    return df_copy


def process_subject_db5(subject_id, global_label_map):
    """处理单个被试，合并3个练习，重映射标签，严格要求16通道EMG"""
    dfs = []
    for exercise in [1, 2, 3]:
        mat_file = f'{dir}/s{subject_id}/S{subject_id}_E{exercise}_A1.mat'
        if not os.path.exists(mat_file):
            print(f"文件不存在: {mat_file}")
            continue
        mat = scio.loadmat(mat_file)
        emg = mat['emg']
        if emg.shape[1] != 16:
            raise ValueError(f"{mat_file} 的emg通道数为{emg.shape[1]}，不是16通道，请检查原始数据！")
        restimulus = mat['restimulus'].flatten()
        rerepetition = mat['rerepetition'].flatten()
        df = pd.DataFrame(emg)
        df['restimulus'] = restimulus
        df['rerepetition'] = rerepetition
        exercise_num = mat['exercise'][0, 0] if isinstance(mat['exercise'], np.ndarray) else mat['exercise']
        df = relabel_dataframe_db5(df, exercise_num, global_label_map)
        dfs.append(df)
    if not dfs:
        print(f"被试{subject_id}无有效数据，跳过")
        return None
    dfall = pd.concat(dfs, ignore_index=True)
    return dfall


if __name__ == '__main__':
    os.makedirs(output_base, exist_ok=True)
    print("正在创建全局标签映射...")
    global_label_map = create_global_label_mapping_db5()
    for j in subject_range:
        print(f"\n处理被试 {j}...")
        dfall = process_subject_db5(j, global_label_map)
        if dfall is None:
            continue
        # 只保留float32类型
        df = dfall.astype(np.float32)
        output_file = f'{output_base}/{dataset}_s{j}raw.h5'
        df.to_hdf(output_file, format='table', key='df', mode='w', complevel=9, complib='blosc')
        print(f'******************{dataset}_s{j}读取完成***********************') 