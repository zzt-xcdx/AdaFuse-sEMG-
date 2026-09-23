import h5py
import scipy.io as scio
import numpy as np
import nina_funcs as nf
import pandas as pd
import os

# 数据集配置
dataset = 'DB4'  # 可以改为 'DB2' 或 'DB4'

if dataset == 'DB4':
    dir = 'D:/DB4'
    subject_range = range(1, 11)  # DB4有10个被试 (s1-s10)
    output_base = 'D:/DB4/data/restimulus/reraw'
else:  # DB2
    dir = 'D:/Pengxiangdong/ZX/DB2/Ninapro'
    subject_range = range(1, 41)  # DB2有40个被试
    output_base = 'D:/Pengxiangdong/ZX/DB2/data/restimulus/reraw'

def create_global_label_mapping_fast():
    """快速创建全局标签映射，正确处理不同Exercise中相同标签号代表不同手势的情况"""
    global_label_map = {}
    current_label = 1
    
    # 扫描所有被试的数据来建立完整的映射
    for subject_id in range(1, 11):
        for exercise in [1, 2, 3]:
            mat_file = f'{dir}/s{subject_id}/S{subject_id}_E{exercise}_A1.mat'
            if os.path.exists(mat_file):
                mat = scio.loadmat(mat_file)
                restimulus = mat['restimulus'].flatten()
                exercise_num = mat['exercise'][0, 0] if isinstance(mat['exercise'], np.ndarray) else mat['exercise']
                
                # 收集该exercise的所有非零标签
                unique_labels = np.unique(restimulus[restimulus > 0])
                
                for label in unique_labels:
                    key = (exercise_num, label)
                    if key not in global_label_map:
                        global_label_map[key] = current_label
                        current_label += 1
    
    # 按exercise分组显示统计
    exercise_stats = {}
    for (ex_num, old_label), new_label in global_label_map.items():
        if ex_num not in exercise_stats:
            exercise_stats[ex_num] = []
        exercise_stats[ex_num].append((old_label, new_label))
    
    print("各Exercise的标签统计:")
    for ex_num in sorted(exercise_stats.keys()):
        labels = sorted(exercise_stats[ex_num], key=lambda x: x[0])
        print(f"Exercise {ex_num}: {len(labels)}个手势")
        print(f"  原始标签: {[x[0] for x in labels]}")
        print(f"  全局标签: {[x[1] for x in labels]}")
    
    print(f"\n总共找到 {len(global_label_map)} 个手势")
    
    # 验证是否正好52个
    if len(global_label_map) == 52:
        print("✅ 成功找到所有52个手势!")
    else:
        print(f"❌ 期望52个手势，实际找到{len(global_label_map)}个")
    
    return global_label_map

def relabel_dataframe_fast(df, exercise_num, global_label_map):
    """快速重新标记DataFrame中的restimulus标签（向量化操作）"""
    df_copy = df.copy()
    restimulus_col = df_copy.columns[12]  # restimulus列
    
    # 创建映射数组
    old_labels = df_copy[restimulus_col].values
    new_labels = old_labels.copy()
    
    # 向量化映射
    for (ex_num, old_label), new_label in global_label_map.items():
        if ex_num == exercise_num:
            mask = (old_labels == old_label) & (old_labels > 0)
            new_labels[mask] = new_label
    
    df_copy[restimulus_col] = new_labels
    return df_copy

def process_subject_fast(subject_id, global_label_map):
    """快速处理单个被试的数据，并统计标签分布"""
    # 一次性读取所有数据
    df1 = nf.get_redata(dir + '/s' + str(subject_id), 'S' + str(subject_id) + '_E1_A1.mat')
    df2 = nf.get_redata(dir + '/s' + str(subject_id), 'S' + str(subject_id) + '_E2_A1.mat')
    df3 = nf.get_redata(dir + '/s' + str(subject_id), 'S' + str(subject_id) + '_E3_A1.mat')

    # 统计原始mat文件的stimulus/restimulus标签分布
    for ex, df, ex_name in zip([1,2,3], [df1,df2,df3], ['E1','E2','E3']):
        mat_file = f'{dir}/s{subject_id}/S{subject_id}_{ex}_A1.mat'
        if os.path.exists(mat_file):
            mat = scio.loadmat(mat_file)
            stimulus = mat['stimulus'].flatten()
            restimulus = mat['restimulus'].flatten()
            stim_labels = np.unique(stimulus[stimulus > 0])
            restim_labels = np.unique(restimulus[restimulus > 0])
            print(f"  {ex_name} stimulus标签:    {sorted(stim_labels)}")
            print(f"  {ex_name} restimulus标签: {sorted(restim_labels)}")

    # 从DataFrame中获取exercise编号（避免重复读取MAT文件）
    exercise1, exercise2, exercise3 = 1, 2, 3
    # 重新标记每个exercise的标签
    df1_relabeled = relabel_dataframe_fast(df1, exercise1, global_label_map)
    df2_relabeled = relabel_dataframe_fast(df2, exercise2, global_label_map)
    df3_relabeled = relabel_dataframe_fast(df3, exercise3, global_label_map)
    # 合并重新标记后的数据
    dfall = pd.concat([df1_relabeled, df2_relabeled, df3_relabeled], ignore_index=True)
    return dfall

if __name__ == '__main__':
    # 创建输出目录
    os.makedirs(output_base, exist_ok=True)
    
    # 为DB4创建全局标签映射
    if dataset == 'DB4':
        print("正在创建全局标签映射...")
        global_label_map = create_global_label_mapping_fast()
    
    for j in subject_range:
        print(f"\n处理被试 {j}...")
        
        if dataset == 'DB4':
            # DB4: 快速处理
            dfall = process_subject_fast(j, global_label_map)
            
            # 检查标签范围
            restimulus_col = dfall.columns[12]
            unique_labels = sorted(dfall[restimulus_col].unique())
            print(f"重新标记后的标签范围: {min(unique_labels)} - {max(unique_labels)}")
            print(f"唯一标签数量: {len(unique_labels)}")
            
        else:
            # DB2: 直接合并，不需要重新标记
            df1 = nf.get_redata(dir + '/s' + str(j), 'S' + str(j) + '_E1_A1.mat')
            df2 = nf.get_redata(dir + '/s' + str(j), 'S' + str(j) + '_E2_A1.mat')
            df3 = nf.get_redata(dir + '/s' + str(j), 'S' + str(j) + '_E3_A1.mat')
            dfall = pd.concat([df1, df2, df3], ignore_index=True)
        
        # 考虑数据放大到1
        dfmax = dfall.copy(deep=True)
        dfmax.iloc[:, :12] = dfmax.iloc[:, :12]
        df = dfmax.astype(np.float32)

        output_file = f'{output_base}/{dataset}_s{j}raw.h5'
        df.to_hdf(output_file, format='table', key='df', mode='w', complevel=9, complib='blosc')

        print(f'******************{dataset}_s{j}读取完成***********************')