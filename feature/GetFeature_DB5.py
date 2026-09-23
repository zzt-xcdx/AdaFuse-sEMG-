import sys
import os
# 将项目根目录添加到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import h5py
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.signal as signal
import matplotlib.pyplot as plt
import nina_funcs as nf
from scipy import signal
from tqdm import tqdm
import gc  # 添加垃圾回收

from Util.function import get_twoSet
from stockwell import st
from sklearn.decomposition import PCA # 导入 PCA
import joblib

# DB5配置
root_data = 'D:/DB5'
dataset = 'DB5'
subject_range = range(1, 11)  # DB5被试编号范围
CHANNEL = 16  # 16通道
WIN_LEN = 400  # 200ms窗口（插值后为400步）

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]

output_base_dir = os.path.join(root_data, 'data/restimulus/Fea')
os.makedirs(output_base_dir, exist_ok=True)

def fea_transpose(data):
    # 为了便于归一化，对矩阵进行转置
    arr_T = np.transpose(np.array(data))
    sc_fea = preprocessing.StandardScaler().fit_transform(arr_T)
    arr_fea = np.transpose(sc_fea)
    return arr_fea

def calculate_stockwell_and_pca_streaming(emg_data, rep_arr, n_components=64, batch_size=10):
    """
    流式计算Stockwell变换并立即进行PCA降维，避免存储巨大的特征矩阵
    """
    print("流式计算Stockwell变换和PCA降维...")
    
    # 确定训练集和测试集的索引
    train_indices = []
    test_indices = []
    for i in range(len(rep_arr)):
        rep = rep_arr[i]
        if rep in train_reps:
            train_indices.append(i)
        elif rep in test_reps:
            test_indices.append(i)
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")
    
    # 获取单个样本的特征维度
    dummy_channel = emg_data[0, :, 0].astype(np.float32)
    dummy_st = st.st(dummy_channel)
    single_channel_flat_len = dummy_st.flatten().shape[0]
    total_sample_flat_len = single_channel_flat_len * emg_data.shape[2]
    
    print(f"原始特征维度: {total_sample_flat_len}")
    print(f"PCA目标维度: {n_components}")
    
    # 使用增量式PCA
    from sklearn.decomposition import IncrementalPCA
    n_components = min(n_components, total_sample_flat_len)
    batch_size = max(32, n_components)
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # 第一阶段：在训练集上拟合PCA
    print("第一阶段：在训练集上拟合PCA...")
    n_train_batches = (len(train_indices) + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_train_batches), desc="Fitting PCA on training data"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]
        
        # 计算当前批次的Stockwell特征
        batch_features = []
        for idx in batch_indices:
            sample = emg_data[idx]
            sample_st_features = []
            for channel in sample.T:
                st_result = st.st(channel.astype(np.float32))
                st_magnitude = np.abs(st_result)
                sample_st_features.append(st_magnitude.flatten())
                del st_result, st_magnitude
            
            sample_features = np.concatenate(sample_st_features)
            batch_features.append(sample_features)
            del sample_st_features, sample_features
            gc.collect()
        
        batch_features = np.array(batch_features, dtype=np.float32)
        if np.any(~np.isfinite(batch_features)):
            raise ValueError(f"训练数据包含无效值(NaN或Inf)，批次 {i}")
        
        ipca.partial_fit(batch_features)
        del batch_features
        gc.collect()
    
    # 第二阶段：对所有数据进行转换
    print("第二阶段：对所有数据进行PCA转换...")
    n_samples = emg_data.shape[0]
    pca_results = np.zeros((n_samples, n_components), dtype=np.float32)
    
    n_transform_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_transform_batches), desc="Applying PCA Transformation"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        # 计算当前批次的Stockwell特征
        batch_features = []
        for idx in range(start_idx, end_idx):
            sample = emg_data[idx]
            sample_st_features = []
            for channel in sample.T:
                st_result = st.st(channel.astype(np.float32))
                st_magnitude = np.abs(st_result)
                sample_st_features.append(st_magnitude.flatten())
                del st_result, st_magnitude
            
            sample_features = np.concatenate(sample_st_features)
            batch_features.append(sample_features)
            del sample_st_features, sample_features
            gc.collect()
        
        batch_features = np.array(batch_features, dtype=np.float32)
        pca_batch = ipca.transform(batch_features)
        pca_results[start_idx:end_idx] = pca_batch
        
        del batch_features, pca_batch
        gc.collect()
    
    return pca_results, ipca

def main():
    """
    主函数：处理所有被试的数据
    """
    # Set random seed
    np.random.seed(42)
    
    # 记录处理结果
    results = {
        'success': [],
        'failed': []
    }
    
    for subject_id in subject_range:
        print(f"\n{'='*50}")
        print(f"处理被试 {subject_id}")
        print(f"{'='*50}")
        
        try:
            # 准备文件路径
            input_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{dataset}_s{subject_id}SegEnhance.h5')
            output_file = os.path.join(output_base_dir, f'{dataset}_s{subject_id}feaEnhance.h5')
            pca_file = os.path.join(output_base_dir, f'{dataset}_s{subject_id}_pca.pkl')
            
            # 检查输入文件
            if not os.path.exists(input_file):
                print(f"跳过文件 {input_file} - 文件不存在")
                continue
            
            print("1. 加载数据...")
            with h5py.File(input_file, 'r') as f:
                if 'emg' not in f or 'label' not in f or 'rep' not in f:
                    raise KeyError("输入文件缺少必要的数据集('emg'、'label'或'rep')")
                emg_data_shape = f['emg'].shape
                labels = f['label'][:]
                rep_arr = f['rep'][:]
            
            print(f"数据形状: {emg_data_shape}")
            print(f"标签数量: {len(labels)}")
            
            # 加载EMG数据
            print("2. 加载EMG数据...")
            with h5py.File(input_file, 'r') as f:
                emg_data = f['emg'][:]
            
            if np.any(~np.isfinite(emg_data)):
                raise ValueError("EMG数据包含无效值(NaN或Inf)")
            
            # 流式计算Stockwell变换和PCA
            print("3. 流式计算Stockwell变换和PCA...")
            emg_pca, pca_model = calculate_stockwell_and_pca_streaming(emg_data, rep_arr)
            del emg_data
            gc.collect()
            
            # 保存PCA模型
            joblib.dump(pca_model, pca_file)
            print(f"PCA模型已保存到: {pca_file}")
            
            # 保存结果
            print("4. 保存结果...")
            if emg_pca.size > 0:
                with h5py.File(output_file, 'w') as f:
                    f.create_dataset('features', data=emg_pca)
                    f.create_dataset('label', data=labels)
                    f.create_dataset('rep', data=rep_arr)
                print(f"特征已保存到: {output_file}")
                results['success'].append(subject_id)
            else:
                raise ValueError("PCA结果为空")
            
        except Exception as e:
            print(f"\n处理被试 {subject_id} 时出错:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            results['failed'].append((subject_id, str(e)))
    
    # 打印总结报告
    print("\n处理完成!")
    print(f"成功处理的被试: {len(results['success'])}")
    print(f"失败的被试: {len(results['failed'])}")
    
    if results['failed']:
        print("\n失败详情:")
        for subject_id, error in results['failed']:
            print(f"被试 {subject_id}: {error}")

if __name__ == '__main__':
    main() 