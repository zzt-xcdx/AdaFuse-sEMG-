#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的特征提取脚本
专注于解决LDA问题，避免复杂的Stockwell变换
"""

import os
import h5py
import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 数据集配置
dataset = 'DB4'
root_data = 'D:/DB4'

def extract_simple_features(emg_data):
    """提取简单的时域特征，避免复杂的Stockwell变换"""
    print("提取简单时域特征...")
    
    n_samples, n_channels, n_samples_per_channel = emg_data.shape
    
    # 简单的时域特征
    features = []
    
    for i in range(n_samples):
        sample_features = []
        for ch in range(n_channels):
            signal = emg_data[i, ch, :]
            
            # 基本统计特征
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            rms_val = np.sqrt(np.mean(signal**2))
            max_val = np.max(signal)
            min_val = np.min(signal)
            
            # 过零率
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
            
            # 绝对均值
            abs_mean = np.mean(np.abs(signal))
            
            # 峰峰值
            peak_to_peak = max_val - min_val
            
            # 方差
            variance = np.var(signal)
            
            # 偏度
            skewness = np.mean(((signal - mean_val) / std_val) ** 3) if std_val > 0 else 0
            
            # 峰度
            kurtosis = np.mean(((signal - mean_val) / std_val) ** 4) if std_val > 0 else 0
            
            channel_features = [
                mean_val, std_val, rms_val, max_val, min_val,
                zero_crossings, abs_mean, peak_to_peak, variance, skewness, kurtosis
            ]
            
            sample_features.extend(channel_features)
        
        features.append(sample_features)
    
    features = np.array(features, dtype=np.float32)
    print(f"特征形状: {features.shape}")
    return features

def apply_pca_and_lda(features, labels, subject_id):
    """应用PCA和LDA"""
    print("应用PCA和LDA...")
    
    # 1. 提取手势标签
    gesture_labels = []
    for i in range(len(labels)):
        if isinstance(labels[i], np.void):
            if 'gesture' in labels[i].dtype.names:
                gesture_labels.append(labels[i]['gesture'])
            else:
                gesture_labels.append(labels[i][0])
        else:
            gesture_labels.append(labels[i])
    
    gesture_labels = np.array(gesture_labels)
    n_classes = len(np.unique(gesture_labels))
    print(f"手势类别数: {n_classes}")
    
    # 2. 确定训练集和测试集
    train_reps = [1, 3, 4, 6]
    test_reps = [2, 5]
    
    train_indices = []
    test_indices = []
    for i in range(len(labels)):
        rep = labels[i]['repetition'] if isinstance(labels[i], np.void) else labels[i]
        if rep in train_reps:
            train_indices.append(i)
        elif rep in test_reps:
            test_indices.append(i)
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")
    
    # 3. 应用PCA
    print("应用PCA...")
    n_components = min(64, features.shape[1])  # 确保不超过特征数
    pca = PCA(n_components=n_components, random_state=42)
    
    # 只在训练集上拟合PCA
    pca.fit(features[train_indices])
    pca_features = pca.transform(features)
    
    print(f"PCA特征形状: {pca_features.shape}")
    
    # 4. 应用LDA
    print("应用LDA...")
    
    # 检查训练集类别
    train_gesture_labels = gesture_labels[train_indices]
    train_unique_classes = np.unique(train_gesture_labels)
    print(f"训练集类别数: {len(train_unique_classes)}")
    
    # 计算LDA维度约束
    n_features = pca_features.shape[1]
    max_lda_dim = min(n_features, len(train_unique_classes) - 1)
    lda_dim = max_lda_dim
    
    print(f"PCA特征数: {n_features}")
    print(f"LDA降维到: {lda_dim} 维")
    
    if lda_dim <= 0:
        raise ValueError(f"LDA维度必须大于0，当前: {lda_dim}")
    
    # 训练LDA
    lda = LDA(n_components=lda_dim, solver='svd', shrinkage='auto')
    lda.fit(pca_features[train_indices], train_gesture_labels)
    
    # 应用LDA变换
    lda_features = lda.transform(pca_features)
    print(f"LDA特征形状: {lda_features.shape}")
    
    # 5. 保存模型
    model_file = f'pca{len(train_unique_classes)}_lda{lda_dim}_subject{subject_id}_simple.pkl'
    joblib.dump({
        'pca': pca,
        'lda': lda
    }, model_file)
    
    print(f"模型保存到: {model_file}")
    
    return lda_features

def process_subject(subject_id):
    """处理单个被试"""
    print(f"\n{'='*50}")
    print(f"处理被试 {subject_id}")
    print(f"{'='*50}")
    
    try:
        # 1. 加载数据
        input_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/DB4_s{subject_id}SegEnhance.h5')
        output_file = os.path.join(root_data, f'data/restimulus/Fea/DB4_s{subject_id}feaEnhance.h5')
        
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return False
        
        print("1. 加载数据...")
        with h5py.File(input_file, 'r') as f:
            emg_data = f['emg'][:]
            labels = f['label'][:]
        
        print(f"数据形状: {emg_data.shape}")
        print(f"标签数量: {len(labels)}")
        
        # 2. 提取特征
        print("\n2. 提取特征...")
        features = extract_simple_features(emg_data)
        
        # 3. 应用PCA和LDA
        print("\n3. 应用PCA和LDA...")
        final_features = apply_pca_and_lda(features, labels, subject_id)
        
        # 4. 保存结果
        print("\n4. 保存结果...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('features', data=final_features)
            f.create_dataset('label', data=labels)
        
        print(f"✅ 成功完成被试 {subject_id}")
        print(f"结果保存到: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理被试 {subject_id} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 简化特征提取脚本 ===")
    print("使用简单时域特征，避免复杂的Stockwell变换")
    
    # 处理被试7和8
    subjects = [7, 8]
    results = {'success': [], 'failed': []}
    
    for subject_id in subjects:
        success = process_subject(subject_id)
        
        if success:
            results['success'].append(subject_id)
        else:
            results['failed'].append(subject_id)
    
    # 打印总结
    print(f"\n{'='*50}")
    print("处理完成!")
    print(f"成功: {len(results['success'])} 个被试")
    print(f"失败: {len(results['failed'])} 个被试")
    
    if results['success']:
        print(f"成功的被试: {results['success']}")
    
    if results['failed']:
        print(f"失败的被试: {results['failed']}")

if __name__ == "__main__":
    main() 