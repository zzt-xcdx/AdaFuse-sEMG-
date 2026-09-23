#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只应用LDA到已完成的PCA特征
避免重复的Stockwell变换和PCA计算
"""

import os
import h5py
import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import IncrementalPCA
import warnings
warnings.filterwarnings('ignore')

# 数据集配置
dataset = 'DB4'
root_data = 'D:/DB4'

def load_pca_features(subject_id):
    """加载已完成的PCA特征"""
    pca_file = f'pca64_lda46_subject{subject_id}.pkl'
    
    if os.path.exists(pca_file):
        print(f"找到已保存的PCA模型: {pca_file}")
        try:
            models = joblib.load(pca_file)
            ipca = models['ipca']
            print("✅ 成功加载PCA模型")
            return ipca, None
        except Exception as e:
            print(f"加载PCA模型失败: {e}")
            return None, None
    else:
        print(f"未找到PCA模型文件: {pca_file}")
        return None, None

def apply_lda_to_features(subject_id, ipca_model):
    """对特征应用LDA"""
    print(f"\n=== 为被试 {subject_id} 应用LDA ===")
    
    # 加载原始数据
    input_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/DB4_s{subject_id}SegEnhance.h5')
    output_file = os.path.join(root_data, f'data/restimulus/Fea/DB4_s{subject_id}feaEnhance.h5')
    
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return False
    
    try:
        # 加载数据
        print("1. 加载数据...")
        with h5py.File(input_file, 'r') as f:
            emg_data = f['emg'][:]
            labels = f['label'][:]
        
        print(f"数据形状: {emg_data.shape}")
        print(f"标签数量: {len(labels)}")
        
        # 提取手势标签
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
        
        # 2. 应用PCA变换
        print("2. 应用PCA变换...")
        # 重塑数据为2D (samples, features)
        n_samples = emg_data.shape[0]
        emg_reshaped = emg_data.reshape(n_samples, -1)
        
        # 应用PCA变换
        pca_features = ipca_model.transform(emg_reshaped)
        print(f"PCA特征形状: {pca_features.shape}")
        
        # 3. 应用LDA
        print("3. 应用LDA...")
        
        # 确定训练集和测试集索引
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
        
        # 检查训练集类别
        train_gesture_labels = gesture_labels[train_indices]
        train_unique_classes = np.unique(train_gesture_labels)
        print(f"训练集类别数: {len(train_unique_classes)}")
        
        # 计算LDA维度
        n_features = pca_features.shape[1]  # PCA后的特征数
        max_lda_dim = min(n_features, len(train_unique_classes) - 1)
        lda_dim = max_lda_dim
        
        print(f"PCA特征数: {n_features}")
        print(f"LDA降维到: {lda_dim} 维")
        
        # 训练LDA
        lda = LDA(n_components=lda_dim, solver='svd', shrinkage='auto')
        lda.fit(pca_features[train_indices], train_gesture_labels)
        
        # 应用LDA变换
        lda_features = lda.transform(pca_features)
        print(f"LDA特征形状: {lda_features.shape}")
        
        # 4. 保存结果
        print("4. 保存结果...")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('features', data=lda_features)
            f.create_dataset('label', data=labels)
        
        # 保存模型
        model_file = f'pca64_lda{lda_dim}_subject{subject_id}_fixed.pkl'
        joblib.dump({
            'ipca': ipca_model,
            'lda': lda
        }, model_file)
        
        print(f"✅ 成功完成被试 {subject_id}")
        print(f"结果保存到: {output_file}")
        print(f"模型保存到: {model_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理被试 {subject_id} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== LDA应用脚本 ===")
    print("直接从已完成的PCA特征开始，避免重复计算")
    
    # 处理被试7和8
    subjects = [7, 8]
    results = {'success': [], 'failed': []}
    
    for subject_id in subjects:
        print(f"\n{'='*50}")
        print(f"处理被试 {subject_id}")
        print(f"{'='*50}")
        
        # 尝试加载已保存的PCA模型
        ipca_model, _ = load_pca_features(subject_id)
        
        if ipca_model is None:
            print(f"❌ 无法加载被试 {subject_id} 的PCA模型，跳过")
            results['failed'].append(subject_id)
            continue
        
        # 应用LDA
        success = apply_lda_to_features(subject_id, ipca_model)
        
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