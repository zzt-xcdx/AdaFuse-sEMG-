#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LDA修复的简单脚本
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def test_lda_constraint():
    """测试LDA的约束条件"""
    print("=== 测试LDA约束条件 ===")
    
    # 模拟数据
    n_samples = 1000
    n_features = 64  # PCA后的特征数
    n_classes = 47   # 手势类别数
    
    # 生成模拟数据
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(1, n_classes + 1, n_samples)
    
    print(f"样本数: {n_samples}")
    print(f"特征数: {n_features}")
    print(f"类别数: {n_classes}")
    print(f"唯一标签: {np.unique(y)}")
    
    # 计算LDA约束
    max_lda_components = min(n_features, n_classes - 1)
    print(f"LDA最大可能维度: {max_lda_components}")
    
    # 测试不同的LDA维度
    test_dims = [max_lda_components, max_lda_components - 1, max_lda_components + 1]
    
    for dim in test_dims:
        try:
            print(f"\n测试LDA维度: {dim}")
            lda = LDA(n_components=dim, solver='lsqr', shrinkage='auto')
            lda.fit(X, y)
            print(f"✅ 成功: LDA维度 {dim}")
        except Exception as e:
            print(f"❌ 失败: LDA维度 {dim} - {e}")

def test_lda_with_subset():
    """测试使用训练集子集的情况"""
    print("\n=== 测试训练集子集LDA ===")
    
    # 模拟数据
    n_samples = 1000
    n_features = 64
    n_classes = 47
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(1, n_classes + 1, n_samples)
    
    # 模拟训练集和测试集分割
    train_indices = np.arange(0, n_samples, 2)  # 偶数索引作为训练集
    test_indices = np.arange(1, n_samples, 2)   # 奇数索引作为测试集
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    
    print(f"总样本数: {n_samples}")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")
    print(f"训练集类别数: {len(np.unique(y_train))}")
    print(f"训练集类别: {np.unique(y_train)}")
    
    # 计算基于训练集的LDA维度
    train_n_classes = len(np.unique(y_train))
    max_lda_dim = min(n_features, train_n_classes - 1)
    
    print(f"基于训练集的LDA最大维度: {max_lda_dim}")
    
    try:
        lda = LDA(n_components=max_lda_dim, solver='lsqr', shrinkage='auto')
        lda.fit(X_train, y_train)
        print(f"✅ 成功: 训练集LDA维度 {max_lda_dim}")
        
        # 测试转换
        X_transformed = lda.transform(X)
        print(f"转换后数据形状: {X_transformed.shape}")
        
    except Exception as e:
        print(f"❌ 失败: {e}")

if __name__ == "__main__":
    test_lda_constraint()
    test_lda_with_subset() 