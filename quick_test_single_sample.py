#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证：单条样本过拟合测试
验证数据、标签和模型是否正常工作
"""

import torch
import torch.nn.functional as F
import numpy as np
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch
import h5py
import os
import sys

# 添加路径并动态导入
sys.path.append('Models/TrainModel')
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", "Models/TrainModel/1EmgFeaTrain_mobileone.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
EMGDataset = train_module.EMGDataset

# 添加模型路径
sys.path.append('Models')

def test_model_with_different_feature_dims():
    """测试不同特征维度下模型的行为"""
    
    try:
        from PyTorchModels_improved import ImprovedFeaAndEmg_se_PyTorch
        print("✓ 成功导入模型")
    except Exception as e:
        print(f"✗ 导入模型失败: {e}")
        return
    
    # 测试参数
    emg_channels = 8
    time_steps = 100
    num_classes = 17
    dropout_rate = 0.4
    
    # 测试1: 特征维度为64（正常情况）
    print("\n=== 测试1: 特征维度为64 ===")
    try:
        model = ImprovedFeaAndEmg_se_PyTorch(
            emg_channels=emg_channels,
            time_steps=time_steps,
            feature_dim=64,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            disable_time_freq=False
        )
        
        # 测试前向传播
        batch_size = 2
        test_feature = torch.randn(batch_size, 64)
        test_emg = torch.randn(batch_size, time_steps, emg_channels, 1)
        
        with torch.no_grad():
            output, gate_weights = model(test_feature, test_emg, return_gate_weights=True)
        
        print(f"✓ 特征维度64测试成功")
        print(f"  输出维度: {output.shape}")
        print(f"  门控权重维度: {gate_weights.shape}")
        
    except Exception as e:
        print(f"✗ 特征维度64测试失败: {e}")
    
    # 测试2: 特征维度为4800（异常情况，应该禁用时频分支）
    print("\n=== 测试2: 特征维度为4800 ===")
    try:
        model = ImprovedFeaAndEmg_se_PyTorch(
            emg_channels=emg_channels,
            time_steps=time_steps,
            feature_dim=4800,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            disable_time_freq=True  # 强制禁用时频分支
        )
        
        # 测试前向传播
        batch_size = 2
        test_feature = torch.randn(batch_size, 4800)
        test_emg = torch.randn(batch_size, time_steps, emg_channels, 1)
        
        with torch.no_grad():
            output, gate_weights = model(test_feature, test_emg, return_gate_weights=True)
        
        print(f"✓ 特征维度4800测试成功（时频分支已禁用）")
        print(f"  输出维度: {output.shape}")
        print(f"  门控权重维度: {gate_weights.shape}")
        
    except Exception as e:
        print(f"✗ 特征维度4800测试失败: {e}")
    
    # 测试3: 自动检测特征维度并禁用时频分支
    print("\n=== 测试3: 自动检测特征维度 ===")
    try:
        feature_dim = 4800  # 模拟读取到的特征维度
        disable_time_freq = feature_dim != 64
        
        if disable_time_freq:
            print(f"  检测到特征维度为{feature_dim}，自动禁用时频分支")
        
        model = ImprovedFeaAndEmg_se_PyTorch(
            emg_channels=emg_channels,
            time_steps=time_steps,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            disable_time_freq=disable_time_freq
        )
        
        # 测试前向传播
        batch_size = 2
        test_feature = torch.randn(batch_size, feature_dim)
        test_emg = torch.randn(batch_size, time_steps, emg_channels, 1)
        
        with torch.no_grad():
            output, gate_weights = model(test_feature, test_emg, return_gate_weights=True)
        
        print(f"✓ 自动检测测试成功")
        print(f"  输出维度: {output.shape}")
        print(f"  门控权重维度: {gate_weights.shape}")
        
    except Exception as e:
        print(f"✗ 自动检测测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始测试修复后的模型...")
    test_model_with_different_feature_dims()
    print("\n测试完成!") 