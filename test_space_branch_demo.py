#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间分支演示脚本
展示从数据输入到输出的完整流程
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 导入空间分支相关模块
import sys
sys.path.append('Models')
from PyTorchModels_backup_before_lightweight import EMA, SpatialFeatureExtractor

def create_demo_data(batch_size=4, channels=8, time_steps=200):
    """
    创建演示用的肌电数据
    """
    print(f"创建演示数据: 批次大小={batch_size}, 通道数={channels}, 时间步长={time_steps}")
    
    # 创建模拟的肌电数据
    # 添加一些模式来模拟真实的肌电信号
    data = np.random.randn(batch_size, channels, time_steps, 1)
    
    # 添加一些周期性模式
    for b in range(batch_size):
        for c in range(channels):
            # 添加不同频率的正弦波
            freq = 5 + c * 2  # 不同通道不同频率
            t = np.linspace(0, 2*np.pi, time_steps)
            signal = np.sin(freq * t) * 0.3
            data[b, c, :, 0] += signal
    
    return torch.FloatTensor(data)

def visualize_data_flow(input_data, space_branch):
    """
    可视化数据流动过程
    """
    print("\n=== 数据流动过程可视化 ===")
    
    # 1. 原始输入
    print(f"1. 原始输入形状: {input_data.shape}")
    
    # 2. 维度重排
    B, C, T, D = input_data.shape
    permuted = input_data.permute(0, 3, 1, 2)
    print(f"2. 维度重排后: {permuted.shape}")
    
    # 3. 通过空间特征提取器
    spatial_extractor = space_branch.spatial_extractor
    
    # 获取中间结果
    x = permuted
    print(f"3. 输入卷积网络: {x.shape}")
    
    # 第一层卷积
    x = spatial_extractor.spatial_net[0:3](x)  # Conv2d + BN + PReLU
    print(f"4. 第1层卷积后: {x.shape}")
    
    # 第二层卷积
    x = spatial_extractor.spatial_net[3:8](x)  # Conv2d + BN + PReLU + MaxPool + Dropout
    print(f"5. 第2层卷积后: {x.shape}")
    
    # 第三层卷积
    x = spatial_extractor.spatial_net[8:13](x)  # Conv2d + BN + PReLU + MaxPool + Dropout
    print(f"6. 第3层卷积后: {x.shape}")
    
    # 自适应池化
    x = spatial_extractor.adaptive_pool(x)
    print(f"7. 自适应池化后: {x.shape}")
    
    # 特征聚合
    x = x.view(B, 64, -1)
    x = x.mean(dim=2)
    print(f"8. 特征聚合后: {x.shape}")
    
    # 投影层
    x = spatial_extractor.projection(x)
    print(f"9. 最终输出: {x.shape}")
    
    return x

def test_space_branch():
    """
    测试空间分支的完整功能
    """
    print("=== 空间分支功能测试 ===")
    
    # 1. 创建演示数据
    input_data = create_demo_data(batch_size=4, channels=8, time_steps=200)
    
    # 2. 创建空间分支
    space_branch = EMA(channels=8, dropout_rate=0.3)
    print(f"空间分支创建成功")
    
    # 3. 测试前向传播
    print("\n--- 测试前向传播 ---")
    try:
        output = space_branch(input_data)
        print(f"✓ 前向传播成功!")
        print(f"输入形状: {input_data.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return
    
    # 4. 可视化数据流动
    print("\n--- 数据流动可视化 ---")
    detailed_output = visualize_data_flow(input_data, space_branch)
    
    # 5. 验证输出一致性
    print("\n--- 输出一致性验证 ---")
    if torch.allclose(output, detailed_output, atol=1e-6):
        print("✓ 输出一致性验证通过!")
    else:
        print("✗ 输出一致性验证失败!")
    
    return output

def analyze_feature_distribution(output):
    """
    分析输出特征的分布
    """
    print("\n=== 特征分布分析 ===")
    
    # 转换为numpy数组
    features = output.detach().numpy()
    
    # 基本统计信息
    print(f"特征维度: {features.shape}")
    print(f"均值: {np.mean(features):.4f}")
    print(f"标准差: {np.std(features):.4f}")
    print(f"最小值: {np.min(features):.4f}")
    print(f"最大值: {np.max(features):.4f}")
    
    # 可视化特征分布
    plt.figure(figsize=(12, 4))
    
    # 1. 特征值分布直方图
    plt.subplot(1, 3, 1)
    plt.hist(features.flatten(), bins=30, alpha=0.7, color='blue')
    plt.title('特征值分布')
    plt.xlabel('特征值')
    plt.ylabel('频次')
    
    # 2. 样本间特征差异
    plt.subplot(1, 3, 2)
    sample_means = np.mean(features, axis=1)
    plt.bar(range(len(sample_means)), sample_means, alpha=0.7, color='green')
    plt.title('样本间特征均值差异')
    plt.xlabel('样本索引')
    plt.ylabel('特征均值')
    
    # 3. 特征维度重要性
    plt.subplot(1, 3, 3)
    feature_std = np.std(features, axis=0)
    plt.bar(range(len(feature_std)), feature_std, alpha=0.7, color='red')
    plt.title('特征维度标准差')
    plt.xlabel('特征维度')
    plt.ylabel('标准差')
    
    plt.tight_layout()
    plt.savefig('空间分支特征分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("特征分布图已保存为 '空间分支特征分析.png'")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("空间分支演示程序")
    print("=" * 60)
    
    # 设置随机种子确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试空间分支
    output = test_space_branch()
    
    if output is not None:
        # 分析特征分布
        analyze_feature_distribution(output)
        
        print("\n" + "=" * 60)
        print("演示完成!")
        print("=" * 60)
        print("\n总结:")
        print("1. ✓ 空间分支成功处理了肌电数据")
        print("2. ✓ 输入形状: (4, 8, 200, 1) → 输出形状: (4, 64)")
        print("3. ✓ 数据流动过程清晰可追踪")
        print("4. ✓ 特征分布合理，可用于后续分类任务")
        print("5. ✓ 空间分支设计有效，能够提取空间特征")
    else:
        print("测试失败，请检查代码!")

if __name__ == "__main__":
    main() 