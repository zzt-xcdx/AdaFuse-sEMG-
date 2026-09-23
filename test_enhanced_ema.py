import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append('.')

# 导入增强的EMA模块和模型
from Models.PyTorchModels_improved_with_ablation import SpatialFeatureExtractor, EMA

def test_spatial_feature_extractor():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建随机测试数据
    batch_size = 16
    emg_channels = 12
    time_steps = 400
    feature_dim = 64
    
    print("\n测试空间特征提取模块...")
    
    # 创建非零的随机测试数据
    emg_data = torch.randn(batch_size, emg_channels, time_steps, 1) * 0.1  # 添加缩放因子
    print(f"EMG数据统计: 均值={emg_data.mean().item():.4f}, 标准差={emg_data.std().item():.4f}")
    
    # 测试空间特征提取器
    print("\n测试SpatialFeatureExtractor...")
    spatial_extractor = SpatialFeatureExtractor(in_channels=emg_channels, out_channels=feature_dim, dropout_rate=0.3)
    
    # 初始化权重
    for m in spatial_extractor.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    spatial_extractor = spatial_extractor.to(device)
    emg_data_device = emg_data.to(device)
    
    # 前向传播
    with torch.no_grad():
        spatial_output = spatial_extractor(emg_data_device)
    
    # 打印输出形状和统计信息
    print(f"输入EMG形状: {emg_data.shape}")
    print(f"空间特征提取器输出形状: {spatial_output.shape}")
    print(f"空间特征统计: 均值={spatial_output.mean().item():.4f}, 标准差={spatial_output.std().item():.4f}")
    
    # 检查输出维度是否正确
    assert spatial_output.shape == (batch_size, feature_dim), f"输出形状应为 {(batch_size, feature_dim)}, 但得到 {spatial_output.shape}"
    
    # 检查输出是否全为0
    assert not torch.allclose(spatial_output, torch.zeros_like(spatial_output)), "输出全为0，可能存在初始化问题"
    
    print("\n测试成功! 空间特征提取器工作正常。")
    
    # 测试EMA模块
    print("\n测试EMA模块...")
    enhanced_ema = EMA(channels=emg_channels, N=1, factor=2, dropout_rate=0.3)
    
    # 初始化权重
    for m in enhanced_ema.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    enhanced_ema = enhanced_ema.to(device)
    
    # 前向传播
    with torch.no_grad():
        enhanced_output = enhanced_ema(emg_data_device)
    
    # 打印输出形状和统计信息
    print(f"EMA输出形状: {enhanced_output.shape}")
    print(f"EMA输出统计: 均值={enhanced_output.mean().item():.4f}, 标准差={enhanced_output.std().item():.4f}")
    
    # 检查输出维度是否正确
    assert enhanced_output.shape == (batch_size, feature_dim), f"输出形状应为 {(batch_size, feature_dim)}, 但得到 {enhanced_output.shape}"
    
    # 检查输出是否全为0
    assert not torch.allclose(enhanced_output, torch.zeros_like(enhanced_output)), "输出全为0，可能存在初始化问题"
    
    print("\n测试成功! EMA模块工作正常。")
    
    # 测试EMA模块的空间特征
    print("\n分析EMA模块的空间特征...")
    
    # 获取空间特征
    spatial_features = spatial_extractor(emg_data_device)
    
    # 打印特征统计信息
    print(f"空间特征统计: 均值={spatial_features.mean().item():.4f}, 标准差={spatial_features.std().item():.4f}")
    
    # 检查特征是否有效
    assert not torch.allclose(spatial_features, torch.zeros_like(spatial_features)), "空间特征全为0，可能存在问题"
    
    # 测试不同输入规模
    print("\n测试不同输入规模...")
    test_sizes = [(8, 12, 200, 1), (32, 12, 400, 1), (16, 8, 300, 1)]
    
    for size in test_sizes:
        test_data = torch.randn(size) * 0.1
        test_data = test_data.to(device)
        
        try:
            with torch.no_grad():
                output = enhanced_ema(test_data)
            print(f"输入形状 {size} 测试成功，输出形状: {output.shape}")
        except Exception as e:
            print(f"输入形状 {size} 测试失败: {str(e)}")
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    test_spatial_feature_extractor() 