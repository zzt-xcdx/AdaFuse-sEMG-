import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加路径
sys.path.append('.')

from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch, SpatialFeatureExtractor, EMA

def test_spatial_feature_extractor():
    """测试修复后的空间特征提取器"""
    print("="*50)
    print("测试修复后的空间特征提取器")
    print("="*50)
    
    # 测试不同的输入通道数
    test_cases = [
        (12, 400, "DB4标准配置"),
        (16, 1000, "DB2标准配置"),
        (8, 200, "较少通道配置")
    ]
    
    for in_channels, time_steps, description in test_cases:
        print(f"\n测试配置: {description}")
        print(f"输入通道数: {in_channels}, 时间步数: {time_steps}")
        
        # 创建模型
        extractor = SpatialFeatureExtractor(
            in_channels=in_channels,
            out_channels=64,
            dropout_rate=0.3
        )
        
        # 创建测试数据
        batch_size = 4
        test_input = torch.randn(batch_size, in_channels, time_steps, 1)
        print(f"输入数据形状: {test_input.shape}")
        
        # 前向传播
        extractor.eval()
        with torch.no_grad():
            output = extractor(test_input)
        
        print(f"输出特征形状: {output.shape}")
        print(f"输出特征范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # 验证输出维度
        expected_shape = (batch_size, 64)
        assert output.shape == expected_shape, f"期望形状 {expected_shape}, 实际形状 {output.shape}"
        print("✅ 维度测试通过")
        
        # 检查第一层卷积的输入通道数
        first_conv = extractor.spatial_net[0]
        actual_in_channels = first_conv.in_channels
        print(f"第一层卷积输入通道数: {actual_in_channels}")
        assert actual_in_channels == in_channels, f"期望 {in_channels}, 实际 {actual_in_channels}"
        print("✅ 输入通道数测试通过")

def test_ema_module():
    """测试修复后的EMA模块"""
    print("\n" + "="*50)
    print("测试修复后的EMA模块")
    print("="*50)
    
    # 测试配置
    channels = 12
    time_steps = 400
    batch_size = 4
    
    # 创建EMA模块
    ema = EMA(channels=channels, N=time_steps, dropout_rate=0.3)
    
    # 创建测试数据
    test_input = torch.randn(batch_size, channels, time_steps, 1)
    print(f"EMA输入数据形状: {test_input.shape}")
    
    # 前向传播
    ema.eval()
    with torch.no_grad():
        output = ema(test_input)
    
    print(f"EMA输出特征形状: {output.shape}")
    print(f"EMA输出特征范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # 验证输出维度
    expected_shape = (batch_size, 64)
    assert output.shape == expected_shape, f"期望形状 {expected_shape}, 实际形状 {output.shape}"
    print("✅ EMA维度测试通过")

def test_full_model():
    """测试完整的修复后模型"""
    print("\n" + "="*50)
    print("测试完整的修复后模型")
    print("="*50)
    
    # 模型参数
    emg_channels = 12
    time_steps = 400
    feature_dim = 128
    num_classes = 10
    batch_size = 4
    
    # 创建模型
    model = ImprovedFeaAndEmg_se_PyTorch(
        emg_channels=emg_channels,
        time_steps=time_steps,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout_rate=0.25
    )
    
    # 创建测试数据
    emg_data = torch.randn(batch_size, emg_channels, time_steps, 1)
    feature_data = torch.randn(batch_size, feature_dim)
    
    print(f"EMG数据形状: {emg_data.shape}")
    print(f"特征数据形状: {feature_data.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output, gate_weights = model(feature_data, emg_data, return_gate_weights=True)
    
    print(f"模型输出形状: {output.shape}")
    print(f"门控权重形状: {gate_weights.shape}")
    print(f"门控权重: 空间分支={gate_weights[0, 0].item():.3f}, 时频分支={gate_weights[0, 1].item():.3f}")
    
    # 验证输出维度
    expected_output_shape = (batch_size, num_classes)
    expected_gate_shape = (batch_size, 2)
    assert output.shape == expected_output_shape, f"期望输出形状 {expected_output_shape}, 实际 {output.shape}"
    assert gate_weights.shape == expected_gate_shape, f"期望门控权重形状 {expected_gate_shape}, 实际 {gate_weights.shape}"
    print("✅ 完整模型测试通过")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

def main():
    """主测试函数"""
    print("开始测试修复后的模型...")
    
    try:
        # 测试空间特征提取器
        test_spatial_feature_extractor()
        
        # 测试EMA模块
        test_ema_module()
        
        # 测试完整模型
        test_full_model()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！模型修复成功！")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 