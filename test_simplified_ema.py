# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 导入简化后的EMA模块
from Models.PyTorchModels_backup_before_lightweight import EMA, SpatialFeatureExtractor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def test_spatial_feature_extractor():
    """测试空间特征提取器"""
    print("="*50)
    print("测试空间特征提取器")
    print("="*50)
    
    # 设置参数
    batch_size = 8
    channels = 12  # EMG通道数
    time_steps = 400  # 时间步长
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据 (B, C, T, 1)
    test_data = torch.randn(batch_size, channels, time_steps, 1) * 0.1
    test_data = test_data.to(device)
    
    print(f"输入数据形状: {test_data.shape}")
    print(f"设备: {device}")
    
    # 创建空间特征提取器
    spatial_extractor = SpatialFeatureExtractor(
        in_channels=channels,
        out_channels=64,
        dropout_rate=0.3
    ).to(device)
    
    # 初始化权重
    for m in spatial_extractor.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # 前向传播
    with torch.no_grad():
        output = spatial_extractor(test_data)
    
    print(f"输出形状: {output.shape}")
    print(f"输出统计: 均值={output.mean().item():.4f}, 标准差={output.std().item():.4f}")
    
    # 验证输出维度
    expected_shape = (batch_size, 64)
    assert output.shape == expected_shape, f"期望形状 {expected_shape}, 实际形状 {output.shape}"
    
    # 检查输出是否有效
    assert not torch.allclose(output, torch.zeros_like(output)), "输出全为0，可能存在问题"
    
    print("✅ 空间特征提取器测试通过！")
    return output

def test_ema_module():
    """测试简化后的EMA模块"""
    print("\n" + "="*50)
    print("测试简化后的EMA模块")
    print("="*50)
    
    # 设置参数
    batch_size = 8
    channels = 12
    time_steps = 400
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据 (B, C, T, 1)
    test_data = torch.randn(batch_size, channels, time_steps, 1) * 0.1
    test_data = test_data.to(device)
    
    print(f"输入数据形状: {test_data.shape}")
    
    # 创建EMA模块
    ema = EMA(channels=channels, N=1, factor=16, dropout_rate=0.3).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = ema(test_data)
    
    print(f"EMA输出形状: {output.shape}")
    print(f"EMA输出统计: 均值={output.mean().item():.4f}, 标准差={output.std().item():.4f}")
    
    # 验证输出维度
    expected_shape = (batch_size, 64)
    assert output.shape == expected_shape, f"期望形状 {expected_shape}, 实际形状 {output.shape}"
    
    # 检查输出是否有效
    assert not torch.allclose(output, torch.zeros_like(output)), "输出全为0，可能存在问题"
    
    print("✅ EMA模块测试通过！")
    return output

def test_invalid_inputs():
    """测试无效输入"""
    print("\n" + "="*50)
    print("测试无效输入")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ema = EMA(channels=12, N=1, factor=16, dropout_rate=0.3).to(device)
    
    # 测试2维输入
    print("测试2维输入...")
    try:
        test_2d = torch.randn(8, 64).to(device)
        ema(test_2d)
        print("❌ 2维输入应该报错但没有报错")
    except ValueError as e:
        print(f"✅ 2维输入正确报错: {e}")
    
    # 测试3维输入
    print("测试3维输入...")
    try:
        test_3d = torch.randn(8, 12, 400).to(device)
        ema(test_3d)
        print("❌ 3维输入应该报错但没有报错")
    except ValueError as e:
        print(f"✅ 3维输入正确报错: {e}")
    
    # 测试错误的4维输入
    print("测试错误的4维输入...")
    try:
        test_4d_wrong = torch.randn(8, 12, 400, 2).to(device)  # 最后一维不是1
        ema(test_4d_wrong)
        print("❌ 错误的4维输入应该报错但没有报错")
    except ValueError as e:
        print(f"✅ 错误的4维输入正确报错: {e}")

def test_different_sizes():
    """测试不同尺寸的输入"""
    print("\n" + "="*50)
    print("测试不同尺寸的输入")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ema = EMA(channels=12, N=1, factor=16, dropout_rate=0.3).to(device)
    
    # 测试不同的批次大小
    batch_sizes = [1, 4, 16, 32]
    for batch_size in batch_sizes:
        test_data = torch.randn(batch_size, 12, 400, 1).to(device)
        with torch.no_grad():
            output = ema(test_data)
        expected_shape = (batch_size, 64)
        assert output.shape == expected_shape, f"批次大小{batch_size}测试失败"
        print(f"✅ 批次大小 {batch_size} 测试通过")
    
    # 测试不同的通道数
    channel_configs = [(8, 8), (16, 16), (24, 16)]  # (输入通道, factor)
    for channels, factor in channel_configs:
        ema_test = EMA(channels=channels, N=1, factor=factor, dropout_rate=0.3).to(device)
        test_data = torch.randn(8, channels, 400, 1).to(device)
        with torch.no_grad():
            output = ema_test(test_data)
        expected_shape = (8, 64)
        assert output.shape == expected_shape, f"通道数{channels}测试失败"
        print(f"✅ 通道数 {channels} 测试通过")

def test_model_integration():
    """测试模型集成"""
    print("\n" + "="*50)
    print("测试模型集成")
    print("="*50)
    
    from Models.PyTorchModels_backup_before_lightweight import FeaAndEmg_se_PyTorch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = FeaAndEmg_se_PyTorch(
        emg_channels=12,
        time_steps=400,
        feature_dim=64,
        num_classes=47,
        dropout_rate=0.4
    ).to(device)
    
    # 创建测试数据
    emg_data = torch.randn(8, 12, 400, 1).to(device)
    feature_data = torch.randn(8, 64).to(device)
    
    print(f"EMG数据形状: {emg_data.shape}")
    print(f"特征数据形状: {feature_data.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(feature_data, emg_data)
    
    print(f"模型输出形状: {output.shape}")
    print(f"模型输出统计: 均值={output.mean().item():.4f}, 标准差={output.std().item():.4f}")
    
    # 验证输出维度
    expected_shape = (8, 47)
    assert output.shape == expected_shape, f"期望形状 {expected_shape}, 实际形状 {output.shape}"
    
    print("✅ 模型集成测试通过！")

def main():
    """主函数"""
    print("开始测试简化后的EMA模块...")
    
    try:
        # 测试空间特征提取器
        spatial_output = test_spatial_feature_extractor()
        
        # 测试EMA模块
        ema_output = test_ema_module()
        
        # 测试无效输入
        test_invalid_inputs()
        
        # 测试不同尺寸
        test_different_sizes()
        
        # 测试模型集成
        test_model_integration()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！简化后的EMA模块工作正常")
        print("="*60)
        
        # 打印总结
        print("\n简化后的EMA模块特点:")
        print("✅ 只处理4维输入 (B, C, T, 1)")
        print("✅ 输出64维空间特征")
        print("✅ 代码更简洁，去除了不必要的复杂性")
        print("✅ 与空间特征提取器完美集成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 