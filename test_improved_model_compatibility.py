import torch
import torch.nn as nn
import numpy as np
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch

def test_improved_model():
    """测试改进后的模型是否能正常工作"""
    
    print("测试改进后的模型...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 模拟数据参数
    batch_size = 4
    emg_channels = 12
    time_steps = 400
    feature_dim = 128
    num_classes = 10
    
    # 创建模拟数据
    emg_data = torch.randn(batch_size, emg_channels, time_steps, 1)  # (B, C, T, 1)
    feature_data = torch.randn(batch_size, feature_dim)  # (B, F)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"EMG数据形状: {emg_data.shape}")
    print(f"特征数据形状: {feature_data.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 创建改进的模型
    model = ImprovedFeaAndEmg_se_PyTorch(
        emg_channels=emg_channels,
        time_steps=time_steps,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout_rate=0.4
    )
    
    # 打印模型结构
    print(f"\n模型结构:")
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 前向传播测试
    print(f"\n前向传播测试...")
    model.eval()
    with torch.no_grad():
        # 测试不带门控权重
        output = model(feature_data, emg_data)
        print(f"输出形状 (不带门控权重): {output.shape}")
        
        # 测试带门控权重
        output, gate_weights = model(feature_data, emg_data, return_gate_weights=True)
        print(f"输出形状 (带门控权重): {output.shape}")
        print(f"门控权重形状: {gate_weights.shape}")
        print(f"门控权重值: {gate_weights}")
    
    # 测试训练模式
    print(f"\n训练模式测试...")
    model.train()
    output, gate_weights = model(feature_data, emg_data, return_gate_weights=True)
    print(f"训练模式输出形状: {output.shape}")
    
    # 测试损失计算
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    print(f"损失值: {loss.item():.4f}")
    
    # 测试反向传播
    print(f"\n反向传播测试...")
    loss.backward()
    print("反向传播成功!")
    
    # 检查梯度
    grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"梯度范数: {grad_norm:.4f}")
    
    print(f"\n✅ 改进后的模型测试通过!")
    
    # 测试不同批次大小
    print(f"\n测试不同批次大小...")
    batch_sizes = [1, 2, 8, 16]
    for bs in batch_sizes:
        emg_data_bs = torch.randn(bs, emg_channels, time_steps, 1)
        feature_data_bs = torch.randn(bs, feature_dim)
        labels_bs = torch.randint(0, num_classes, (bs,))
        
        with torch.no_grad():
            output_bs, gate_weights_bs = model(feature_data_bs, emg_data_bs, return_gate_weights=True)
            print(f"批次大小 {bs}: 输出形状 {output_bs.shape}, 门控权重形状 {gate_weights_bs.shape}")

if __name__ == "__main__":
    test_improved_model() 