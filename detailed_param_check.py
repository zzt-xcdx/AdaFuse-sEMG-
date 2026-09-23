import torch
import sys
import os
sys.path.append('.')

# 导入模型
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch

def detailed_param_check():
    print("🔍 详细检查模型参数量...")
    
    # 模拟参数
    emg_channels = 12
    time_steps = 400
    feature_dim = 64
    num_classes = 47  # DB4的手势数量
    dropout_rate = 0.25
    
    # 创建模型
    model = ImprovedFeaAndEmg_se_PyTorch(
        emg_channels=emg_channels,
        time_steps=time_steps,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    print(f"📊 模型结构:")
    print(model)
    
    # 使用PyTorch官方方法计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 官方方法计算的参数量:")
    print(f"   总参数量: {total_params:,} ({total_params/1e6:.3f} M)")
    print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.3f} M)")
    
    # 检查分类器
    print(f"\n🔍 分类器详细参数:")
    classifier_params = 0
    for i, layer in enumerate(model.classifier):
        if hasattr(layer, 'weight'):
            param_count = sum(p.numel() for p in layer.parameters())
            print(f"   分类器[{i}] {type(layer).__name__}: {param_count:,} params")
            classifier_params += param_count
    
    print(f"   分类器总参数: {classifier_params:,}")
    
    # 检查Gate
    gate_params = sum(p.numel() for p in model.gate.parameters())
    print(f"   Gate参数: {gate_params:,}")
    
    # 检查时频分支
    time_freq_params = sum(p.numel() for p in model.time_freq_branch.parameters())
    print(f"   时频分支参数: {time_freq_params:,}")
    
    # 检查空间分支
    space_params = sum(p.numel() for p in model.space_branch.parameters())
    print(f"   空间分支参数: {space_params:,}")
    
    # 验证计算
    calculated_total = classifier_params + gate_params + time_freq_params + space_params
    print(f"\n🔍 验证计算:")
    print(f"   分类器: {classifier_params:,}")
    print(f"   Gate: {gate_params:,}")
    print(f"   时频分支: {time_freq_params:,}")
    print(f"   空间分支: {space_params:,}")
    print(f"   计算总和: {calculated_total:,} ({calculated_total/1e6:.3f} M)")
    print(f"   官方总和: {total_params:,} ({total_params/1e6:.3f} M)")
    
    return total_params

if __name__ == "__main__":
    detailed_param_check() 