import torch
import sys
import os
sys.path.append('.')

# 导入模型
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch

def simple_param_check():
    print("🔍 简单参数检查...")
    
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
    
    # 直接计算每个模块的参数
    space_params = sum(p.numel() for p in model.space_branch.parameters())
    time_freq_params = sum(p.numel() for p in model.time_freq_branch.parameters())
    gate_params = sum(p.numel() for p in model.gate.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    total_calculated = space_params + time_freq_params + gate_params + classifier_params
    total_official = sum(p.numel() for p in model.parameters())
    
    print(f"📊 各模块参数量:")
    print(f"   空间分支: {space_params:,} ({space_params/1e6:.3f}M)")
    print(f"   时频分支: {time_freq_params:,} ({time_freq_params/1e6:.3f}M)")
    print(f"   Gate: {gate_params:,} ({gate_params/1e6:.3f}M)")
    print(f"   分类器: {classifier_params:,} ({classifier_params/1e6:.3f}M)")
    print(f"   计算总和: {total_calculated:,} ({total_calculated/1e6:.3f}M)")
    print(f"   官方总和: {total_official:,} ({total_official/1e6:.3f}M)")
    
    # 检查分类器每一层
    print(f"\n🔍 分类器每层参数:")
    for i, layer in enumerate(model.classifier):
        if hasattr(layer, 'weight'):
            params = sum(p.numel() for p in layer.parameters())
            print(f"   层{i}: {type(layer).__name__} - {params:,} params")
    
    return total_official

if __name__ == "__main__":
    simple_param_check() 