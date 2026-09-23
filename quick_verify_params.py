import torch
import sys
import os
sys.path.append('.')

# 导入模型
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch

def verify_model_params():
    print("🔍 验证模型参数量...")
    
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
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 模型参数统计:")
    print(f"   总参数量: {total_params:,} ({total_params/1e6:.3f} M)")
    print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.3f} M)")
    
    # 检查是否达到目标
    target_params = 0.22e6  # 0.22M
    tolerance = 0.05e6      # ±5%
    
    if abs(total_params - target_params) <= tolerance:
        print(f"✅ 参数量恢复成功！目标: {target_params/1e6:.2f}M ± {tolerance/1e6:.2f}M")
    else:
        print(f"❌ 参数量未达标！目标: {target_params/1e6:.2f}M ± {tolerance/1e6:.2f}M")
    
    # 详细参数分布
    print(f"\n🔍 各模块参数量:")
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 1000:  # 只显示参数量较大的模块
                print(f"   {name}: {param_count:,} params")
    
    return total_params

if __name__ == "__main__":
    verify_model_params() 