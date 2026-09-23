import torch
import sys
import os
sys.path.append('.')

# 导入模型
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch

def check_module_connections():
    print("🔍 检查模块衔接和输入输出...")
    
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
    
    print(f"📊 模型结构概览:")
    print(f"   EMG通道数: {emg_channels}")
    print(f"   时间步数: {time_steps}")
    print(f"   特征维度: {feature_dim}")
    print(f"   类别数: {num_classes}")
    
    # 创建测试数据
    batch_size = 2
    feature_data = torch.randn(batch_size, feature_dim)
    emg_data = torch.randn(batch_size, emg_channels, time_steps, 1)
    
    print(f"\n🔍 测试数据形状:")
    print(f"   特征数据: {feature_data.shape}")
    print(f"   EMG数据: {emg_data.shape}")
    
    # 检查各个模块的输入输出
    print(f"\n🔍 各模块输入输出检查:")
    
    # 1. 空间分支
    print(f"\n   1. 空间分支 (ImprovedEMA):")
    try:
        space_features = model.space_branch(emg_data)
        print(f"      输入: {emg_data.shape}")
        print(f"      输出: {space_features.shape}")
        print(f"      ✅ 空间分支正常")
    except Exception as e:
        print(f"      ❌ 空间分支错误: {e}")
        return
    
    # 2. 时频分支
    print(f"\n   2. 时频分支 (LinearResidualBlock x3):")
    try:
        time_freq_features = model.time_freq_branch(feature_data)
        print(f"      输入: {feature_data.shape}")
        print(f"      输出: {time_freq_features.shape}")
        print(f"      ✅ 时频分支正常")
    except Exception as e:
        print(f"      ❌ 时频分支错误: {e}")
        return
    
    # 3. 特征融合
    print(f"\n   3. 特征融合:")
    try:
        combined_features = torch.cat([space_features, time_freq_features], dim=1)
        print(f"      空间特征: {space_features.shape}")
        print(f"      时频特征: {time_freq_features.shape}")
        print(f"      融合后: {combined_features.shape}")
        print(f"      ✅ 特征融合正常")
    except Exception as e:
        print(f"      ❌ 特征融合错误: {e}")
        return
    
    # 4. Gate层
    print(f"\n   4. Gate层:")
    try:
        gate_weight = model.gate(combined_features)
        print(f"      输入: {combined_features.shape}")
        print(f"      输出: {gate_weight.shape}")
        print(f"      ✅ Gate层正常")
    except Exception as e:
        print(f"      ❌ Gate层错误: {e}")
        return
    
    # 5. 加权融合
    print(f"\n   5. 加权融合:")
    try:
        w = gate_weight.expand_as(space_features)
        gated_space_features = space_features * w
        gated_time_freq_features = time_freq_features * (1 - w)
        final_features = gated_space_features + gated_time_freq_features
        print(f"      Gate权重: {gate_weight.shape}")
        print(f"      加权空间特征: {gated_space_features.shape}")
        print(f"      加权时频特征: {gated_time_freq_features.shape}")
        print(f"      最终特征: {final_features.shape}")
        print(f"      ✅ 加权融合正常")
    except Exception as e:
        print(f"      ❌ 加权融合错误: {e}")
        return
    
    # 6. 分类器
    print(f"\n   6. 分类器:")
    try:
        output = model.classifier(final_features)
        print(f"      输入: {final_features.shape}")
        print(f"      输出: {output.shape}")
        print(f"      ✅ 分类器正常")
    except Exception as e:
        print(f"      ❌ 分类器错误: {e}")
        return
    
    # 7. 完整前向传播
    print(f"\n   7. 完整前向传播:")
    try:
        with torch.no_grad():
            full_output, gate_weights = model(feature_data, emg_data, return_gate_weights=True)
        print(f"      完整输出: {full_output.shape}")
        print(f"      Gate权重: {gate_weights.shape}")
        print(f"      ✅ 完整前向传播正常")
    except Exception as e:
        print(f"      ❌ 完整前向传播错误: {e}")
        return
    
    print(f"\n🎉 所有模块衔接检查通过！")
    
    # 检查是否有未使用的模块
    print(f"\n🔍 检查未使用的模块:")
    unused_modules = []
    
    # GatedFusion类定义了但从未使用
    if 'GatedFusion' in globals():
        print(f"   ⚠️  GatedFusion类定义了但从未使用")
        unused_modules.append('GatedFusion')
    
    # EMA类定义了但从未使用（只用了ImprovedEMA）
    if 'EMA' in globals():
        print(f"   ⚠️  EMA类定义了但从未使用")
        unused_modules.append('EMA')
    
    if not unused_modules:
        print(f"   ✅ 没有未使用的模块")
    else:
        print(f"   📝 建议删除未使用的模块: {unused_modules}")

if __name__ == "__main__":
    check_module_connections() 