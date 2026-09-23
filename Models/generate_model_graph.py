import torch
from PyTorchModels_improved import ImprovedFeaAndEmg_se_PyTorch

# 使用DB2的参数
model = ImprovedFeaAndEmg_se_PyTorch(
    emg_channels=12,
    time_steps=500,
    feature_dim=64,  # ← 确认这个值！
    num_classes=49,  # DB2
    dropout_rate=0.25
)

def count_params(module):
    return sum(p.numel() for p in module.parameters())

# 详细统计
space_params = count_params(model.space_branch)
time_params = count_params(model.time_domain_branch)
freq_params = count_params(model.time_freq_branch)
gate_params = count_params(model.gate)
classifier_params = count_params(model.classifier)
total_params = count_params(model)

print(f"{'模块':<30} {'参数量':<15} {'百分比'}")
print("-" * 60)
print(f"{'CEINet (空间分支)':<30} {space_params:<15,} {space_params/total_params*100:>6.1f}%")
print(f"{'DTANet (时域分支)':<30} {time_params:<15,} {time_params/total_params*100:>6.1f}%")
print(f"{'IFFNet (频域分支)':<30} {freq_params:<15,} {freq_params/total_params*100:>6.1f}%")
print(f"{'FLAG (融合模块)':<30} {gate_params:<15,} {gate_params/total_params*100:>6.1f}%")
print(f"{'分类器':<30} {classifier_params:<15,} {classifier_params/total_params*100:>6.1f}%")
print("-" * 60)
print(f"{'总计':<30} {total_params:<15,} {'100.0%':>6}")

# 验证是否等于358,768
if total_params == 358768:
    print("\n✅ 参数量匹配！")
else:
    print(f"\n⚠️ 参数量不匹配！期望358,768，实际{total_params:,}")