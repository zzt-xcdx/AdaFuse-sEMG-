import torch
import numpy as np
from PyTorchModels import FeaAndEmg_se_PyTorch

# ========== 配置 ==========
model_path = 'results/best_model_subject_1.pth'  # ← 改成你的路径
num_classes = 17  # DB2 是 17 类（0-16）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 加载模型 ==========
print("正在加载模型...")
model = FeaAndEmg_se_PyTorch(num_classes=num_classes)

try:
    checkpoint = torch.load(model_path, map_location=device)

    # 检查是否是完整的 checkpoint（包含 epoch/optimizer）还是纯权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载成功！来自 Epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 加载成功！")

    model.to(device)
    model.eval()

except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

# ========== 测试模型结构 ==========
print("\n模型结构检查:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
print(f"  可训练参数: {trainable_params:,}")

# ========== 测试前向传播 ==========
print("\n前向传播测试:")
dummy_input = torch.randn(4, 260, 12).to(device)  # (batch, time, channels)

try:
    with torch.no_grad():
        output = model(dummy_input)

        # 检查输出格式
        if isinstance(output, tuple):
            logits, gate_weights = output
            print(f"  ✅ 输出形状: {logits.shape} (预期: [4, {num_classes}])")
            print(f"  ✅ 门控权重: {[f'{w.item() * 100:.1f}%' for w in gate_weights]}")
        else:
            logits = output
            print(f"  ✅ 输出形状: {logits.shape} (预期: [4, {num_classes}])")
            print(f"  ⚠️ 未返回门控权重")

        # 检查输出范围（应该是 logits，不是概率）
        print(f"  输出范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        # 测试预测
        predictions = torch.argmax(logits, dim=1)
        print(f"  预测类别: {predictions.cpu().numpy()}")

except Exception as e:
    print(f"  ❌ 前向传播失败: {e}")
    exit()

print("\n✅ 模型可用！可以进行后续实验。")

