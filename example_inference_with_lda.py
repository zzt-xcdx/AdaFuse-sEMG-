import torch
import numpy as np
import joblib
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch

def preprocess_with_pca_lda(raw_stockwell_4800, subject_id=7):
    """
    使用PCA+LDA变换器预处理原始Stockwell特征
    
    Args:
        raw_stockwell_4800: 原始Stockwell特征，形状为(4800,)
        subject_id: 被试ID，用于加载对应的变换器
    
    Returns:
        预处理后的特征张量，形状为(1, 52)
    """
    # 加载PCA+LDA变换器
    trans = joblib.load(f'pca64_lda52_subject{subject_id}.pkl')
    
    # 应用PCA变换
    x = trans['ipca'].transform(raw_stockwell_4800.reshape(1, -1))
    
    # 应用LDA变换
    x = trans['lda'].transform(x)
    
    # 转换为PyTorch张量
    return torch.tensor(x, dtype=torch.float32)  # shape (1, 52)

def inference_example():
    """
    推理示例：展示如何使用PCA+LDA预处理和模型推理
    """
    # 假设我们有一个原始Stockwell特征
    # 这里用随机数据作为示例
    raw_stockwell_4800 = np.random.randn(4800).astype(np.float32)
    
    # 预处理特征
    processed_features = preprocess_with_pca_lda(raw_stockwell_4800, subject_id=7)
    print(f"预处理后特征形状: {processed_features.shape}")
    
    # 假设我们有一个EMG信号片段 (1, channels, time_steps, 1)
    emg_signal = torch.randn(1, 8, 300, 1)  # 示例：1个样本，8个通道，300个时间步
    print(f"EMG信号形状: {emg_signal.shape}")
    
    # 加载训练好的模型
    # 注意：这里需要根据实际情况调整参数
    model = ImprovedFeaAndEmg_se_PyTorch(
        emg_channels=8,
        time_steps=300,
        feature_dim=52,  # 现在是52维而不是64维
        num_classes=53,  # 根据实际类别数调整
        dropout_rate=0.25
    )
    
    # 设置为评估模式
    model.eval()
    
    # 进行推理
    with torch.no_grad():
        output, gate_weights = model(processed_features, emg_signal, return_gate_weights=True)
        
        # 获取预测结果
        predicted_class = torch.argmax(output, dim=1)
        
        print(f"模型输出形状: {output.shape}")
        print(f"预测类别: {predicted_class.item()}")
        print(f"门控权重: {gate_weights}")

if __name__ == "__main__":
    print("PCA+LDA推理示例")
    print("=" * 50)
    
    try:
        inference_example()
        print("\n推理示例执行成功！")
    except Exception as e:
        print(f"\n推理示例执行失败: {e}")
        print("请确保：")
        print("1. 已运行GetFeature.py生成PCA+LDA变换器文件")
        print("2. 模型文件存在且参数正确")
        print("3. 输入数据格式正确") 