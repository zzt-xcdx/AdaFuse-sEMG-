import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def check_pca_data():
    """验证PCA分析的数据是否正确"""
    
    # 设置数据路径
    root_data = 'D:/DB4'
    subject_id = 5  # 检查被试5的数据
    
    # 1. 检查原始EMG数据
    print("="*50)
    print("1. 检查原始EMG数据")
    print("="*50)
    
    emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/DB4_s{subject_id}SegEnhance.h5')
    feature_file = os.path.join(root_data, f'data/restimulus/Fea/DB4_s{subject_id}feaEnhance.h5')
    
    with h5py.File(emg_file, 'r') as f:
        emg_data = f['emg'][:]
        labels = f['label'][:]
        print(f"EMG数据形状: {emg_data.shape}")
        print(f"标签数量: {len(labels)}")
        print(f"EMG数据范围: [{emg_data.min():.3f}, {emg_data.max():.3f}]")
        print(f"EMG数据均值: {emg_data.mean():.3f}")
        print(f"EMG数据标准差: {emg_data.std():.3f}")
        
        # 检查前几个样本
        print(f"\n前3个样本的EMG数据:")
        for i in range(min(3, emg_data.shape[0])):
            print(f"样本{i}: 形状={emg_data[i].shape}, 范围=[{emg_data[i].min():.3f}, {emg_data[i].max():.3f}]")
    
    # 2. 检查PCA特征数据
    print("\n" + "="*50)
    print("2. 检查PCA特征数据")
    print("="*50)
    
    with h5py.File(feature_file, 'r') as f:
        features = f['features'][:]
        feature_labels = f['label'][:]
        print(f"PCA特征形状: {features.shape}")
        print(f"特征标签数量: {len(feature_labels)}")
        print(f"PCA特征范围: [{features.min():.3f}, {features.max():.3f}]")
        print(f"PCA特征均值: {features.mean():.3f}")
        print(f"PCA特征标准差: {features.std():.3f}")
        
        # 检查前几个样本
        print(f"\n前3个样本的PCA特征:")
        for i in range(min(3, features.shape[0])):
            print(f"样本{i}: 形状={features[i].shape}, 前5个值={features[i][:5]}")
    
    # 3. 验证数据对应关系
    print("\n" + "="*50)
    print("3. 验证数据对应关系")
    print("="*50)
    
    print(f"EMG样本数: {emg_data.shape[0]}")
    print(f"PCA特征样本数: {features.shape[0]}")
    print(f"EMG标签数: {len(labels)}")
    print(f"PCA特征标签数: {len(feature_labels)}")
    
    if emg_data.shape[0] == features.shape[0]:
        print("✅ 样本数量匹配")
    else:
        print("❌ 样本数量不匹配")
    
    if len(labels) == len(feature_labels):
        print("✅ 标签数量匹配")
    else:
        print("❌ 标签数量不匹配")
    
    # 4. 检查标签对应关系
    print("\n" + "="*50)
    print("4. 检查标签对应关系")
    print("="*50)
    
    # 检查前10个标签
    print("前10个EMG标签:")
    for i in range(min(10, len(labels))):
        if isinstance(labels[i], np.void):
            print(f"  样本{i}: {dict(labels[i])}")
        else:
            print(f"  样本{i}: {labels[i]}")
    
    print("\n前10个PCA特征标签:")
    for i in range(min(10, len(feature_labels))):
        if isinstance(feature_labels[i], np.void):
            print(f"  样本{i}: {dict(feature_labels[i])}")
        else:
            print(f"  样本{i}: {feature_labels[i]}")
    
    # 5. 可视化验证
    print("\n" + "="*50)
    print("5. 可视化验证")
    print("="*50)
    
    # 绘制PCA特征的前几个主成分
    plt.figure(figsize=(15, 5))
    
    # 前4个主成分的分布
    for i in range(min(4, features.shape[1])):
        plt.subplot(1, 4, i+1)
        plt.hist(features[:, i], bins=50, alpha=0.7)
        plt.title(f'主成分 {i+1} 分布')
        plt.xlabel('特征值')
        plt.ylabel('频次')
    
    plt.tight_layout()
    plt.savefig('pca_components_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制前两个主成分的散点图
    plt.figure(figsize=(10, 8))
    
    # 根据手势类别着色
    unique_gestures = np.unique([label['gesture'] if isinstance(label, np.void) else label for label in feature_labels])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_gestures)))
    
    for i, gesture in enumerate(unique_gestures):
        mask = [label['gesture'] == gesture if isinstance(label, np.void) else label == gesture for label in feature_labels]
        plt.scatter(features[mask, 0], features[mask, 1], 
                   c=[colors[i]], label=f'手势{gesture}', alpha=0.6, s=20)
    
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.title('PCA特征的前两个主成分分布')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存为 'pca_components_distribution.png' 和 'pca_scatter_plot.png'")

if __name__ == '__main__':
    check_pca_data() 