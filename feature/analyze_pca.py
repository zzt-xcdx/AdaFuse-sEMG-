import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import os

def analyze_pca_variance(feature_file):
    """
    分别分析训练集和测试集的PCA降维的累计方差曲线和k95/k98值
    """
    print(f"Loading features from {feature_file}...")
    with h5py.File(feature_file, 'r') as f:
        if 'features' in f:
            features = f['features'][:]
            labels = f['label'][:]
        elif 'fea_all' in f:
            features = f['fea_all'][:]
            labels = f['label'][:]
        else:
            raise KeyError("无法找到特征数据键 (expected 'features' or 'fea_all')")
    
    print(f"Feature shape: {features.shape}")
    
    # 分离训练集和测试集
    train_indices = []
    test_indices = []
    for i in range(len(labels)):
        rep = labels[i]['repetition'] if isinstance(labels[i], np.void) else labels[i]
        if rep in train_reps:
            train_indices.append(i)
        elif rep in test_reps:
            test_indices.append(i)
    
    train_features = features[train_indices]
    test_features = features[test_indices]
    
    # 在训练集上拟合PCA
    print("Computing PCA on training set...")
    pca_train = PCA().fit(train_features)
    
    # 计算训练集的累计方差
    cum_train = np.cumsum(pca_train.explained_variance_ratio_)
    
    # 找到训练集的k95和k98
    k95_train = np.searchsorted(cum_train, 0.95) + 1
    k98_train = np.searchsorted(cum_train, 0.98) + 1
    
    print("\n训练集分析结果:")
    print(f"95% 方差信息≈{k95_train} 维")
    print(f"98% 方差信息≈{k98_train} 维")
    
    # 在测试集上计算方差解释率
    print("\nComputing variance explained on test set...")
    test_transformed = pca_train.transform(test_features)
    test_reconstructed = pca_train.inverse_transform(test_transformed)
    total_variance = np.var(test_features, axis=0).sum()
    explained_variance = np.var(test_reconstructed, axis=0).sum()
    variance_ratio_test = explained_variance / total_variance
    
    print(f"测试集上的总方差解释率: {variance_ratio_test:.4f}")
    
    # 绘制训练集和测试集的累计方差曲线
    plt.figure(figsize=(12, 6))
    
    # 训练集曲线
    plt.plot(cum_train, label='训练集累计方差', color='blue')
    plt.axhline(0.95, color='r', linestyle='--', label='95% 方差')
    plt.axhline(0.98, color='g', linestyle='--', label='98% 方差')
    
    # 在测试集上计算不同维度的方差解释率
    test_variance_ratios = []
    for n_components in range(1, len(pca_train.components_) + 1):
        pca_n = PCA(n_components=n_components)
        test_transformed_n = pca_train.transform(test_features)[:, :n_components]
        test_reconstructed_n = pca_train.inverse_transform(test_transformed_n)
        explained_variance_n = np.var(test_reconstructed_n, axis=0).sum()
        test_variance_ratios.append(explained_variance_n / total_variance)
    
    # 测试集曲线
    plt.plot(test_variance_ratios, label='测试集累计方差', color='red', linestyle='--')
    
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差')
    plt.title('PCA累计方差曲线 (训练集 vs 测试集)')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pca_variance_curve_train_test.png'))
    plt.close()
    
    # 根据训练集k95值给出建议
    if k95_train <= 70:
        recommended_dim = 64
    elif k95_train <= 90:
        recommended_dim = 96
    else:
        recommended_dim = 128
    
    print(f"\n建议的PCA维度: {recommended_dim}")
    print(f"理由: 训练集k95={k95_train}，根据规则选择最接近的2的幂次方")
    print(f"注意: 此建议仅基于训练集数据，避免数据泄露")
    
    return k95_train, k98_train, recommended_dim

def main():
    # 设置数据目录
    root_data = 'D:/DB2'
    
    # 分析所有被试的数据
    all_k95 = []
    all_k98 = []
    all_recommended = []
    
    for subject_id in range(1, 41):
        feature_file = os.path.join(root_data, f'data/restimulus/Fea/DB2_s{subject_id}feaEnhance.h5')
        
        if not os.path.exists(feature_file):
            print(f"跳过被试 {subject_id}: 文件不存在")
            continue
            
        print(f"\n分析被试 {subject_id}...")
        try:
            k95, k98, recommended = analyze_pca_variance(feature_file)
            all_k95.append(k95)
            all_k98.append(k98)
            all_recommended.append(recommended)
        except Exception as e:
            print(f"处理被试 {subject_id} 时出错: {e}")
    
    # 计算统计信息
    if all_k95:
        print("\n所有被试的统计信息:")
        print(f"k95 平均值: {np.mean(all_k95):.1f} ± {np.std(all_k95):.1f}")
        print(f"k98 平均值: {np.mean(all_k98):.1f} ± {np.std(all_k98):.1f}")
        print(f"建议维度分布:")
        print(f"64维: {all_recommended.count(64)}个被试")
        print(f"96维: {all_recommended.count(96)}个被试")
        print(f"128维: {all_recommended.count(128)}个被试")

if __name__ == '__main__':
    main() 