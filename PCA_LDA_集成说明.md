# PCA→LDA 集成说明文档

## 概述

本项目已成功集成PCA→LDA降维流程，将原来的64维PCA特征进一步降维到52维，提高特征的可判别性和模型性能。

## 主要改动

### 1. GetFeature.py 文件修改

#### 1.1 新增导入
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib
```

#### 1.2 PCA后追加LDA处理
在 `calculate_pca_from_batches()` 函数末尾添加：
```python
# ---- 3.1 追加 LDA ↓ ----
n_classes = len(np.unique(labels))
lda_dim   = n_classes - 1               # =52（53 类时）

print(f"Fitting LDA  ({lda_dim} dims) on training data...")
lda = LDA(n_components=lda_dim, solver='lsqr', shrinkage='auto')
lda.fit(pca_results[train_indices], labels[train_indices])

# 对所有样本做 LDA 变换
emg_lda = lda.transform(pca_results).astype(np.float32)

# 保存变换器，推理阶段要一起用
joblib.dump(
    {'ipca': ipca, 'lda': lda},
    f'pca64_lda{lda_dim}_subject{subject_id}.pkl'
)
# --------------------------

# 清理 PCA 原数组，返回 LDA 结果
del pca_results
return emg_lda
```

#### 1.3 特征选择适配
修改 `apply_feature_selection()` 函数，从64维改为52维：
```python
# 创建52维的零矩阵
selected_features_52d = np.zeros((features.shape[0], 52), dtype=features.dtype)
```

### 2. PyTorchModels_improved.py 无需修改

该文件已经使用 `feature_dim` 动态读取特征维度，会自动适应52维输入。

### 3. 训练脚本修改

#### 3.1 模型导入

```python
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch
```

#### 3.2 模型实例化
```python
model = ImprovedFeaAndEmg_se_PyTorch(emg_channels, time_steps, feature_dim, num_classes, dropout_rate=dropout_rate).to(device)
```

## 使用方法

### 1. 特征提取阶段

运行修改后的 `GetFeature.py`：
```bash
cd feature
python GetFeature.py
```

这将生成：
- H5特征文件：包含52维LDA特征
- PKL变换器文件：`pca64_lda52_subject{subject_id}.pkl`

### 2. 训练阶段

直接运行修改后的训练脚本：
```bash
cd Models/TrainModel
python 1EmgFeaTrain_backup.py
```

训练脚本会自动：
- 读取52维特征数据
- 动态设置模型的特征维度
- 正常进行训练

### 3. 推理阶段

#### 3.1 离线特征（推荐）
如果特征已经写入H5文件，直接读取即可：
```python
with h5py.File('feature_file.h5', 'r') as f:
    features = f['features'][:]  # shape: (N, 52)
```

#### 3.2 在线预处理
如果需要实时处理原始Stockwell特征：
```python
import joblib

# 加载变换器
trans = joblib.load('pca64_lda52_subject7.pkl')

def preprocess(raw_stockwell_4800):
    x = trans['ipca'].transform(raw_stockwell_4800.reshape(1,-1))
    x = trans['lda'].transform(x)
    return torch.tensor(x, dtype=torch.float32)    # shape (1, 52)
```

## 文件结构

```
Adafuse/
├── feature/
│   └── GetFeature.py                    # 已修改：PCA→LDA
├── Models/
│   ├── PyTorchModels_improved.py        # 无需修改
│   └── TrainModel/
│       └── 1EmgFeaTrain_backup.py      # 已修改：使用改进模型
├── pca64_lda52_subject7.pkl             # 生成的变换器文件
├── pca64_lda52_subject8.pkl             # 生成的变换器文件
├── example_inference_with_lda.py        # 推理示例脚本
└── PCA_LDA_集成说明.md                   # 本文档
```

## 维度变化

- **原始Stockwell特征**: 4800维
- **PCA降维后**: 64维
- **LDA降维后**: 52维 (53类手势时)
- **最终特征**: 52维，写入H5文件

## 注意事项

### 1. 类别数变化
如果手势类别数改变，LDA维度会自动调整：
- 53类 → 52维
- 40类 → 39维
- 等等

### 2. 变换器文件管理
每个被试都有独立的变换器文件，确保：
- 训练和推理使用相同的变换器
- 变换器文件路径正确
- 变换器文件完整

### 3. 内存管理
LDA处理会增加一些内存使用，但通过批处理和垃圾回收已优化。

## 故障排除

### 1. 维度不匹配错误
检查：
- 特征文件是否为52维
- 模型是否正确读取 `feature_dim`
- 变换器文件是否完整

### 2. 变换器加载失败
检查：
- 文件路径是否正确
- 是否已运行特征提取
- 文件是否损坏

### 3. 训练失败
检查：
- 数据加载是否正常
- 模型参数是否正确
- GPU内存是否充足

## 性能提升

通过PCA→LDA集成，预期获得：
- 更好的特征判别性
- 减少过拟合风险
- 提高模型泛化能力
- 保持计算效率

## 总结

PCA→LDA集成已完成，主要特点：
1. **自动化流程**：训练时自动生成变换器
2. **向后兼容**：现有模型架构无需修改
3. **灵活配置**：自动适应不同类别数
4. **完整文档**：提供详细使用说明

如有问题，请参考本文档或查看相关代码注释。 