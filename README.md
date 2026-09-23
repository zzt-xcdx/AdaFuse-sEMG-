# AdaFuse

**Ada**ptive multi-branch **fuse** for sEMG gesture recognition — 基于多通道自适应融合网络的肌电信号手势识别（Ninapro DB2/DB4/DB5）。

毕设方向：基于多通道特征融合网络的肌电信号手势识别方法研究。

## 项目结构（主要代码）

```
Adafuse/
├── GetRaw.py
├── Denoise/EMGFilter.py               # EMG 去噪滤波
├── Preprocess/DFactionSeg.py          # 分段、标准化、降采样等
├── feature/GetFeature.py              # Stockwell 时频特征 + PCA/LDA
├── Models/
│   ├── PyTorchModels_improved.py      # 多分支 + 门控融合模型
│   └── trainandtest.py                # 主训练入口
├── Util/                              # 数据划分、绘图、工具函数
└──
```

实验脚本、消融版本、预处理备份等见仓库内其他目录；**权重、训练曲线、混淆矩阵等结果默认不上传 Git**（见 `.gitignore`）。

## 快速开始

1. 准备 Ninapro 数据（本地路径在训练脚本中配置，如 `D:/DB4`）。
2. 预处理：`GetRaw` → `EMGFilter` → `Preprocess/DFactionSeg` → `feature/GetFeature.py` → `trainandtest.py `即可。
3. 训练：`python Models/TrainModel/trainandtest.py`（按脚本内 参数 修改）。

## 环境

- Python 3.8+
- PyTorch、NumPy、h5py、scikit-learn、matplotlib、scipy  
- Stockwell：`stockwell` 或项目内 ST 实现（以 `GetFeature.py` 为准）

## 推送到 Git

远程仓库：[AdaFuse-sEMG-](https://github.com/zzt-xcdx/AdaFuse-sEMG-)

```bash
git remote set-url origin https://github.com/zzt-xcdx/AdaFuse-sEMG-.git
git add .
git status   # 确认未包含 .pth / results / .h5 等（见 .gitignore）
git commit -m "Initial commit: AdaFuse core pipeline"
git push -u origin main
```

## 说明

- 数据集协议遵循 [Ninapro](https://ninapro.hevs.ch/) 公开基准。
- 其他方法备份（能量核、VNet、SHAP 等）为历史实验，非主链路必需。
- 主要文件就上面提到那些，其他的都不是那么重要。

## 如果对您的研究有帮助，请动动您发财的小手，帮我点一个免费的star! 我会非常感谢
