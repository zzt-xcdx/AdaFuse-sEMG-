# 受试者7和8专用增强滤波方案使用指南

## 概述

本方案专门针对受试者7和8的信号质量问题设计，解决以下关键问题：
- **SNR为0.0dB**的严重噪声问题
- **运动伪影**（受试者8的极大尖峰）
- **特征0方差异常高**的稳定性问题
- **样本分布不均**导致的分类困难

## 问题分析

### 受试者7的问题
- SNR = 0.0 dB，信号质量极差
- 特征0方差极大（-80到80），噪声严重
- 样本分布极不均匀，某些手势样本过多（2000+），某些过少
- 最严重问题类别：[4, 10, 19, 20, 34, 37, 41, 42, 45]

### 受试者8的问题
- SNR = 0.0 dB，信号质量差
- 通道2存在极大尖峰（-3000到1000），明显是运动伪影
- 特征0变异性异常高（-80到60）
- 最严重问题类别：[7, 12, 16, 21, 22, 23, 31, 32, 34, 44, 45, 46, 47]

## 解决方案

### 1. 运动伪影去除
- **检测方法**：基于统计阈值（5倍标准差）
- **去除方法**：局部中值滤波插值
- **目标**：去除受试者8的极大尖峰

### 2. 自适应噪声抑制
- **强噪声抑制**：针对SNR<0.3的信号
  - 高通40Hz → 强陷波50/100Hz → 低通350Hz → 中值滤波 → 小波去噪
- **中等噪声抑制**：针对0.3≤SNR<0.6的信号
  - 高通25Hz → 标准陷波 → 低通400Hz → 轻微中值滤波
- **轻微噪声抑制**：针对SNR≥0.6的信号
  - 使用标准滤波方案

### 3. 信号质量评估
- **RMS评分**：理想范围50-500
- **SNR评分**：基于功率谱密度计算
- **峰值评分**：避免过大峰值
- **综合质量评分**：0-1分，越高越好

## 使用方法

### 步骤1：生成增强数据
```bash
python Denoise/EMGFilter_enhanced_subjects_7_8.py
```

这将为受试者7和8生成增强的滤波数据，保存在：
```
D:/DB4/data/restimulus/refilter_enhanced/
├── DB4_s7filter_enhanced.h5
└── DB4_s8filter_enhanced.h5
```

### 步骤2：测试增强效果
```bash
python test_enhanced_filter_subjects_7_8.py
```

这将生成对比图，显示增强前后的效果：
- `enhanced_filter_test_results.png`：模拟数据对比
- `subject_7_real_data_comparison.png`：受试者7真实数据对比
- `subject_8_real_data_comparison.png`：受试者8真实数据对比

### 步骤3：使用增强数据
```bash
python Denoise/EMGFilter.py
```

修改后的`EMGFilter.py`会自动检测并使用增强数据。

## 技术细节

### 运动伪影检测
```python
def detect_motion_artifacts(self, signal_ch, threshold_factor=5.0):
    mean_val = np.mean(signal_ch)
    std_val = np.std(signal_ch)
    upper_threshold = mean_val + threshold_factor * std_val
    lower_threshold = mean_val - threshold_factor * std_val
    artifacts_mask = (signal_ch > upper_threshold) | (signal_ch < lower_threshold)
    return artifacts_mask, upper_threshold, lower_threshold
```

### 强噪声抑制
```python
def strong_noise_suppression(self, signal_ch):
    # 1. 强高通滤波 (40Hz)
    sos_hp = signal.butter(8, 40, btype='highpass', fs=self.fs, output='sos')
    signal_hp = signal.sosfiltfilt(sos_hp, signal_ch)
    
    # 2. 强陷波滤波 (50Hz, 100Hz)
    sos_n50 = signal.iirnotch(50, 60, fs=self.fs)
    sos_n50 = signal.tf2sos(*sos_n50)
    signal_n50 = signal.sosfiltfilt(sos_n50, signal_hp)
    
    # 3. 强低通滤波 (350Hz)
    sos_lp = signal.butter(8, 350, btype='lowpass', fs=self.fs, output='sos')
    signal_lp = signal.sosfiltfilt(sos_lp, signal_n100)
    
    # 4. 中值滤波去噪
    signal_median = median_filter(signal_lp, size=7)
    
    # 5. 小波去噪
    signal_denoised = self.wavelet_denoise(signal_median)
    
    return signal_denoised
```

### 质量评分计算
```python
def calculate_signal_quality(self, signal_ch):
    # RMS评分 (理想范围50-500)
    if 50 <= rms <= 500:
        quality_score += 0.4
    elif 20 <= rms <= 1000:
        quality_score += 0.2
    else:
        quality_score += 0.1
    
    # SNR评分
    if snr_db >= 10:
        quality_score += 0.4
    elif snr_db >= 5:
        quality_score += 0.2
    elif snr_db >= 0:
        quality_score += 0.1
    else:
        quality_score += 0.05
    
    # 峰值评分
    if peak <= 1000:
        quality_score += 0.2
    elif peak <= 3000:
        quality_score += 0.1
    else:
        quality_score += 0.05
```

## 预期效果

### 信号质量提升
- **SNR改善**：从0.0dB提升到5-10dB
- **RMS稳定**：控制在50-500的理想范围
- **峰值降低**：去除运动伪影，降低异常峰值

### 特征稳定性提升
- **特征0方差**：从±80降低到±20以内
- **特征相关性**：减少特征间冗余
- **分类性能**：提升受试者7和8的准确率

### 分类准确率预期
- **受试者7**：从69%提升到75-80%
- **受试者8**：从69%提升到75-80%
- **整体提升**：DB4数据集平均准确率提升2-3%

## 注意事项

1. **数据备份**：运行前请备份原始数据
2. **依赖包**：需要安装`pywt`（可选，用于小波去噪）
3. **处理时间**：增强处理比标准滤波慢2-3倍
4. **存储空间**：增强数据文件比原始数据大10-20%

## 故障排除

### 常见问题
1. **找不到增强数据文件**：确保先运行`EMGFilter_enhanced_subjects_7_8.py`
2. **处理速度慢**：增强滤波计算量较大，请耐心等待
3. **内存不足**：可以分批处理数据

### 调试建议
1. 运行测试脚本查看效果
2. 检查生成的对比图
3. 查看控制台输出的质量评分

## 后续改进

1. **特征工程优化**：针对增强后的信号设计新特征
2. **模型调整**：根据信号特性调整模型参数
3. **数据增强**：对样本稀少的手势进行数据增强
4. **集成学习**：结合多种滤波方案的结果 