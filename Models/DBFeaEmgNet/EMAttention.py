import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

# 注意：SpatialFeatureExtractor 和 EMA 类已迁移到 Models/PyTorchModels_improved_with_ablation.py
# 如果需要使用这些类，请从 Models.PyTorchModels 导入 