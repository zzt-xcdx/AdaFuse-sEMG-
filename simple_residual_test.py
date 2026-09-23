# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SimpleLinearBlock(nn.Module):
    """简单的线性块，没有残差连接"""
    def __init__(self, in_features, out_features, dropout_rate):
        super(SimpleLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class LinearResidualBlock(nn.Module):
    """带残差连接的线性块"""
    def __init__(self, in_features, out_features, dropout_rate):
        super(LinearResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        identity = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out += self.shortcut(identity)
        return out

class ResidualModel(nn.Module):
    """带残差连接的模型"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(ResidualModel, self).__init__()
        self.features = nn.Sequential(
            LinearResidualBlock(input_dim, 64, dropout_rate),
            LinearResidualBlock(64, 128, dropout_rate),
            LinearResidualBlock(128, 64, dropout_rate)
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleModel(nn.Module):
    """简单线性模型，没有残差连接"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            SimpleLinearBlock(input_dim, 64, dropout_rate),
            SimpleLinearBlock(64, 128, dropout_rate),
            SimpleLinearBlock(128, 64, dropout_rate)
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def generate_synthetic_data(n_samples=1000, input_dim=64, num_classes=47):
    """生成合成数据进行测试"""
    print("生成合成数据...")
    
    # 生成随机特征
    X = np.random.randn(n_samples, input_dim)
    
    # 生成标签（模拟分类任务）
    y = np.random.randint(0, num_classes, n_samples)
    
    # 分割数据
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"数据生成完成:")
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"验证集: {X_val.shape}, 标签: {y_val.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(model, X_train, y_train, X_val, y_val, num_epochs=30, lr=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"开始训练 {model.__class__.__name__}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        train_acc = accuracy_score(y_train, predicted.cpu().numpy()) * 100
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = accuracy_score(y_val, val_predicted.cpu().numpy()) * 100
        
        # 记录历史
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.2f}%')
    
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    print(f"最终验证准确率: {val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_val_acc': val_acc,
        'training_time': training_time
    }

def plot_comparison(residual_result, simple_result):
    """绘制对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 训练损失对比
    axes[0, 0].plot(residual_result['train_losses'], label='残差模型 (训练)', color='blue')
    axes[0, 0].plot(residual_result['val_losses'], label='残差模型 (验证)', color='blue', linestyle='--')
    axes[0, 0].plot(simple_result['train_losses'], label='简单模型 (训练)', color='orange')
    axes[0, 0].plot(simple_result['val_losses'], label='简单模型 (验证)', color='orange', linestyle='--')
    
    axes[0, 0].set_title('训练和验证损失对比')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练准确率对比
    axes[0, 1].plot(residual_result['train_accs'], label='残差模型 (训练)', color='blue')
    axes[0, 1].plot(residual_result['val_accs'], label='残差模型 (验证)', color='blue', linestyle='--')
    axes[0, 1].plot(simple_result['train_accs'], label='简单模型 (训练)', color='orange')
    axes[0, 1].plot(simple_result['val_accs'], label='简单模型 (验证)', color='orange', linestyle='--')
    
    axes[0, 1].set_title('训练和验证准确率对比')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 最终性能对比
    model_names = ['残差模型', '简单模型']
    final_val_accs = [residual_result['final_val_acc'], simple_result['final_val_acc']]
    training_times = [residual_result['training_time'], simple_result['training_time']]
    
    axes[1, 0].bar(model_names, final_val_accs, color=['blue', 'orange'])
    axes[1, 0].set_title('最终验证准确率对比')
    axes[1, 0].set_ylabel('准确率 (%)')
    
    for i, v in enumerate(final_val_accs):
        axes[1, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center')
    
    axes[1, 1].bar(model_names, training_times, color=['blue', 'orange'])
    axes[1, 1].set_title('训练时间对比')
    axes[1, 1].set_ylabel('时间 (秒)')
    
    for i, v in enumerate(training_times):
        axes[1, 1].text(i, v + 0.1, f'{v:.1f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig('simple_residual_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("开始残差连接对比实验（简化版）...")
    
    # 生成合成数据
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_data()
    
    # 获取数据维度
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"输入维度: {input_dim}, 类别数: {num_classes}")
    
    # 创建模型
    residual_model = ResidualModel(input_dim, num_classes)
    simple_model = SimpleModel(input_dim, num_classes)
    
    print(f"\n残差模型参数数量: {sum(p.numel() for p in residual_model.parameters())}")
    print(f"简单模型参数数量: {sum(p.numel() for p in simple_model.parameters())}")
    
    # 训练残差模型
    print(f"\n{'='*50}")
    print("训练残差模型")
    print(f"{'='*50}")
    residual_result = train_model(residual_model, X_train, y_train, X_val, y_val)
    
    # 训练简单模型
    print(f"\n{'='*50}")
    print("训练简单模型")
    print(f"{'='*50}")
    simple_result = train_model(simple_model, X_train, y_train, X_val, y_val)
    
    # 测试模型
    print(f"\n{'='*50}")
    print("测试模型性能")
    print(f"{'='*50}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # 测试残差模型
    residual_model.eval()
    with torch.no_grad():
        residual_outputs = residual_model(X_test_tensor)
        _, residual_predicted = torch.max(residual_outputs.data, 1)
        residual_test_acc = accuracy_score(y_test, residual_predicted.cpu().numpy()) * 100
    
    # 测试简单模型
    simple_model.eval()
    with torch.no_grad():
        simple_outputs = simple_model(X_test_tensor)
        _, simple_predicted = torch.max(simple_outputs.data, 1)
        simple_test_acc = accuracy_score(y_test, simple_predicted.cpu().numpy()) * 100
    
    print(f"残差模型测试准确率: {residual_test_acc:.2f}%")
    print(f"简单模型测试准确率: {simple_test_acc:.2f}%")
    
    # 绘制对比结果
    plot_comparison(residual_result, simple_result)
    
    # 打印总结
    print(f"\n{'='*60}")
    print("对比实验总结")
    print(f"{'='*60}")
    print(f"残差模型:")
    print(f"  验证准确率: {residual_result['final_val_acc']:.2f}%")
    print(f"  测试准确率: {residual_test_acc:.2f}%")
    print(f"  训练时间: {residual_result['training_time']:.2f}秒")
    print()
    print(f"简单模型:")
    print(f"  验证准确率: {simple_result['final_val_acc']:.2f}%")
    print(f"  测试准确率: {simple_test_acc:.2f}%")
    print(f"  训练时间: {simple_result['training_time']:.2f}秒")
    print()
    
    # 计算改进
    acc_improvement = residual_test_acc - simple_test_acc
    time_difference = residual_result['training_time'] - simple_result['training_time']
    
    print(f"残差连接带来的改进:")
    print(f"  准确率提升: {acc_improvement:.2f}%")
    print(f"  时间差异: {time_difference:.2f}秒")
    
    if acc_improvement > 0:
        print(f"  ✅ 残差连接提高了性能！")
    else:
        print(f"  ❌ 残差连接没有提高性能")
    
    print("\n对比实验完成！")

if __name__ == "__main__":
    main() 