import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Models.PyTorchModels_improved_with_ablation import FeaAndEmg_se_PyTorch
import matplotlib.pyplot as plt

# 设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 测试BN层顺序的模型
class TestBNOrderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, bn_order='traditional'):
        super(TestBNOrderModel, self).__init__()
        self.bn_order = bn_order
        
        if bn_order == 'traditional':
            # 传统顺序：Conv -> BN -> ReLU
            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        else:
            # 实验顺序：Conv -> ReLU -> BN
            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.classifier(x)

def test_bn_order_effect():
    """测试BN层顺序对训练效果的影响"""
    
    # 生成模拟数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟EMG特征数据
    batch_size = 64
    feature_dim = 128
    num_classes = 10
    
    # 生成训练数据
    X_train = torch.randn(batch_size * 10, feature_dim)
    y_train = torch.randint(0, num_classes, (batch_size * 10,))
    
    # 生成测试数据
    X_test = torch.randn(batch_size * 2, feature_dim)
    y_test = torch.randint(0, num_classes, (batch_size * 2,))
    
    # 测试两种BN顺序
    orders = ['traditional', 'experimental']
    results = {}
    
    for order in orders:
        print(f"\n测试 {order} 顺序...")
        
        # 创建模型
        model = TestBNOrderModel(feature_dim, 256, num_classes, bn_order=order)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        train_losses = []
        test_accuracies = []
        
        for epoch in range(50):
            # 训练
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # 测试
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test)
                    _, predicted = torch.max(test_outputs.data, 1)
                    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
                    test_accuracies.append(accuracy)
                    print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Test Acc = {accuracy:.4f}")
        
        results[order] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
    
    # 绘制比较结果
    plt.figure(figsize=(12, 5))
    
    # 训练损失比较
    plt.subplot(1, 2, 1)
    for order in orders:
        plt.plot(results[order]['train_losses'], label=f'{order} order')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 测试准确率比较
    plt.subplot(1, 2, 2)
    epochs = list(range(0, 50, 10))
    for order in orders:
        plt.plot(epochs, results[order]['test_accuracies'], 
                marker='o', label=f'{order} order')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bn_order_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印最终结果
    print("\n最终结果比较:")
    for order in orders:
        final_acc = results[order]['test_accuracies'][-1]
        final_loss = results[order]['train_losses'][-1]
        print(f"{order} 顺序: 最终准确率 = {final_acc:.4f}, 最终损失 = {final_loss:.4f}")

if __name__ == "__main__":
    test_bn_order_effect() 