# 1、GoogLeNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 使用CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 1x1分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        # 3x3分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        # 5x5分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        # 池化分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

# GoogLeNet模型
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Inception模块：缩减通道数，避免冗余
        self.inception1 = Inception(32, 16, 16, 24, 8, 16, 16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception2 = Inception(72, 24, 24, 32, 12, 24, 24)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(104, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.inception1(x)
        x = self.maxpool2(x)
        x = self.inception2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 初始化组件
model = GoogLeNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 计算指标函数
def calculate_metrics(all_targets, all_predictions):
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

# 训练函数
def train_model(num_epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        acc, _, _, _ = calculate_metrics(all_targets, all_predictions)
        train_losses.append(epoch_loss)
        train_accuracies.append(acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
        print(f'训练集 - 准确率: {acc:.4f}\n')

    return train_losses, train_accuracies

# 测试函数
def test_model():
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * accuracy_score(all_targets, all_predictions)
    test_acc, test_prec, test_rec, test_f1 = calculate_metrics(all_targets, all_predictions)

    print("="*50)
    print("GoogLeNet模型测试结果")
    print("="*50)
    print(f'测试集准确率: {accuracy:.2f}%')
    print(f'详细指标 - 准确率: {test_acc:.4f}, 精确率: {test_prec:.4f}, 召回率: {test_rec:.4f}, F1分数: {test_f1:.4f}')
    print("\n分类报告:")
    print(classification_report(all_targets, all_predictions, digits=4))

    return test_acc, test_prec, test_rec, test_f1

# 可视化函数
def plot_training_metrics(train_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'g-', linewidth=2)
    plt.title('GoogLeNet训练损失变化', fontsize=14)
    plt.xlabel('训练轮次（Epoch）', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    print("开始训练GoogLeNet模型...")
    train_losses, train_accuracies = train_model(num_epochs=15)
    print("\n开始测试GoogLeNet模型...")
    test_model()
    plot_training_metrics(train_losses)

    torch.save(model.state_dict(), 'D:\\School\\course\\Thi-up pic\\Last\\models\\GoogleNet_model.pth')
    loss_data_save_path = 'D:\\School\\course\\Thi-up pic\\Last\\models\\GoogLeNet_loss_data.npz'
    np.savez(
        loss_data_save_path,
        train_losses=np.array(train_losses),
        train_accuracies=np.array(train_accuracies)
    )
    print(f"损失数据已保存到: {loss_data_save_path}")
    print("模型已保存！")
