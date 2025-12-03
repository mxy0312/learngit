'''
搭建ResNet网络模型训练手写数据集
'''
import random
from torch import nn
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json
import pickle

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)  # 设置随机种子

# 超参数设置
batch_size = 64
learning_rate = 0.001
num_epochs = 6

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet通常使用32×32或224×224，这里用32×32更快
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转为3通道
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels,  out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y+x)
        return y

class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
                                Residual(512, 512, use_1conv=False, strides=1))

        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, 10))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


# 初始化模型和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18(Residual).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练记录
history = {
    'train_loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

# 添加变量用于跟踪最优模型
best_accuracy = 0.0
best_model_state = None


def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算训练准确率
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 99:
            train_acc = 100. * correct / total
            print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {total_loss / 100:.3f}, acc: {train_acc:.2f}%')
            total_loss = 0
            correct = 0
            total = 0

    avg_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_loss)
    print(f'Epoch {epoch + 1} 平均损失: {avg_loss:.4f}')

    # 更新学习率
    scheduler.step()


def evaluate():
    global best_accuracy, best_model_state
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    # 计算指标
    accuracy = 100. * correct / total
    precision = 100 * precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = 100 * recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = 100 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    history['accuracy'].append(accuracy)
    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1)

    # 检查是否为最优模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # 保存当前模型状态
        best_model_state = {
            'epoch': len(history['accuracy']),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'loss': test_loss / len(test_loader)
        }
        print(f'发现新的最优模型! 准确率: {accuracy:.2f}%')

        # 立即保存最优模型到文件
        save_best_model(immediate_save=True)

    print(f'测试集准确率: {accuracy:.2f}%')
    print(f'测试集精确率: {precision:.2f}%')
    print(f'测试集召回率: {recall:.2f}%')
    print(f'测试集F1分数: {f1:.2f}%')
    print(f'测试集损失: {test_loss / len(test_loader):.4f}')

    return all_preds, all_labels


def save_best_model(immediate_save=False):
    """保存最优模型"""
    import os

    # 修改为相对路径，或者创建在当前目录下
    # 方案1: 保存在当前目录的 saved_models 文件夹
    # save_dir = 'saved_models'

    # 方案2: 如果确实需要绝对路径，可以使用用户目录
    # import os
    # save_dir = os.path.expanduser('~/saved_models')  # 用户主目录

    # 方案3: 直接保存在当前目录
    save_dir = '.'  # 当前目录

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")

    if best_model_state is not None:
        save_path = os.path.join(save_dir, 'ResNet_best.pth')

        # 尝试保存并检查是否成功
        try:
            torch.save(best_model_state, save_path)
            if immediate_save:
                print(f"已保存最优模型到 '{save_path}' (准确率: {best_accuracy:.2f}%)")
            else:
                print(f"最优模型已保存为 '{save_path}' (准确率: {best_accuracy:.2f}%)")

            # 验证文件是否成功保存
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                print(f"✓ 模型文件保存成功，大小: {os.path.getsize(save_path)} 字节")
            else:
                print("✗ 模型文件保存失败或文件为空")

        except Exception as e:
            print(f"保存模型时出错: {e}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"保存路径: {os.path.abspath(save_path)}")
    else:
        if not immediate_save:
            print("警告: 没有找到最优模型可保存")


def plot_results():
    """绘制训练结果图表"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'bo-')
    ax1.set_title('训练损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # 准确率曲线 - 修改y轴范围为90-100
    ax2.plot(epochs, history['accuracy'], 'ro-')
    ax2.set_title('测试准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([90, 100])

    # 多指标对比 - 修改y轴范围为90-100
    ax3.plot(epochs, history['accuracy'], 'ro-', label='准确率')
    ax3.plot(epochs, history['precision'], 'go-', label='精确率')
    ax3.plot(epochs, history['recall'], 'bo-', label='召回率')
    ax3.plot(epochs, history['f1'], 'mo-', label='F1分数')
    ax3.set_title('评估指标对比')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('百分比 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([90, 100])

    # 最终指标柱状图 - 修改y轴范围为90-100
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [history['accuracy'][-1], history['precision'][-1],
              history['recall'][-1], history['f1'][-1]]
    colors = ['red', 'green', 'blue', 'purple']

    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_title('最终评估指标')
    ax4.set_ylabel('百分比 (%)')
    ax4.set_ylim([90, 100])

    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('1.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_final_summary(all_preds, all_labels):
    """打印最终结果摘要"""
    print("\n" + "=" * 60)
    print("最终模型评估结果")
    print("=" * 60)
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终测试准确率: {history['accuracy'][-1]:.2f}%")
    print(f"最终精确率: {history['precision'][-1]:.2f}%")
    print(f"最终召回率: {history['recall'][-1]:.2f}%")
    print(f"最终F1分数: {history['f1'][-1]:.2f}%")

    # 打印最优模型信息
    if best_model_state is not None:
        print(f"最优模型准确率: {best_accuracy:.2f}% (Epoch {best_model_state['epoch']})")

    # 打印混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,}")
    print("=" * 60)



# 在训练完成后，保存历史数据
def save_history(history, model_name):
    """保存训练历史"""
    # 保存为JSON
    with open(f'{model_name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 保存为pickle
    with open(f'{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    print(f"{model_name}历史数据已保存")


# 主训练流程
if __name__ == '__main__':
    print(f"开始训练ResNet-18 (设备: {device})")
    print("=" * 50)

    # 打印模型结构
    print("模型结构:")
    print(model)
    print("=" * 50)

    # 训练和评估
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        train(epoch)
        all_preds, all_labels = evaluate()
        print("-" * 50)

    print("训练完成!")

    # 保存最优模型
    save_best_model()

    # 结果可视化
    plot_results()
    print_final_summary(all_preds, all_labels)
    save_history(history, 'ResNet')