'''
搭建GoogLeNet网络模型训练手写数据集
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
num_epochs = 10

# 定义数据转换，将图像调整为 3×224×224
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整尺寸到 224×224
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转为3通道
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()

        # 1x1卷积分支
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 -> 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 3x3池化 -> 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 在通道维度上拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=False):  # 设置为False
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 3a
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        # Inception 3b
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 4a
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        # Inception 4b
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        # Inception 4c
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        # Inception 4d
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        # Inception 4e
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 5a
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        # Inception 5b
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)



        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # 初始卷积层
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        # Inception 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 最终分类
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)


        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 初始化模型和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GoogLeNet(num_classes=10).to(device)  # MNIST有10个类别
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练记录
history = {
    'train_loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}


def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        # 处理GoogLeNet的多输出
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            # 训练时返回主输出和辅助输出
            main_output, aux1_output, aux2_output = outputs
            loss = criterion(main_output, target) + 0.3 * criterion(aux1_output, target) + 0.3 * criterion(aux2_output, target)
        else:
            # 测试时只返回主输出
            loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {total_loss / 100:.3f}')
            total_loss = 0

    avg_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_loss)
    print(f'Epoch {epoch + 1} 平均损失: {avg_loss:.4f}')


def evaluate():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    precision = 100 * precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = 100 * recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = 100 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    history['accuracy'].append(accuracy)
    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1)

    print(f'测试集准确率: {accuracy:.2f}%')
    print(f'测试集精确率: {precision:.2f}%')
    print(f'测试集召回率: {recall:.2f}%')
    print(f'测试集F1分数: {f1:.2f}%')

    return all_preds, all_labels


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
    ax2.set_ylim([90, 100])  # 设置y轴范围为90-100

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
    ax3.set_ylim([90, 100])  # 设置y轴范围为90-100

    # 最终指标柱状图 - 修改y轴范围为90-100
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [history['accuracy'][-1], history['precision'][-1],
              history['recall'][-1], history['f1'][-1]]
    colors = ['red', 'green', 'blue', 'purple']

    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_title('最终评估指标')
    ax4.set_ylabel('百分比 (%)')
    ax4.set_ylim([90, 100])  # 设置y轴范围为90-100

    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('1.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_final_summary(all_preds, all_labels):
    """打印最终结果摘要"""
    print("\n" + "=" * 50)
    print("最终模型评估结果")
    print("=" * 50)
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终测试准确率: {history['accuracy'][-1]:.2f}%")
    print(f"最终精确率: {history['precision'][-1]:.2f}%")
    print(f"最终召回率: {history['recall'][-1]:.2f}%")
    print(f"最终F1分数: {history['f1'][-1]:.2f}%")

    # 打印混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)
    print("=" * 50)


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
    print(f"开始训练GoogLeNet (设备: {device})")
    print("=" * 50)

    # 训练和评估
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        train(epoch)
        all_preds, all_labels = evaluate()
        print("-" * 50)

    print("训练完成!")

    # 结果可视化
    plot_results()
    print_final_summary(all_preds, all_labels)
    save_history(history, 'GoogLeNet')