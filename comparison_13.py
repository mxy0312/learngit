# 加载模型+绘制损失曲线
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 使用CPU
device = torch.device("cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# googleNet模型
class MiniInception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(MiniInception, self).__init__()
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

        # Inception模块：缩减通道数
        self.inception1 = MiniInception(32, 16, 16, 24, 8, 16, 16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception2 = MiniInception(72, 24, 24, 32, 12, 24, 24)

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

# ResNet模型
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: res = self.downsample(x)
        out += res
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ch = 16
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, 2, 1)
        self.layer2 = self._make_layer(32, 2, 2)
        self.layer3 = self._make_layer(64, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 10)
    def _make_layer(self, out_ch, blocks, stride):
        downsample = None
        if stride !=1 or self.in_ch != out_ch:
            downsample = nn.Sequential(nn.Conv2d(self.in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
        layers = [BasicBlock(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks): layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(-1, 64)
        x = self.dropout(x)
        return self.fc(x)

# ====================== 3. 加载模型并测试 ======================
def load_and_evaluate(model_class, weights_path, loss_path, model_name):
    """
    加载保存的模型权重和损失数据，返回测试指标和损失曲线数据
    """
    # 1. 初始化模型结构
    model = model_class().to(device)
    # 2. 加载保存的权重
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # 3. 切换到评估模式
    model.eval()

    # 4. 测试并计算指标
    all_preds = []
    all_targets = []
    start_time = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_targets, all_preds) * 100
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    infer_time = time.time() - start_time

    # 加载损失数据
    loss_data = np.load(loss_path)
    train_losses = loss_data['train_losses']

    return {
        "模型": model_name,
        "准确率(%)": round(acc, 2),
        "精确率": round(prec, 4),
        "召回率": round(rec, 4),
        "F1": round(f1, 4),
        "推理时间(秒)": round(infer_time, 2),
        "训练损失": train_losses
    }

# ====================== 4. 配置权重路径并对比 ======================
if __name__ == "__main__":
    # 配置每个模型的“权重路径”和“损失数据路径”
    model_configs = [
        (GoogLeNet,
         'D:\\School\\course\\Thi-up pic\\Last\\models\\GoogleNet_model.pth',
         'D:\\School\\course\\Thi-up pic\\Last\\models\\GoogLeNet_loss_data.npz',
         "GoogLeNet"),
        (ResNet,
         'D:\\School\\course\\Thi-up pic\\Last\\models\\ResNet_model.pth',
         'D:\\School\\course\\Thi-up pic\\Last\\models\\ResNet_loss_data.npz',
         "ResNet")
    ]

    # 批量加载并对比
    results = []
    for config in model_configs:
        print(f"正在加载 {config[3]}...")
        # 传递四个参数：模型类、权重路径、损失路径、模型名
        res = load_and_evaluate(config[0], config[1], config[2], config[3])
        results.append(res)

    # 打印对比结果
    print("\n" + "=" * 80)
    print("已保存模型对比结果")
    print("=" * 80)
    print(f"{'模型':<12} {'准确率(%)':<12} {'精确率':<10} {'召回率':<10} {'F1':<10} {'推理时间(秒)':<12}")
    print("-" * 80)
    for res in results:
        print(
            f"{res['模型']:<12} {res['准确率(%)']:<12} {res['精确率']:<10} {res['召回率']:<10} {res['F1']:<10} {res['推理时间(秒)']:<12}")

    # 可视化对比
    models = [res['模型'] for res in results]
    accs = [res['准确率(%)'] for res in results]
    times = [res['推理时间(秒)'] for res in results]
    precs = [res['精确率'] for res in results]  # 提取精确率
    recs = [res['召回率'] for res in results]  # 提取召回率
    f1s = [res['F1'] for res in results]  # 提取F1分数

    # 设置画布大小
    plt.figure(figsize=(10, 6))

    # 1. 准确率对比
    plt.subplot(2, 3, 1)
    bars1 = plt.bar(models, accs, color='skyblue')
    plt.title('模型准确率对比', fontsize=11)
    plt.ylabel('准确率(%)')
    plt.ylim(98, 100)  # 聚焦差异区间
    plt.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.03,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. 推理时间对比
    plt.subplot(2, 3, 2)
    bars2 = plt.bar(models, times, color='lightcoral')
    plt.title('模型推理时间对比', fontsize=11)
    plt.ylabel('推理时间(秒)')
    plt.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. 精确率对比
    plt.subplot(2, 3, 3)
    bars3 = plt.bar(models, precs, color='lightgreen')
    plt.title('模型精确率对比', fontsize=11)
    plt.ylabel('精确率')
    plt.ylim(0.98, 1.0)
    plt.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0005,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. 召回率对比
    plt.subplot(2, 3, 4)
    bars4 = plt.bar(models, recs, color='gold')
    plt.title('模型召回率对比', fontsize=11)
    plt.ylabel('召回率')
    plt.ylim(0.98, 1.0)
    plt.grid(axis='y', alpha=0.3)

    for bar in bars4:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0005,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 5. F1分数对比
    plt.subplot(2, 3, 5)
    bars5 = plt.bar(models, f1s, color='plum')
    plt.title('模型F1分数对比', fontsize=11)
    plt.ylabel('F1分数')
    plt.ylim(0.98, 1.0)
    plt.grid(axis='y', alpha=0.3)

    for bar in bars5:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0005,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6. 损失曲线对比
    plt.subplot(2, 3, 6)
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    for i, res in enumerate(results):
        epochs = range(1, len(res['训练损失']) + 1)
        plt.plot(epochs, res['训练损失'], color=colors[i], linestyle=linestyles[i],
                 linewidth=2, label=res['模型'])
    plt.title('模型训练损失曲线对比', fontsize=11)
    plt.xlabel('训练轮次（Epoch）')
    plt.ylabel('损失值')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.show()