import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据加载和预处理
print("=" * 50)
print("数据加载与预处理")
print("=" * 50)

df = pd.read_csv("train.csv")
df_clean = df.dropna()
x_values = df_clean['x'].values
y_values = df_clean['y'].values

print(f"数据集大小: {len(x_values)} 样本")

# 数据划分：训练集80%，验证集20%
split_idx = int(0.8 * len(x_values))
indices = np.random.permutation(len(x_values))

train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

x_train, y_train = x_values[train_indices], y_values[train_indices]
x_val, y_val = x_values[val_indices], y_values[val_indices]

# 数据标准化（使用训练集的统计量）
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std
x_val_norm = (x_val - x_mean) / x_std
y_val_norm = (y_val - y_mean) / y_std

# 转换为Tensor
x_train_tensor = torch.FloatTensor(x_train_norm).reshape(-1, 1)
y_train_tensor = torch.FloatTensor(y_train_norm).reshape(-1, 1)
x_val_tensor = torch.FloatTensor(x_val_norm).reshape(-1, 1)
y_val_tensor = torch.FloatTensor(y_val_norm).reshape(-1, 1)

# 创建DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"训练集: {len(x_train)} 样本")
print(f"验证集: {len(x_val)} 样本")


# 2. 改进的模型架构
class ImprovedLinearModel(nn.Module):
    def __init__(self, use_deeper=False):
        super().__init__()
        if use_deeper:
            # 更深的网络结构
            self.net = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
        else:
            # 简单线性模型
            self.net = nn.Linear(1, 1)

        # 改进的初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重和偏置参数为正态分布"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 权重w使用正态分布初始化 N(0, 0.1)
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                # 偏置b使用正态分布初始化 N(0, 0.1)
                nn.init.normal_(m.bias, mean=0.0, std=0.1)

    def forward(self, x):
        return self.net(x)


# 3. 早停机制
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0


# 4. 改进的训练函数
def train_with_optimizer(opt_name, opt_class, lr=0.01, epochs=200,
                         use_scheduler=True, use_deeper=False, patience=15):
    model = ImprovedLinearModel(use_deeper=use_deeper)
    criterion = nn.MSELoss()

    # LBFGS需要特殊处理
    is_lbfgs = (opt_name == 'LBFGS')

    if is_lbfgs:
        # LBFGS使用较小的学习率
        optimizer = opt_class(model.parameters(), lr=min(lr, 0.1), max_iter=20)
        use_scheduler = False  # LBFGS不使用学习率调度
    else:
        optimizer = opt_class(model.parameters(), lr=lr)

    # 学习率调度器
    if use_scheduler and not is_lbfgs:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )

    # 早停
    early_stopping = EarlyStopping(patience=patience)

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_losses = []

        if is_lbfgs:
            # LBFGS需要闭包函数
            def closure():
                optimizer.zero_grad()
                y_pred = model(x_train_tensor)
                loss = criterion(y_pred, y_train_tensor)
                loss.backward()
                return loss

            optimizer.step(closure)

            # 计算训练损失
            with torch.no_grad():
                train_pred = model(x_train_tensor)
                train_loss = criterion(train_pred, y_train_tensor)
                train_losses.append(train_loss.item())
        else:
            # 其他优化器使用批处理
            for batch_x, batch_y in train_loader:
                y_pred = model(batch_x)
                loss = criterion(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

        # 验证模式
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor)

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 调整学习率
        if use_scheduler and not is_lbfgs:
            scheduler.step(val_loss)

        # 早停检查
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print(f"  早停触发于第 {epoch + 1} 轮")
            break

    # 加载最佳模型
    model.load_state_dict(early_stopping.best_model)
    best_val_loss = early_stopping.best_loss

    return history, best_val_loss, model.state_dict()


# 5. 优化器比较（按练习5-2要求）
print("\n" + "=" * 50)
print("优化器性能比较")
print("=" * 50)

optimizers_dict = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD,
    'LBFGS': torch.optim.LBFGS,
    'RMSprop': torch.optim.RMSprop,
    'Rprop': torch.optim.Rprop,
    'SGD': torch.optim.SGD,
}

results = {}

for opt_name, opt_class in optimizers_dict.items():
    print(f"\n训练 {opt_name}...")
    history, best_loss, best_state = train_with_optimizer(
        opt_name, opt_class, lr=0.01, epochs=200
    )
    results[opt_name] = {
        'history': history,
        'best_loss': best_loss,
        'best_state': best_state
    }
    print(f"{opt_name}: 最佳验证损失 = {best_loss:.6f}")

# 6. 学习率网格搜索
print("\n" + "=" * 50)
print("学习率调优实验")
print("=" * 50)

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
lr_results = {}

for lr in learning_rates:
    print(f"\n测试学习率: {lr}")
    history, best_loss, best_state = train_with_optimizer(
        'Adam', torch.optim.Adam, lr=lr, epochs=200
    )
    lr_results[f'LR={lr}'] = {
        'history': history,
        'best_loss': best_loss,
        'best_state': best_state
    }
    print(f"  最佳验证损失 = {best_loss:.6f}")

# 7. 测试更深的网络
print("\n" + "=" * 50)
print("网络深度实验")
print("=" * 50)

depth_results = {}
for use_deeper in [False, True]:
    name = "深层网络" if use_deeper else "线性模型"
    print(f"\n训练 {name}...")
    history, best_loss, best_state = train_with_optimizer(
        'Adam', torch.optim.Adam, lr=0.01, epochs=200, use_deeper=use_deeper
    )
    depth_results[name] = {
        'history': history,
        'best_loss': best_loss,
        'best_state': best_state
    }
    print(f"{name}: 最佳验证损失 = {best_loss:.6f}")

# 8. 可视化结果
print("\n生成可视化图表...")

# 优化器比较可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 训练损失对比
axes[0, 0].set_title('训练损失曲线对比', fontsize=12, fontweight='bold')
for opt_name in results.keys():
    axes[0, 0].plot(results[opt_name]['history']['train_loss'],
                    label=opt_name, linewidth=2)
axes[0, 0].set_xlabel('训练轮数')
axes[0, 0].set_ylabel('训练损失')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 验证损失对比
axes[0, 1].set_title('验证损失曲线对比', fontsize=12, fontweight='bold')
for opt_name in results.keys():
    axes[0, 1].plot(results[opt_name]['history']['val_loss'],
                    label=opt_name, linewidth=2)
axes[0, 1].set_xlabel('训练轮数')
axes[0, 1].set_ylabel('验证损失')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# 最佳损失对比
axes[1, 0].set_title('优化器最佳损失对比', fontsize=12, fontweight='bold')
opt_names = list(results.keys())
best_losses = [results[name]['best_loss'] for name in opt_names]
bars = axes[1, 0].bar(range(len(opt_names)), best_losses, alpha=0.7)
axes[1, 0].set_xticks(range(len(opt_names)))
axes[1, 0].set_xticklabels(opt_names, rotation=45, ha='right')
axes[1, 0].set_ylabel('最佳验证损失')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 为每个柱子添加数值标签
for i, (bar, loss) in enumerate(zip(bars, best_losses)):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

# 学习率调整曲线
axes[1, 1].set_title('学习率变化（Adam优化器）', fontsize=12, fontweight='bold')
axes[1, 1].plot(results['Adam']['history']['lr'], linewidth=2, color='purple')
axes[1, 1].set_xlabel('训练轮数')
axes[1, 1].set_ylabel('学习率')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 学习率实验可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].set_title('不同学习率的验证损失曲线', fontsize=12, fontweight='bold')
for lr_name, result in lr_results.items():
    axes[0].plot(result['history']['val_loss'], label=lr_name, linewidth=2)
axes[0].set_xlabel('训练轮数')
axes[0].set_ylabel('验证损失')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

axes[1].set_title('学习率与最佳损失的关系', fontsize=12, fontweight='bold')
lr_names = [f"{lr}" for lr in learning_rates]
lr_losses = [lr_results[f'LR={lr}']['best_loss'] for lr in learning_rates]
bars = axes[1].bar(lr_names, lr_losses, alpha=0.7, color='steelblue')
axes[1].set_xlabel('学习率')
axes[1].set_ylabel('最佳验证损失')
axes[1].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, loss in zip(bars, lr_losses):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 找到并保存全局最佳模型
print("\n" + "=" * 50)
print("保存最佳模型")
print("=" * 50)

all_results = {**results, **lr_results, **depth_results}
best_model_name = min(all_results.keys(), key=lambda x: all_results[x]['best_loss'])
best_result = all_results[best_model_name]

print(f"\n全局最佳模型配置: {best_model_name}")
print(f"最佳验证损失: {best_result['best_loss']:.6f}")

# 保存模型和训练信息
os.makedirs("best_model", exist_ok=True)

torch.save({
    'model_state_dict': best_result['best_state'],
    'best_val_loss': best_result['best_loss'],
    'x_mean': x_mean,
    'x_std': x_std,
    'y_mean': y_mean,
    'y_std': y_std,
    'model_name': best_model_name,
    'train_history': best_result['history']
}, "best_model/best_model.pth")

print(f"✓ 模型已保存到: best_model/best_model.pth")

# 10. 生成性能报告
print("\n" + "=" * 50)
print("性能总结报告")
print("=" * 50)

print("\n【优化器排名】")
sorted_opts = sorted(results.items(), key=lambda x: x[1]['best_loss'])
for i, (name, result) in enumerate(sorted_opts, 1):
    print(f"{i}. {name:15s} - 验证损失: {result['best_loss']:.6f}")

print("\n【学习率排名】")
sorted_lrs = sorted(lr_results.items(), key=lambda x: x[1]['best_loss'])
for i, (name, result) in enumerate(sorted_lrs, 1):
    print(f"{i}. {name:10s} - 验证损失: {result['best_loss']:.6f}")

print("\n【网络架构比较】")
for name, result in depth_results.items():
    print(f"{name:10s} - 验证损失: {result['best_loss']:.6f}")

print("\n" + "=" * 50)
print("训练完成！")
print("=" * 50)