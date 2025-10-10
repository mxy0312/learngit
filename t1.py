import pandas as pd
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取和数据预处理
# 从CSV文件中读取数据
df = pd.read_csv('train.csv')

df = df.dropna(subset=['y'])

x_q995 = np.percentile(df['x'], 99.5)

df = df[df['x'] <= x_q995]

x_data = t.tensor(df['x'].values, dtype=t.float32).view(-1, 1)
y_data = t.tensor(df['y'].values, dtype=t.float32).view(-1, 1)

class LinearModel(nn.Module):

    def __init__(self):

        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

    def forward(self, x):

        return self.linear(x)

def train_model(optimizer, optimizer_name, epochs=1000, lr=0.001):

    model = LinearModel()
    criterion = nn.MSELoss()

    if optimizer_name == 'SGD':
        # 创建随机梯度下降优化器
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        # 创建Adam优化器
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        # 创建RMSprop优化器
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # 初始化训练历史记录字典，用于存储训练过程中的各项指标
    history = {'loss': [], 'w': [], 'b': [], 'grad_w': [], 'grad_b': []}

    for epoch in range(epochs):
        # 前向传播计算预测值
        y_pred = model(x_data)

        loss = criterion(y_pred, y_data)

        # 记录当前轮次的损失值和模型参数
        history['loss'].append(loss.item())
        history['w'].append(model.linear.weight.item())
        history['b'].append(model.linear.bias.item())

        # 清零梯度缓存，准备进行反向传播
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()

        history['grad_w'].append(model.linear.weight.grad.item() if model.linear.weight.grad is not None else 0)
        history['grad_b'].append(model.linear.bias.grad.item() if model.linear.bias.grad is not None else 0)

        # 更新模型参数
        optimizer.step()

    return model, history

# 定义要比较的优化器列表，包含优化器名称和对应的类
optimizers = [
    ('SGD', optim.SGD),
    ('Adam', optim.Adam),
    ('RMSprop', optim.RMSprop)
]


# 训练并比较三种优化器的性能
epochs = 2000
lr = 0.0000008
results = {}
a = 'SGD'
b = 'optim.SGD'
# 使用指定优化器训练模型
model, history = train_model( b,a, epochs=epochs, lr=lr)
results[a] = history

# 使用训练好的模型进行预测
x_test = t.Tensor([[50.0]])
y_test = model(x_test)
print(f'当epoch={epochs}时，并且lr={lr}时，使用{a}优化器预测结果: x=50.0, y_pred={y_test.item():.4f}')

# 三种优化器对比
# optimizers = [
#     ('SGD', optim.SGD),
#     ('Adam', optim.Adam),
#     ('RMSprop', optim.RMSprop)
# ]
#
# epochs = 150000
# lr = 0.00001
# res = {}
# for x,y in optimizers:
#     model, history = train_model(y, x, epochs=epochs, lr=lr)
#     res[x] = history
#
#     # 使用训练好的模型进行预测
#     x_test = t.Tensor([[50.0]])
#     y_test = model(x_test)
#
# # 1. 优化器性能对比可视化
# plt.figure(figsize=(10, 6))
# # 损失曲线对比图
# plt.subplot(1,1,1)
# for opt_name, history in res.items():
#     # 只显示前200轮的损失值以便更好地观察差异
#     plt.plot(history['loss'][:], label=opt_name)
# plt.xlabel('训练轮次 (Epoch)')
# plt.ylabel('损失值 (Loss)')
# plt.title('三种优化器损失曲线对比')
# plt.legend()
# plt.grid(True)

# 2. 参数w和b的调节过程可视化
adam_history = results[a]

# 权重w的变化过程图
plt.subplot(2, 2, 1)
plt.plot(adam_history['w'])
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('权重 w')
plt.title(f'{a}优化器: 权重w调节过程')
plt.grid(True)

# 偏置b的变化过程图
plt.subplot(2, 2, 2)
plt.plot(adam_history['b'])
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('偏置 b')
plt.title(f'{a}优化器: 偏置b调节过程')
plt.grid(True)

# 3. 学习率影响分析
learning_rates = [0.0000005, 0.0000008, 0.000001, 0.000005]
lr_losses = {}

# 使用不同学习率训练模型并记录损失
for lr in learning_rates:
    _, history = train_model(b, a, epochs=2000, lr=lr)
    lr_losses[lr] = history['loss']

# 绘制不同学习率下的损失曲线
plt.subplot(2, 2, 3)
for lr, losses in lr_losses.items():
    plt.plot(losses, label=f'LR={lr}')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('损失值 (Loss)')
plt.title('不同学习率对训练的影响')
plt.legend()
plt.grid(True)

# 4. 训练轮次影响分析（固定学习率=0.01）
epochs_list = [1000, 1500, 2000, 2500]
epoch_losses = {}

# 使用不同训练轮次训练模型并记录损失
for epochs in epochs_list:
    _, history = train_model(b, a, epochs=epochs, lr=0.0000008)
    epoch_losses[epochs] = history['loss']

# 绘制不同训练轮次下的损失曲线
plt.subplot(2, 2, 4)
for epochs, losses in epoch_losses.items():
    plt.plot(losses, label=f'Epochs={epochs}')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('损失值 (Loss)')
plt.title('不同训练轮次对收敛的影响')
plt.legend()
plt.grid(True)

# 调整子图布局并保存图像
plt.tight_layout()
plt.show()

# 输出最佳性能结果
final_losses = {opt: history['loss'][-1] for opt, history in results.items()}
best_optimizer = min(final_losses, key=final_losses.get)
print(f"{best_optimizer}最终损失: {final_losses[best_optimizer]:.6f}")