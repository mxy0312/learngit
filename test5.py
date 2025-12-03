import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原代码数据部分
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # 使用正态分布初始化权重和偏置
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.linear.bias, mean=0.0, std=1.0)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# 存储训练过程的列表
all_losses = []
all_weights = []
all_biases = []


# 训练函数
def train_with_optimizer(optimizer_name, model, criterion, x_data, y_data, epochs=1000, lr=0.01):
    # 选择优化器
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_name == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
    elif optimizer_name == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

    losses = []
    weights = []
    biases = []

    for epoch in range(epochs):
        if optimizer_name == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                y_pred = model(x_data)
                loss = criterion(y_pred, y_data)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
        else:
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        weights.append(model.linear.weight.item())
        biases.append(model.linear.bias.item())

        if epoch % 100 == 0:
            print(epoch, loss.item())

    return losses, weights, biases, model.linear.weight.item(), model.linear.bias.item()


# 可视化函数
def plot_training_results(results):
    plt.figure(figsize=(15, 10))

    # 损失曲线对比
    plt.subplot(2, 3, 1)
    for name, (losses, weights, biases, final_w, final_b) in results.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves Comparison')
    plt.legend()
    plt.grid(True)

    # 权重变化过程
    plt.subplot(2, 3, 2)
    for name, (losses, weights, biases, final_w, final_b) in results.items():
        plt.plot(weights, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Weight w')
    plt.title('Weight Adjustment Process')
    plt.legend()
    plt.grid(True)

    # 偏置变化过程
    plt.subplot(2, 3, 3)
    for name, (losses, weights, biases, final_w, final_b) in results.items():
        plt.plot(biases, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Bias b')
    plt.title('Bias Adjustment Process')
    plt.legend()
    plt.grid(True)

    # 拟合结果
    plt.subplot(2, 3, 4)
    x_np = x_data.numpy().flatten()
    y_np = y_data.numpy().flatten()
    plt.scatter(x_np, y_np, color='black', label='True data')

    for name, (losses, weights, biases, final_w, final_b) in results.items():
        x_line = np.linspace(0, 4, 100)
        y_line = final_w * x_line + final_b
        plt.plot(x_line, y_line, label=f'{name}: y={final_w:.3f}x+{final_b:.3f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitting Results')
    plt.legend()
    plt.grid(True)

    # 参数空间轨迹
    plt.subplot(2, 3, 5)
    for name, (losses, weights, biases, final_w, final_b) in results.items():
        plt.plot(weights, biases, label=name)
        plt.scatter([final_w], [final_b], marker='*', s=100)
    plt.xlabel('Weight w')
    plt.ylabel('Bias b')
    plt.title('Parameter Space Trajectory')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# 参数调节研究
def study_parameters():
    epochs_list = [100, 500, 1000, 2000]
    lr_list = [0.001, 0.01, 0.1, 0.5]

    plt.figure(figsize=(12, 5))

    # Epoch数量影响
    plt.subplot(1, 2, 1)
    final_losses_epoch = []
    for epochs in epochs_list:
        model = LinearModel()
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_losses_epoch.append(loss.item())

    plt.plot(epochs_list, final_losses_epoch, 'o-')
    plt.xlabel('Epochs')
    plt.ylabel('Final Loss')
    plt.title('Effect of Epoch Number')
    plt.grid(True)

    # 学习率影响
    plt.subplot(1, 2, 2)
    final_losses_lr = []
    for lr in lr_list:
        model = LinearModel()
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for epoch in range(1000):
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_losses_lr.append(loss.item())

    plt.semilogx(lr_list, final_losses_lr, 'o-')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Loss')
    plt.title('Effect of Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('parameter_study.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主执行部分
if __name__ == "__main__":
    # 原代码的训练部分（保持格式，但包装成函数）
    print("=== Original SGD Optimizer ===")
    model_original = LinearModel()
    criterion_original = torch.nn.MSELoss(size_average=False)

    losses_original = []
    weights_original = []
    biases_original = []

    optimizer_original = torch.optim.SGD(model_original.parameters(), lr=0.01)
    for epoch in range(1000):
        y_pred = model_original(x_data)
        loss = criterion_original(y_pred, y_data)
        losses_original.append(loss.item())
        weights_original.append(model_original.linear.weight.item())
        biases_original.append(model_original.linear.bias.item())

        if epoch % 100 == 0:
            print(epoch, loss.item())

        optimizer_original.zero_grad()
        loss.backward()
        optimizer_original.step()

    print('w=', model_original.linear.weight.item())
    print('b=', model_original.linear.bias.item())

    x_test = torch.Tensor([[4.0]])
    y_test = model_original(x_test)
    print('y_pred=', y_test.data.item())

    # 使用这三种优化器进行比较
    print("\n=== Comparing Different Optimizers ===")
    optimizers_to_compare = ['Adamax', 'ASGD', 'LBFGS']
    results = {}

    # 添加原SGD结果
    results['SGD'] = (losses_original, weights_original, biases_original,
                      model_original.linear.weight.item(), model_original.linear.bias.item())

    # 训练其他优化器
    for optimizer_name in optimizers_to_compare:
        print(f"\n--- Training with {optimizer_name} ---")
        model = LinearModel()
        criterion = torch.nn.MSELoss(size_average=False)
        losses, weights, biases, final_w, final_b = train_with_optimizer(
            optimizer_name, model, criterion, x_data, y_data, epochs=1000, lr=0.01)
        results[optimizer_name] = (losses, weights, biases, final_w, final_b)

        print(f'{optimizer_name} - w={final_w:.6f}, b={final_b:.6f}')
        x_test = torch.Tensor([[4.0]])
        y_test = model(x_test)
        print(f'{optimizer_name} - y_pred={y_test.data.item():.6f}')

    # 绘制对比图
    print("\n=== Generating Visualizations ===")
    plot_training_results(results)

    # 参数调节研究
    print("\n=== Studying Parameter Effects ===")
    study_parameters()