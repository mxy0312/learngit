import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练数据
try:
    data = pd.read_csv('train.csv')
    print("成功读取 train.csv 文件")
    print(f"原始数据形状: {data.shape}")

    if 'x' in data.columns and 'y' in data.columns:
        x_data = data['x'].values
        y_data = data['y'].values
    else:
        x_data = data.iloc[:, 0].values
        y_data = data.iloc[:, 1].values

    print(
        f"原始数据范围: x=[{np.nanmin(x_data):.2f}, {np.nanmax(x_data):.2f}], y=[{np.nanmin(y_data):.2f}, {np.nanmax(y_data):.2f}]")
    print(f"NaN值数量: x中有{np.sum(np.isnan(x_data))}个, y中有{np.sum(np.isnan(y_data))}个")

    # 更严格的数据清洗
    # 1. 移除NaN和Inf值
    finite_mask = (np.isfinite(x_data) & np.isfinite(y_data))
    x_data = x_data[finite_mask]
    y_data = y_data[finite_mask]
    print(f"移除NaN/Inf后数据长度: {len(x_data)}")

    if len(x_data) == 0:
        raise ValueError("所有数据都是NaN，使用示例数据")

    # 2. 移除极端异常值（使用更保守的方法）
    # 只有当数据量足够大时才进行异常值检测
    if len(x_data) > 10:
        # 计算四分位距
        q1_x, q3_x = np.percentile(x_data, [25, 75])
        q1_y, q3_y = np.percentile(y_data, [25, 75])
        iqr_x, iqr_y = q3_x - q1_x, q3_y - q1_y

        # 只移除非常极端的异常值（使用较大的倍数）
        if iqr_x > 0 and iqr_y > 0:  # 确保IQR不为0
            outlier_mask = ((x_data >= q1_x - 5 * iqr_x) & (x_data <= q3_x + 5 * iqr_x) &
                            (y_data >= q1_y - 5 * iqr_y) & (y_data <= q3_y + 5 * iqr_y))
            x_data = x_data[outlier_mask]
            y_data = y_data[outlier_mask]
            print(f"移除极端异常值后数据长度: {len(x_data)}")

    # 最终检查
    if len(x_data) < 3:
        raise ValueError("清洗后数据太少，使用示例数据")

    print(f"最终使用数据长度: {len(x_data)}")
    print(f"最终数据范围: x=[{x_data.min():.2f}, {x_data.max():.2f}], y=[{y_data.min():.2f}, {y_data.max():.2f}]")

except (FileNotFoundError, ValueError) as e:
    print(f"数据加载问题: {e}")
    print("使用示例数据")
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# 将数据转换为PyTorch张量
x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

# 原始训练代码（增加NaN检查）
w = torch.tensor([1.0], requires_grad=True)


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(torch.tensor(4.0)).item())

# 训练（增加NaN检查）
for epoch in range(100):
    epoch_loss = 0.0
    valid_samples = 0

    for x, y in zip(x_tensor, y_tensor):
        # 跳过任何可能的NaN值
        if torch.isnan(x) or torch.isnan(y):
            continue

        l = loss(x, y)

        # 检查损失是否为NaN
        if torch.isnan(l):
            print(f"警告: 在epoch {epoch}发现NaN损失，跳过")
            continue

        l.backward()

        # 检查梯度是否为NaN
        if torch.isnan(w.grad):
            print(f"警告: 在epoch {epoch}发现NaN梯度，重置梯度")
            w.grad.data.zero_()
            continue

        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

        epoch_loss += l.item()
        valid_samples += 1

    # 检查权重是否变成NaN
    if torch.isnan(w).any():
        print(f"权重变成NaN，重新初始化")
        w.data = torch.tensor([1.0])
        w.requires_grad = True

    if epoch % 20 == 0 and valid_samples > 0:
        avg_loss = epoch_loss / valid_samples
        print("progress:", epoch, f"avg_loss: {avg_loss:.6f}", f"w: {w.item():.6f}")

print("predict (after training)", 4, forward(torch.tensor(4.0)).item())

# 训练完成后的权重
trained_w = w.item()

# 如果训练后权重仍然是NaN，使用简单的解析解
if np.isnan(trained_w):
    print("训练权重为NaN，计算解析解...")
    # 简单线性回归: w = Σ(xy) / Σ(x²)
    sum_xy = np.sum(x_data * y_data)
    sum_x2 = np.sum(x_data * x_data)
    if sum_x2 != 0:
        trained_w = sum_xy / sum_x2
        w.data = torch.tensor([trained_w])
        print(f"使用解析解权重: w = {trained_w:.6f}")
    else:
        trained_w = 1.0
        print("数据问题，使用默认权重 w = 1.0")

# ================== 生成w与loss关系图 ==================
print(f"\n生成权重w与损失loss的关系图...")

# 在训练得到的w值周围扫描不同的w值
w_values = np.linspace(max(0.1, trained_w - 1.5), trained_w + 1.5, 100)
loss_values = []

print(f"扫描w值范围: [{w_values[0]:.2f}, {w_values[-1]:.2f}]")

for w_test in w_values:
    # 临时设置权重
    with torch.no_grad():
        w.data = torch.tensor([w_test])

    # 计算所有数据点的平均损失
    total_loss = 0.0
    with torch.no_grad():
        for x, y in zip(x_tensor, y_tensor):
            total_loss += loss(x, y).item()

    avg_loss = total_loss / len(x_tensor)
    loss_values.append(avg_loss)

# 找到最小损失对应的w值
min_loss_idx = np.argmin(loss_values)
optimal_w = w_values[min_loss_idx]
min_loss = loss_values[min_loss_idx]

# 绘制w与loss的关系图
plt.figure(figsize=(10, 6))
plt.plot(w_values, loss_values, 'b-', linewidth=2, label='Loss vs Weight w')
plt.axvline(x=trained_w, color='red', linestyle='--', alpha=0.8,
            label=f'训练得到的w={trained_w:.4f}')
plt.plot(optimal_w, min_loss, 'ro', markersize=8,
         label=f'最优点: w={optimal_w:.4f}, loss={min_loss:.4f}')
plt.plot(trained_w, loss_values[np.argmin(np.abs(w_values - trained_w))], 'g^',
         markersize=8, label=f'训练结果: loss={loss_values[np.argmin(np.abs(w_values - trained_w))]:.4f}')

plt.xlabel('权重 Weight (w)')
plt.ylabel('平均损失 Average Loss')
plt.title('权重w与损失Loss的关系')
plt.legend()
plt.grid(True, alpha=0.3)

# 恢复训练得到的权重
with torch.no_grad():
    w.data = torch.tensor([trained_w])

plt.tight_layout()
plt.show()

print(f"\n实验结果:")
print(f"训练得到的权重: w = {trained_w:.6f}")
print(f"理论最优权重: w = {optimal_w:.6f}")
print(f"最小损失值: {min_loss:.6f}")
print(f"权重误差: {abs(trained_w - optimal_w):.6f}")