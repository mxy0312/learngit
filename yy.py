import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义模型和损失函数
def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

def mse(w, b):
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        loss_val = loss(x_val, y_val, w, b)
        l_sum += loss_val
    return l_sum / len(x_data)


def train_model():
    best_w = 0
    best_b = 0
    best_loss = float('inf')

    for w in np.arange(0.0, 4.1, 0.1):
        for b in np.arange(-2.0, 2.1, 0.1):
            current_loss = mse(w, b)
            if current_loss < best_loss:
                best_loss = current_loss
                best_w = w
                best_b = b

    return best_w, best_b, best_loss


best_w, best_b, best_loss = train_model()
print(f"最优参数: w = {best_w:.2f}, b = {best_b:.2f}, 最小MSE = {best_loss:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
w_list = np.arange(0.0, 4.1, 0.1)
mse_list_w = [mse(w, best_b) for w in w_list]

plt.plot(w_list, mse_list_w, 'b-', linewidth=2)
plt.title(f'Weight vs MSE (b fixed at {best_b:.1f})', fontsize=12)
plt.xlabel('Weight (w)', fontsize=10)
plt.ylabel('MSE', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

min_index_w = np.argmin(mse_list_w)
min_w = w_list[min_index_w]
min_mse_w = mse_list_w[min_index_w]
plt.scatter(min_w, min_mse_w, color='red', s=80, zorder=5)
plt.annotate(f'Min MSE: {min_mse_w:.2f}\nw: {min_w:.1f}',
             xy=(min_w, min_mse_w),
             xytext=(min_w + 0.5, min_mse_w + 2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=8,
             color='red')

plt.subplot(1, 2, 2)
b_list = np.arange(-2.0, 2.1, 0.1)
mse_list_b = [mse(best_w, b) for b in b_list]

plt.plot(b_list, mse_list_b, 'g-', linewidth=2)
plt.title(f'Bias vs MSE (w fixed at {best_w:.1f})', fontsize=12)
plt.xlabel('Bias (b)', fontsize=10)
plt.ylabel('MSE', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

min_index_b = np.argmin(mse_list_b)
min_b = b_list[min_index_b]
min_mse_b = mse_list_b[min_index_b]
plt.scatter(min_b, min_mse_b, color='red', s=80, zorder=5)
plt.annotate(f'Min MSE: {min_mse_b:.2f}\nb: {min_b:.1f}',
             xy=(min_b, min_mse_b),
             xytext=(min_b + 0.3, min_mse_b + 2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=8,
             color='red')

plt.tight_layout()
plt.show()

print("\n测试最优模型:")
for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val, best_w, best_b)
    print(f"x={x_val}, 真实值={y_val}, 预测值={y_pred:.2f}, 误差={abs(y_pred - y_val):.2f}")

plt.figure(figsize=(8, 6))
x_range = np.linspace(0, 4, 100)
y_pred_range = forward(x_range, best_w, best_b)

plt.scatter(x_data, y_data, color='blue', s=100, label='真实数据', zorder=5)
plt.plot(x_range, y_pred_range, 'r-', label=f'拟合直线: y = {best_w:.2f}x + {best_b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()