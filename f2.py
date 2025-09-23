import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

def forward(x, w, b):
    return x * w + b

def compute_loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return np.mean((y_pred - y) **2)

w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)

W, B = np.meshgrid(w_range, b_range)

loss_values = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        loss_values[i, j] = compute_loss(x, y, W[i, j], B[i, j])

min_idx = np.unravel_index(np.argmin(loss_values), loss_values.shape)
best_w = W[min_idx]
best_b = B[min_idx]
min_loss = loss_values[min_idx]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(W, B, loss_values,
                       cmap='coolwarm',
                       alpha=0.8,
                       edgecolor='none')

ax.scatter(best_w, best_b, min_loss,
           color='black',
           s=200,
           marker='*',
           label=f'最优参数: w={best_w:.2f}, b={best_b:.2f}')

ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('b', fontsize=12)
ax.set_zlabel('损失 (MSE)', fontsize=12)
ax.set_title('线性模型 y = w*x + b 的损失曲面', fontsize=15)

cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10)
cbar.set_label('损失值', rotation=270, labelpad=20)

ax.legend()
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()

print(f"最优参数: w = {best_w:.4f}, b = {best_b:.4f}")
print(f"最小损失值: {min_loss:.6f}")
