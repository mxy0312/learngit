import numpy as np
import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
def forward(x):
    return x * w
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)
plt.figure(figsize=(10, 6))
plt.plot(w_list, mse_list, 'b-', linewidth=2)
plt.title('Weight vs Mean Squared Error (MSE)', fontsize=14)
plt.xlabel('Weight (w)', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 标记最小MSE点
min_mse = min(mse_list)
min_index = mse_list.index(min_mse)
min_w = w_list[min_index]

plt.scatter(min_w, min_mse, color='red', s=100, zorder=5)
plt.annotate(f'Min MSE: {min_mse:.2f}\nw: {min_w}',
             xy=(min_w, min_mse),
             xytext=(min_w+0.5, min_mse+5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10,
             color='red')

plt.tight_layout()
plt.show()