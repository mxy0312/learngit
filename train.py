import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
df_clean = df.dropna()
x_data = df_clean['x']
y_data = df_clean['y']

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
b_list = []
mse_list = []

for w in np.arange(-1, 3, 0.1):
    for b in np.arange(-10, 10, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / len(x_data))

# 转换为numpy数组便于处理
w_arr = np.array(w_list)
b_arr = np.array(b_list)
mse_arr = np.array(mse_list)

# 创建图形
plt.figure(figsize=(15, 5))

# 修正第一个子图：找到最接近b=0的点
plt.subplot(1, 2, 1)
w_unique = np.unique(w_arr)
mse_w = []

for w_val in w_unique:
    # 找到该w值对应的所有点
    indices = np.where(w_arr == w_val)
    if len(indices[0]) > 0:
        # 在这些点中找到b最接近0的点
        b_subset = b_arr[indices]
        mse_subset = mse_arr[indices]
        # 找到b值最接近0的索引
        closest_b_idx = np.argmin(np.abs(b_subset))
        mse_w.append(mse_subset[closest_b_idx])

plt.plot(w_unique, mse_w, 'b-', linewidth=2)
plt.xlabel('w')
plt.ylabel('MSE Loss')
plt.title('Loss vs w (b≈0)')
plt.grid(True)

# 修正第二个子图：找到最接近w=2的点
plt.subplot(1, 2, 2)
b_unique = np.unique(b_arr)
mse_b = []

for b_val in b_unique:
    # 找到该b值对应的所有点
    indices = np.where(b_arr == b_val)
    if len(indices[0]) > 0:
        # 在这些点中找到w最接近2的点
        w_subset = w_arr[indices]
        mse_subset = mse_arr[indices]
        # 找到w值最接近2的索引
        closest_w_idx = np.argmin(np.abs(w_subset - 1.0))
        mse_b.append(mse_subset[closest_w_idx])

plt.plot(b_unique, mse_b, 'r-', linewidth=2)
plt.xlabel('b')
plt.ylabel('MSE Loss')
plt.title('Loss vs b (w≈1)')
plt.grid(True)

plt.tight_layout()
plt.show()