import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('train.csv')

# 删除 y 缺失的行
df = df.dropna(subset=['y'])

x_q995 = np.percentile(df['x'], 99.5)
df = df[df['x'] <= x_q995]

x_data = df['x'].values
y_data = df['y'].values

# 简单归一化
x_data = (x_data - np.mean(x_data)) / np.std(x_data)
y_data = (y_data - np.mean(y_data)) / np.std(y_data)

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return np.mean((y_pred - y) ** 2)

w_list = np.linspace(0.5, 1.5, 50)
w_loss_list = [loss(x_data, y_data, w, b=0) for w in w_list]

b_list = np.linspace(-1.0, 1.0, 50)
b_loss_list = [loss(x_data, y_data, w=1.0, b=b) for b in b_list]


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(w_list, w_loss_list, 'b-')
plt.xlabel('Weight (w)')
plt.ylabel('Loss (MSE)')
plt.title('Loss vs Weight')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(b_list, b_loss_list, 'r-')
plt.xlabel('Bias (b)')
plt.ylabel('Loss (MSE)')
plt.title('Loss vs Bias')
plt.grid(True)

plt.tight_layout()
plt.show()