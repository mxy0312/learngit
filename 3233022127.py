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

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
def forward(x):
    return x * w + b
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0,4.1,0.1):
        print('w=', w)
        print('b=',b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum / 3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3)

# 转换为numpy数组便于处理
w_arr = np.array(w_list)
b_arr = np.array(b_list)
mse_arr = np.array(mse_list)

# 创建图形
plt.figure(figsize=(15, 5))
# 1. w和loss的关系（固定b=0）
plt.subplot(1, 2, 1)
w_unique = np.unique(w_arr)
mse_w = []
for w_val in w_unique:
    # 选择b=0时的MSE值
    indices = np.where((w_arr == w_val) & (b_arr == 0.0))
    if len(indices[0]) > 0:
        mse_w.append(mse_arr[indices[0][0]])
plt.plot(w_unique, mse_w, 'b-', linewidth=2)
plt.xlabel('w')
plt.ylabel('MSE Loss')
plt.title('Loss vs w (b=0)')
plt.grid(True)
# 2. b和loss的关系（固定w=2）
plt.subplot(1, 2, 2)
b_unique = np.unique(b_arr)
mse_b = []
for b_val in b_unique:
    # 选择w=2时的MSE值
    indices = np.where((b_arr == b_val) & (w_arr == 2.0))
    if len(indices[0]) > 0:
        mse_b.append(mse_arr[indices[0][0]])
plt.plot(b_unique, mse_b, 'r-', linewidth=2)
plt.xlabel('b')
plt.ylabel('MSE Loss')
plt.title('Loss vs b (w=2)')
plt.grid(True)
plt.tight_layout()
plt.show()

