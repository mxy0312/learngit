import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'train.csv'
df = pd.read_csv(file)

cols = ['x', 'y']
if not all(c in df.columns for c in cols):
    raise KeyError(f'CSV 必须包含 {cols} 两列，当前列名：{list(df.columns)}')

na_cnt = df[cols].isna().sum()
if na_cnt.any():
    print('发现缺失值\n', na_cnt)
    df = df.dropna(subset=cols)   # 简单丢弃；可换成填充
    print('缺失行已删除，剩余样本数：', len(df))

x = df['x'].to_numpy(dtype=float)
y = df['y'].to_numpy(dtype=float)

x_bar, y_bar = x.mean(), y.mean()
w_opt = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
b_opt = y_bar - w_opt * x_bar


W = np.linspace(w_opt - 2, w_opt + 2, 200)
B = np.linspace(b_opt - 2, b_opt + 2, 200)

loss_w = [(np.mean((w * x + b_opt - y) ** 2)) for w in W]

loss_b = [(np.mean((w_opt * x + b - y) ** 2)) for b in B]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(W, loss_w, label='MSE')
plt.axvline(w_opt, color='r', ls='--', label=f'opt w={w_opt:.4f}')
plt.xlabel('w'); plt.ylabel('MSE'); plt.title('Loss vs. w')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(B, loss_b, label='MSE')
plt.axvline(b_opt, color='r', ls='--', label=f'opt b={b_opt:.4f}')
plt.xlabel('b'); plt.ylabel('MSE'); plt.title('Loss vs. b')
plt.legend()

plt.tight_layout()
plt.show()