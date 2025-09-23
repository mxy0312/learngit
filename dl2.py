# 导入所需要的工具包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练数据
try:
    data = pd.read_csv('train.csv')
    print("成功读取 train.csv 文件")
    if 'x' in data.columns and 'y' in data.columns:
        x_data = data['x'].values
        y_data = data['y'].values
    else:
        x_data = data.iloc[:, 0].values
        y_data = data.iloc[:, 1].values

    # 数据清洗：移除nan值和异常值
    print(f"原始数据长度: {len(x_data)}")

    # 检查nan值
    nan_mask = ~(np.isnan(x_data) | np.isnan(y_data) | np.isinf(x_data) | np.isinf(y_data))
    x_data = x_data[nan_mask]
    y_data = y_data[nan_mask]
    print(f"移除nan/inf后数据长度: {len(x_data)}")


    # 移除极端异常值（使用IQR方法）
    def remove_outliers(x, y, factor=3):
        # 对x和y分别计算四分位距
        q1_x, q3_x = np.percentile(x, [25, 75])
        q1_y, q3_y = np.percentile(y, [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y

        # 定义异常值边界
        lower_x, upper_x = q1_x - factor * iqr_x, q3_x + factor * iqr_x
        lower_y, upper_y = q1_y - factor * iqr_y, q3_y + factor * iqr_y

        # 保留非异常值
        mask = (x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y)
        return x[mask], y[mask]


    x_data, y_data = remove_outliers(x_data, y_data)
    print(f"移除异常值后数据长度: {len(x_data)}")

except FileNotFoundError:
    print("未找到 train.csv 文件，使用示例数据")
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([3.0, 5.0, 7.0, 9.0, 11.0])

print(f"清洗后数据范围: x=[{x_data.min():.2f}, {x_data.max():.2f}], y=[{y_data.min():.2f}, {y_data.max():.2f}]")
print(f"数据样本数: {len(x_data)}")


# 计算正向传播结果 y_pred = k * x + b
def forward(x, k, b):
    return k * x + b


# 计算损失值 MSE = mean((y_pred - y)^2)
def calculate_mse(k, b):
    total_loss = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val, k, b)
        total_loss += (y_pred - y_val) ** 2
    return total_loss / len(x_data)


# 使用解析解找到最佳参数
def find_optimal_parameters():
    try:
        # 构造设计矩阵 X = [1, x]
        X = np.column_stack([np.ones(len(x_data)), x_data])
        # 正规方程: θ = (X^T * X)^(-1) * X^T * y
        theta = np.linalg.inv(X.T @ X) @ X.T @ y_data
        b_optimal, k_optimal = theta[0], theta[1]

        # 检查结果是否有效
        if np.isnan(k_optimal) or np.isnan(b_optimal):
            print("警告: 解析解计算出现nan值，使用备用方法")
            # 使用简单的最小二乘法
            k_optimal = np.corrcoef(x_data, y_data)[0, 1] * (np.std(y_data) / np.std(x_data))
            b_optimal = np.mean(y_data) - k_optimal * np.mean(x_data)

        return k_optimal, b_optimal
    except Exception as e:
        print(f"计算最佳参数时出错: {e}")
        # 使用简单线性回归的备用方法
        k_optimal = np.sum((x_data - np.mean(x_data)) * (y_data - np.mean(y_data))) / np.sum(
            (x_data - np.mean(x_data)) ** 2)
        b_optimal = np.mean(y_data) - k_optimal * np.mean(x_data)
        return k_optimal, b_optimal


# 找到最佳参数
k_optimal, b_optimal = find_optimal_parameters()
print(f"最佳参数: k={k_optimal:.4f}, b={b_optimal:.4f}")

# 实验1: 固定最佳b，观察k与loss的关系
print("\n=== 实验1: k与loss的关系（固定b为最佳值）===")
k_list = []
mse_k_list = []

# 使用linspace替代arange避免浮点数精度问题
k_values = np.linspace(k_optimal - 2, k_optimal + 2, 41)  # 41个点，步长约0.1
print(f"k范围: {k_optimal - 2:.2f} 到 {k_optimal + 2:.2f}")

for k in k_values:
    mse = calculate_mse(k, b_optimal)
    k_list.append(k)
    mse_k_list.append(mse)
    if abs(k - k_optimal) < 0.1:  # 在最佳k附近打印
        print(f'k={k:.2f}, b={b_optimal:.4f} (最佳), MSE={mse:.6f}')

# 实验2: 固定最佳k，观察b与loss的关系
print("\n=== 实验2: b与loss的关系（固定k为最佳值）===")
b_list = []
mse_b_list = []

# 使用linspace替代arange避免浮点数精度问题
b_values = np.linspace(b_optimal - 2, b_optimal + 2, 41)  # 41个点，步长约0.1
print(f"b范围: {b_optimal - 2:.2f} 到 {b_optimal + 2:.2f}")

for b in b_values:
    mse = calculate_mse(k_optimal, b)
    b_list.append(b)
    mse_b_list.append(mse)
    if abs(b - b_optimal) < 0.1:  # 在最佳b附近打印
        print(f'k={k_optimal:.4f} (最佳), b={b:.2f}, MSE={mse:.6f}')

# 检查数据
print(f"\n=== 调试信息 ===")
print(f"k_list长度: {len(k_list)}, 前5个值: {k_list[:5]}")
print(f"mse_k_list长度: {len(mse_k_list)}, 前5个值: {mse_k_list[:5]}")
print(f"b_list长度: {len(b_list)}, 前5个值: {b_list[:5]}")
print(f"mse_b_list长度: {len(mse_b_list)}, 前5个值: {mse_b_list[:5]}")
print(f"k_optimal: {k_optimal}, b_optimal: {b_optimal}")

# 创建图形显示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 图1: k vs loss
if len(k_list) > 0 and len(mse_k_list) > 0:
    ax1.plot(k_list, mse_k_list, 'b-', linewidth=2, marker='o', markersize=4, label='Loss vs k')
    ax1.axvline(x=k_optimal, color='red', linestyle='--', alpha=0.8, label=f'最佳k={k_optimal:.3f}')
    min_loss = calculate_mse(k_optimal, b_optimal)
    ax1.plot(k_optimal, min_loss, 'ro', markersize=8, label='最小loss点')
    ax1.set_xlabel('Weight k')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'k与Loss的关系 (固定b={b_optimal:.3f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    print(f"图1绘制成功，k范围: [{min(k_list):.2f}, {max(k_list):.2f}]")
else:
    ax1.text(0.5, 0.5, '数据为空', ha='center', va='center', transform=ax1.transAxes)
    print("图1数据为空")

# 图2: b vs loss
if len(b_list) > 0 and len(mse_b_list) > 0:
    ax2.plot(b_list, mse_b_list, 'g-', linewidth=2, marker='s', markersize=4, label='Loss vs b')
    ax2.axvline(x=b_optimal, color='red', linestyle='--', alpha=0.8, label=f'最佳b={b_optimal:.3f}')
    min_loss = calculate_mse(k_optimal, b_optimal)
    ax2.plot(b_optimal, min_loss, 'ro', markersize=8, label='最小loss点')
    ax2.set_xlabel('Bias b')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title(f'b与Loss的关系 (固定k={k_optimal:.3f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    print(f"图2绘制成功，b范围: [{min(b_list):.2f}, {max(b_list):.2f}]")
else:
    ax2.text(0.5, 0.5, '数据为空', ha='center', va='center', transform=ax2.transAxes)
    print("图2数据为空")

plt.tight_layout()
plt.show()

# 输出结果
min_loss = calculate_mse(k_optimal, b_optimal)
print(f"\n=== 结果总结 ===")
print(f"最佳拟合直线: y = {k_optimal:.4f}x + {b_optimal:.4f}")
print(f"最小MSE损失: {min_loss:.6f}")