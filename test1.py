import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 读取train.csv数据
def load_data(filename):
    try:
        data = pd.read_csv(filename)
        x_data = data['x'].values
        y_data = data['y'].values
        print("成功读取train.csv数据")
        return x_data, y_data
    except:
        # 如果文件不存在，创建示例数据
        print("未找到train.csv，创建示例数据")
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([2.1, 3.9, 6.2, 8.1, 9.8])

        # 保存为CSV文件
        df = pd.DataFrame({'x': x_data, 'y': y_data})
        df.to_csv('train.csv', index=False)
        return x_data, y_data


# 定义模型和损失函数
def forward(x, w, b):
    return x * w + b


def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


# 主程序
def main():
    # 读取数据
    x_data, y_data = load_data('train.csv')

    # 参数范围
    w_range = np.arange(0.0, 4.1, 0.1)
    b_range = np.arange(-2.0, 2.1, 0.1)

    # 存储结果
    w_list = []
    b_list = []
    mse_list = []

    # 训练循环
    for w in w_range:
        for b in b_range:
            l_sum = 0
            for x_val, y_val in zip(x_data, y_data):
                loss_val = loss(x_val, y_val, w, b)
                l_sum += loss_val

            mse = l_sum / len(x_data)
            w_list.append(w)
            b_list.append(b)
            mse_list.append(mse)

    # 找到最小损失对应的参数
    min_index = np.argmin(mse_list)
    best_w = w_list[min_index]
    best_b = b_list[min_index]
    min_mse = mse_list[min_index]

    print(f"最优参数: w = {best_w:.2f}, b = {best_b:.2f}")
    print(f"最小MSE损失: {min_mse:.4f}")

    # 转换为numpy数组便于处理
    w_arr = np.array(w_list)
    b_arr = np.array(b_list)
    mse_arr = np.array(mse_list)

    # 绘制w和loss的关系图（固定b为最优值）
    plt.figure(figsize=(12, 5))

    # 图1: w和loss的关系（固定b=best_b）
    plt.subplot(1, 2, 1)
    # 选择b接近最优值的点
    b_tolerance = 0.05
    mask_b = np.abs(b_arr - best_b) < b_tolerance
    plt.plot(w_arr[mask_b], mse_arr[mask_b], 'b-')
    plt.scatter(best_w, min_mse, color='red', s=50, zorder=5)
    plt.xlabel('权重 w')
    plt.ylabel('损失 Loss (MSE)')
    plt.title('权重 w 与损失的关系\n(偏置 b 固定为最优值)')
    plt.grid(True, alpha=0.3)

    # 图2: b和loss的关系（固定w为最优值）
    plt.subplot(1, 2, 2)
    # 选择w接近最优值的点
    w_tolerance = 0.05
    mask_w = np.abs(w_arr - best_w) < w_tolerance
    plt.plot(b_arr[mask_w], mse_arr[mask_w], 'r-')
    plt.scatter(best_b, min_mse, color='green', s=50, zorder=5)
    plt.xlabel('偏置 b')
    plt.ylabel('损失 Loss (MSE)')
    plt.title('偏置 b 与损失的关系\n(权重 w 固定为最优值)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('w_b_loss_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制最终拟合结果
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='blue', label='真实数据', alpha=0.7)

    # 使用最优参数进行预测
    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_pred = forward(x_line, best_w, best_b)
    plt.plot(x_line, y_pred, color='red', linewidth=2,
             label=f'拟合曲线: y = {best_w:.2f}x + {best_b:.2f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('线性回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('regression_fit.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存结果到CSV
    results_df = pd.DataFrame({
        'w': w_list,
        'b': b_list,
        'mse': mse_list
    })
    results_df.to_csv('training_results.csv', index=False)
    print("训练结果已保存到 training_results.csv")


if __name__ == "__main__":
    main()