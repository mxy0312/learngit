import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CountriesDataset(Dataset):
    """国家数据集"""

    def __init__(self, filepath, target_column='Total Ecological Footprint'):
        data = pd.read_csv(filepath)
        data = self.preprocess_data(data)

        feature_columns = [col for col in data.columns if col != target_column]
        x_data = data[feature_columns].values.astype(np.float32)
        y_data = data[target_column].values.astype(np.float32)

        # 标准化
        self.x_mean = x_data.mean(axis=0)
        self.x_std = x_data.std(axis=0)
        self.y_mean = y_data.mean()
        self.y_std = y_data.std()

        x_data = (x_data - self.x_mean) / (self.x_std + 1e-8)
        y_data = (y_data - self.y_mean) / (self.y_std + 1e-8)

        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)
        self.len = len(data)

    def preprocess_data(self, data):
        """数据预处理"""
        processed_data = data.copy()

        for column in processed_data.columns:
            if processed_data[column].dtype == 'object':
                # 处理货币格式
                if processed_data[column].astype(str).str.contains('\$', na=False).any():
                    temp_col = processed_data[column].astype(str).str.replace('$', '', regex=False)
                    temp_col = temp_col.str.replace(',', '', regex=False)
                    temp_col = temp_col.str.replace('"', '', regex=False)
                    processed_data[column] = pd.to_numeric(temp_col, errors='coerce')
                else:
                    # 分类编码
                    unique_vals = processed_data[column].dropna().unique()
                    val_to_num = {val: i for i, val in enumerate(unique_vals)}
                    processed_data[column] = processed_data[column].map(val_to_num)

            # 缺失值处理
            if processed_data[column].isnull().any():
                if pd.api.types.is_numeric_dtype(processed_data[column]):
                    processed_data[column] = processed_data[column].fillna(processed_data[column].median())
                else:
                    processed_data[column] = processed_data[column].fillna(0)

        return processed_data

    def inverse_transform_y(self, y_normalized):
        """反标准化"""
        return y_normalized * self.y_std + self.y_mean

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class FiveLayerNetwork(torch.nn.Module):
    """5层神经网络: 输入 -> 7 -> 6 -> 5 -> 4 -> 1"""

    def __init__(self, input_size):
        super(FiveLayerNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 7),
            torch.nn.ReLU(),
            torch.nn.Linear(7, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.layers(x)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_and_evaluate():
    """训练和评估模型"""
    print("=" * 60)
    print("国家生态足迹预测 - 5层神经网络模型")
    print("=" * 60)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 数据加载
    print("\n加载数据集...")
    dataset = CountriesDataset('countries.csv')
    print(f"✓ 数据集大小: {len(dataset)} 样本")
    print(f"✓ 特征维度: {dataset.x_data.shape[1]}")

    # 数据划分 (70% 训练, 15% 验证, 15% 测试)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 模型初始化
    print(f"\n初始化模型...")
    input_size = dataset.x_data.shape[1]
    model = FiveLayerNetwork(input_size).to(device)
    print(model)

    # 优化器配置
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    early_stopping = EarlyStopping(patience=15)

    # 训练记录
    train_losses = []
    val_losses = []

    # 训练循环
    print("\n开始训练...\n")
    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 学习率调度和早停
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch:3d}/100] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')

        if early_stopping.early_stop:
            print(f'\n早停触发！最佳验证损失: {early_stopping.best_loss:.6f}')
            model.load_state_dict(early_stopping.best_model)
            break

    print(f"\n✓ 训练完成！收敛轮数: {epoch + 1}")

    # 保存模型
    os.makedirs("best_model", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': input_size,
        'train_history': {'train_losses': train_losses, 'val_losses': val_losses}
    }, 'best_model/countries_model.pt')
    print("✓ 模型已保存: best_model/countries_model.pt")

    # 测试评估
    print("\n评估模型性能...\n")
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().squeeze().tolist() if outputs.dim() > 1 else outputs.cpu().tolist())
            actuals.extend(labels.tolist())

    # 反标准化
    predictions_original = dataset.inverse_transform_y(np.array(predictions))
    actuals_original = dataset.inverse_transform_y(np.array(actuals))

    # 计算指标
    mse = np.mean((predictions_original - actuals_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_original - actuals_original))
    r2 = 1 - (np.sum((actuals_original - predictions_original) ** 2) /
              np.sum((actuals_original - actuals_original.mean()) ** 2))

    print(f"【测试集性能指标】")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    # 可视化
    print("\n生成可视化图表...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', linewidth=2)
    axes[0].plot(val_losses, label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('损失值', fontsize=11)
    axes[0].set_title('训练和验证损失曲线', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 预测vs实际
    axes[1].scatter(actuals_original, predictions_original, alpha=0.6, s=50)
    min_val = min(actuals_original.min(), predictions_original.min())
    max_val = max(actuals_original.max(), predictions_original.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
    axes[1].set_xlabel('实际值', fontsize=11)
    axes[1].set_ylabel('预测值', fontsize=11)
    axes[1].set_title(f'预测值 vs 实际值 (R²={r2:.4f})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: training_results.png")
    plt.show()

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == '__main__':
    train_and_evaluate()