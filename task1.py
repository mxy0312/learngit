import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib as mpl

def setup_plotting():
    mpl.rcParams.update(mpl.rcParamsDefault)

    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['mathtext.fontset'] = 'stixsans'
    plt.rcParams['mathtext.default'] = 'regular'

    plt.rcParams['figure.figsize'] = (12, 8)

setup_plotting()

torch.manual_seed(42)
np.random.seed(42)

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return None, None, None, None

    try:
        data = pd.read_csv(csv_path)
        print(f"Data loaded successfully! Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nFirst 5 rows:\n{data.head()}")

        missing_values = data.isnull().sum()
        print(f"\nMissing values:\n{missing_values}")

        initial_shape = data.shape[0]
        data = data.dropna()
        final_shape = data.shape[0]
        print(f"Removed {initial_shape - final_shape} rows with missing values")

        if data.shape[0] == 0:
            print("Error: No data after removing missing values!")
            return None, None, None, None

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        return X_scaled, y_scaled, scaler_X, scaler_y

    except Exception as e:
        print(f"Data loading error: {e}")
        return None, None, None, None

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

        print(f"Model initialized: input_dim={input_dim}")

    def forward(self, x):
        return self.linear(x)

def train_model(optimizer_class, optimizer_name, X_train, y_train, X_val, y_val,
                lr=0.01, epochs=1000):
    input_dim = X_train.shape[1]
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()

    if optimizer_name == 'SGD' or 'SGD_lr' in optimizer_name:
        optimizer = optimizer_class(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optimizer_class(model.parameters(), lr=lr, betas=(0.9, 0.999))
    elif optimizer_name == 'RMSprop':
        optimizer = optimizer_class(model.parameters(), lr=lr, alpha=0.99)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    weights_history = []
    biases_history = []

    print(f"Training {optimizer_name}, lr: {lr}, epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        weights_history.append(model.linear.weight.data.numpy().copy())
        biases_history.append(model.linear.bias.data.numpy().copy())

        if epoch % 200 == 0:
            print(
                f'{optimizer_name} - Epoch {epoch:4d}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

    print(f'{optimizer_name} training completed, final loss: {val_losses[-1]:.6f}')
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'weights_history': weights_history,
        'biases_history': biases_history,
        'name': optimizer_name
    }

def plot_optimizer_comparison(results):
    if not results:
        print("No results to visualize")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for result in results:
        ax1.plot(result['train_losses'], label=result['name'], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    for result in results:
        ax2.plot(result['val_losses'], label=result['name'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    try:
        plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    except Exception as e:
        print(f"Save warning: {e}")
        plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_parameters_evolution(results):
    if not results:
        print("No results to visualize")
        return

    num_features = results[0]['weights_history'][0].shape[1]
    fig, axes = plt.subplots(2, len(results), figsize=(15, 10))

    if len(results) == 1:
        axes = axes.reshape(2, 1)

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for idx, result in enumerate(results):
        weights = np.array(result['weights_history'])
        biases = np.array(result['biases_history'])

        for i in range(min(num_features, len(colors))):
            axes[0, idx].plot(weights[:, 0, i], color=colors[i], label=f'Weight {i + 1}', linewidth=2)
        axes[0, idx].set_title(f'{result["name"]} - Weights')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Weight Value')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        axes[1, idx].plot(biases, color='black', linewidth=2, label='Bias')
        axes[1, idx].set_title(f'{result["name"]} - Bias')
        axes[1, idx].set_xlabel('Epoch')
        axes[1, idx].set_ylabel('Bias Value')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()

    try:
        plt.savefig('parameters_evolution.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    except Exception as e:
        print(f"Save warning: {e}")
        plt.savefig('parameters_evolution.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    learning_rates = [0.001, 0.01, 0.1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for lr in learning_rates:
        result = train_model(torch.optim.SGD, 'SGD', X_train, y_train, X_val, y_val,
                             lr=lr, epochs=800)
        ax1.plot(result['val_losses'][:150], label=f'LR = {lr}', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Learning Rate Tuning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    epochs_list = [500, 1000, 1500]
    for epochs in epochs_list:
        result = train_model(torch.optim.SGD, 'SGD', X_train, y_train, X_val, y_val,
                             lr=0.01, epochs=epochs)
        ax2.plot(result['val_losses'], label=f'Epochs = {epochs}', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Training Epochs Tuning')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    try:
        plt.savefig('hyperparameter_tuning.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    except Exception as e:
        print(f"Save warning: {e}")
        plt.savefig('hyperparameter_tuning.png', dpi=300, bbox_inches='tight')

    plt.show()

def main():
    print("=== Linear Regression Model Training ===\n")

    X, y, scaler_X, scaler_y = load_data('train.csv')

    if X is None:
        print("Data loading failed, exiting...")
        return

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    print(f"Tensor shapes: X {X_tensor.shape}, y {y_tensor.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    print(f"Data split: train {X_train.shape}, validation {X_val.shape}")

    optimizers = [
        (torch.optim.SGD, 'SGD'),
        (torch.optim.Adam, 'Adam'),
        (torch.optim.RMSprop, 'RMSprop')
    ]

    results = []
    for optimizer_class, optimizer_name in optimizers:
        print(f"\n{'=' * 50}")
        print(f"Training {optimizer_name} optimizer...")
        result = train_model(optimizer_class, optimizer_name, X_train, y_train,
                             X_val, y_val, lr=0.01, epochs=800)  # 减少轮次以加快速度
        results.append(result)

    plot_optimizer_comparison(results)
    plot_parameters_evolution(results)
    plot_hyperparameter_tuning(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()