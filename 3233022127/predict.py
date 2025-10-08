import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def load_best_model():
    """加载最佳模型"""
    checkpoint = torch.load("best_model/best_model.pth", weights_only=False)
    model = LinearModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"加载模型: {checkpoint['model_name']}")
    print(f"最佳损失: {checkpoint['best_loss']:.6f}")

    return model, checkpoint


def predict(x_values):
    """预测函数"""
    model, checkpoint = load_best_model()

    if not isinstance(x_values, list):
        x_values = [x_values]

    x_tensor = torch.FloatTensor(x_values).reshape(-1, 1)
    x_norm = (x_tensor - checkpoint['x_mean']) / checkpoint['x_std']

    with torch.no_grad():
        y_norm = model(x_norm)

    y_pred = y_norm * checkpoint['y_std'] + checkpoint['y_mean']
    return y_pred.numpy().flatten()


if __name__ == "__main__":
    # 测试预测
    test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    predictions = predict(test_values)

    print("\n=== 预测结果 ===")
    for x, y in zip(test_values, predictions):
        print(f"x = {x}, y_pred = {y:.4f}")