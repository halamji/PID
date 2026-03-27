import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('/workspaces/PID/PIDT.csv', encoding='latin1')

# 提取特征和标签
features = data[['P', 'I', 'D']]
labels = data['T']

# 分割数据集，train:test = 4:1
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 定义多种算法
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'MLP Regressor': MLPRegressor(max_iter=1000, random_state=42)
}

# 存储结果
results = {}

# 训练和评估
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'y_pred': y_pred
    }

# 输出结果
print("Model Comparison Results:")
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        if metric != 'y_pred':
            print(f"  {metric}: {value:.4f}")
    print()

# 可视化：模型指标对比
model_names = list(results.keys())
mse_values = [results[name]['MSE'] for name in model_names]
mae_values = [results[name]['MAE'] for name in model_names]
r2_values = [results[name]['R2'] for name in model_names]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].bar(model_names, mse_values, color='skyblue')
axes[0].set_title('MSE Comparison')
axes[0].set_ylabel('MSE')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(model_names, mae_values, color='lightgreen')
axes[1].set_title('MAE Comparison')
axes[1].set_ylabel('MAE')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(model_names, r2_values, color='salmon')
axes[2].set_title('R2 Comparison')
axes[2].set_ylabel('R2')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/workspaces/PID/model_metrics_comparison.png')
plt.show()

# 可视化：预测值 vs 实际值
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, name in enumerate(model_names):
    y_pred = results[name]['y_pred']
    axes[i].scatter(y_test, y_pred, alpha=0.6)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[i].set_xlabel('Actual T')
    axes[i].set_ylabel('Predicted T')
    axes[i].set_title(f'{name}: Actual vs Predicted')

plt.tight_layout()
plt.savefig('/workspaces/PID/prediction_vs_actual.png')
plt.show()