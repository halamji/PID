import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# 读取采样数据
sampled_data = pd.read_csv('/workspaces/PID/pid_sampled.csv')

# 读取原始数据用于训练
# 分类数据：PID.csv
classification_data = pd.read_csv('/workspaces/PID/PID.csv', encoding='latin1')
X_class = classification_data[['P', 'I', 'D']]
y_class = classification_data['onehot']

# 回归数据：PIDT.csv
regression_data = pd.read_csv('/workspaces/PID/PIDT.csv', encoding='latin1')
X_reg = regression_data[['P', 'I', 'D']]
y_reg = regression_data['T']

# 训练分类模型（使用 Random Forest 作为效果最好的）
clf = RandomForestClassifier(random_state=42)
clf.fit(X_class, y_class)

# 训练回归模型（使用 Random Forest 作为效果最好的）
reg = RandomForestRegressor(random_state=42)
reg.fit(X_reg, y_reg)

# 对采样数据进行预测
X_sample = sampled_data[['P', 'I', 'D']]

# 预测 onehot
predicted_onehot = clf.predict(X_sample)

# 预测 T
predicted_T = reg.predict(X_sample)

# 创建新 DataFrame
predicted_data = pd.DataFrame({
    'P': sampled_data['P'],
    'I': sampled_data['I'],
    'D': sampled_data['D'],
    'T': predicted_T,
    'onehot': predicted_onehot
})

# 保存到新 CSV
predicted_data.to_csv('/workspaces/PID/pid_predicted.csv', index=False)

print("推理完成，结果保存至 /workspaces/PID/pid_predicted.csv")
print(f"共 {len(predicted_data)} 条记录")