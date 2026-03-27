import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('/workspaces/PID/PID.csv', encoding='latin1')

# 提取特征和标签
features = data[['P', 'I', 'D']]
labels = data['onehot']

# 分割数据集，train:test = 4:1
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# 定义多种算法
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# 存储结果
results = {}

# 训练和评估
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # 计算 AUC
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
    else:
        # 对于不支持 predict_proba 的模型，如 SVM 默认
        auc_score = None
        fpr, tpr = None, None
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_score,
        'FPR': fpr,
        'TPR': tpr
    }

# 输出结果
print("Model Comparison Results:")
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        if metric == 'AUC' and value is not None:
            print(f"  {metric}: {value:.4f}")
        elif metric in ['FPR', 'TPR']:
            continue  # 不打印 FPR 和 TPR
        else:
            print(f"  {metric}: {value:.4f}")
    print()

# 可视化：准确率对比图
model_names = list(results.keys())
accuracies = [results[name]['Accuracy'] for name in model_names]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/PID/model_comparison.png')
plt.show()

# 可视化：AUC ROC 曲线
plt.figure(figsize=(10, 6))
for name, metrics in results.items():
    if metrics['AUC'] is not None:
        plt.plot(metrics['FPR'], metrics['TPR'], label=f'{name} (AUC = {metrics["AUC"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('/workspaces/PID/auc_comparison.png')
plt.show()