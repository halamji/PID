import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('/workspaces/PID/PIDT.csv', encoding='latin1')

# 选择四列
columns = ['P', 'I', 'D', 'T']
df = data[columns]

# 计算 Pearson 相关系数
pearson_corr = df.corr(method='pearson')

# 计算 Spearman 相关系数
spearman_corr = df.corr(method='spearman')

# 计算 Kendall 相关系数
kendall_corr = df.corr(method='kendall')

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pearson 热力图
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Pearson Correlation Heatmap')

# Spearman 热力图
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Spearman Correlation Heatmap')

# Kendall 热力图
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', ax=axes[2], vmin=-1, vmax=1)
axes[2].set_title('Kendall Correlation Heatmap')

# 调整布局
plt.tight_layout()

# 保存图
plt.savefig('/workspaces/PID/correlation_heatmaps.png')
plt.show()