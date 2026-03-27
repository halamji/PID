import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
data = pd.read_csv('/workspaces/PID/PIDT.csv', encoding='latin1')

# 提取列
P = data['P']
I = data['I']
D = data['D']
T = data['T']

# 创建 3D 图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图，颜色根据 T
scatter = ax.scatter(P, I, D, c=T, cmap='viridis', s=50)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('T Value')

# 设置标签
ax.set_xlabel('P')
ax.set_ylabel('I')
ax.set_zlabel('D')
ax.set_title('3D Visualization of P, I, D with T as Color')

# 保存图
plt.savefig('/workspaces/PID/pid_3d_visualization.png')
plt.show()