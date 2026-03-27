import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取预测结果
data = pd.read_csv('/workspaces/PID/pid_predicted.csv')

P = data['P']
I = data['I']
D = data['D']
T = data['T']
onehot = data['onehot']

# 3D 散点图: P/I/D，颜色由 T 控制，点形由 onehot 控制
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 颜色和大小映射
norm = plt.Normalize(T.min(), T.max())
colors = plt.cm.viridis(norm(T))

marker_map = {0: 'o', 1: '^'}
for cls in sorted(data['onehot'].unique()):
    mask = data['onehot'] == cls
    ax.scatter(P[mask], I[mask], D[mask],
               c=colors[mask],
               marker=marker_map[cls],
               edgecolor='k',
               s=20,
               alpha=0.7,
               label=f'onehot={cls}')

# 添加颜色条
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array([])
cb = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=20)
cb.set_label('Predicted T')

ax.set_xlabel('P')
ax.set_ylabel('I')
ax.set_zlabel('D')
ax.set_title('4D Visualization of pid_predicted.csv (P,I,D + T color, onehot marker)')
ax.legend()

plt.tight_layout()
plt.savefig('/workspaces/PID/pid_predicted_4d.png', dpi=300)
plt.show()
