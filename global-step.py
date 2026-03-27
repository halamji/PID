import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from itertools import combinations

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取预测结果文件
"""
请评委阅读：
下面csv_file地址可自定义，但请注意，如果文件名中包含中文则可能出现报错的情况，若出现报错请将地址替换为英文，并将相对地址替换为绝对地址（路径中没有中文）。
"""
try:
    data = pd.read_csv('/workspaces/PID/PIDT.csv', encoding='latin1')
    print(f"成功加载数据，共 {len(data)} 条记录")
except FileNotFoundError:
    print("找不到/workspaces/PID/PIDT.csv文件")
    exit()
"""下面代码为分别绘制每个4D表征图的代码"""
# 定义特征和目标变量
features = ['P', 'I', 'D', 'T']
target = 'onehot'

# 根据目标变量onehot值分箱，用于颜色映射
dms_bins = data[target]  # 直接使用 onehot
colors = ['red' if x == 1 else 'blue' for x in dms_bins]

# 生成所有可能的三个特征的组合
feature_combinations = list(combinations(features, 3))

# 为每个组合创建一个3D图，用颜色表示第四个特征
for i, combo in enumerate(feature_combinations):
    # 创建画布
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取当前组合的三个特征
    x_feature, y_feature, z_feature = combo
    
    # 确定未被选中的特征作为颜色映射
    color_features = [f for f in features if f not in combo]
    
    # 绘制散点图
    scatter = ax.scatter(data[x_feature], data[y_feature], data[z_feature], 
                        c=colors, s=100, alpha=0.8, edgecolors='w')
    
    # 添加颜色条说明
    # 对于离散颜色，手动添加图例
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='onehot=1')
    blue_patch = mpatches.Patch(color='blue', label='onehot=0')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_zlabel(z_feature)
    ax.set_title(f'4D : {x_feature}, {y_feature}, {z_feature}, {target}')
    
    plt.tight_layout()
    
    # 保存图片
    filename = f'4D_visualization{x_feature}_{y_feature}_{z_feature}_{target}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"已保存图片: {filename}")
    
    # 关闭当前画布以节省内存
    plt.close(fig)

print(f"已生成所有四维表征图，共 {len(feature_combinations)} 张")
"""下面代码为在同一个画布中绘制所有4D表征图的代码"""

# 定义特征和目标变量
features = ['P', 'I', 'D', 'T']
target = 'onehot'

# 根据目标变量onehot值分箱，用于颜色映射
dms_bins = data[target]  # 直接使用 onehot
colors = ['red' if x == 1 else 'blue' for x in dms_bins]

# 生成所有可能的三个特征的组合
feature_combinations = list(combinations(features, 3))

# 创建一个大画布
fig = plt.figure(figsize=(24, 20))

# 为每个组合创建一个子图
for i, combo in enumerate(feature_combinations):
    # 计算子图位置
    row = i // 3
    col = i % 3
    
    # 添加3D子图
    ax = fig.add_subplot(4, 3, i+1, projection='3d')
    
    # 获取当前组合的三个特征
    x_feature, y_feature, z_feature = combo
    
    # 绘制散点图
    scatter = ax.scatter(data[x_feature], data[y_feature], data[z_feature], 
                        c=colors, s=50, alpha=0.8, edgecolors='w')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_zlabel(z_feature)
    ax.set_title(f'{x_feature}, {y_feature}, {z_feature}')
    
    # 设置视角
    ax.view_init(elev=30, azim=45)

# 添加一个公共的颜色条
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='rainbow'), cax=cbar_ax)
cbar.set_label(f'{target}value')
cbar.set_ticks([0.15, 0.85])
cbar.set_ticklabels([f"low ({data[target].min():.2f})", 
                     f"high ({data[target].max():.2f})"])

# 添加主标题
plt.suptitle(f'4D visualization (color shows{target})', fontsize=16, y=0.95)

# 调整布局
plt.tight_layout(rect=[0, 0, 0.9, 0.9])  # 为颜色条留出空间
plt.savefig('all_4d_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"已生成包含所有四维表征图的组合图片")