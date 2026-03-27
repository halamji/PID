import numpy as np
import pandas as pd

# 定义范围
p_range = [0.3, 3]
i_range = [12, 35]
d_range = [10, 20]

# 采样点数，每个维度
n_p = 50  # P
n_i = 50  # I
n_d = 40  # D, 50*50*40=100,000

# 生成均匀采样
p_values = np.linspace(p_range[0], p_range[1], n_p)
i_values = np.linspace(i_range[0], i_range[1], n_i)
d_values = np.linspace(d_range[0], d_range[1], n_d)

# 创建网格
P, I, D = np.meshgrid(p_values, i_values, d_values, indexing='ij')

# 展平
p_flat = P.flatten()
i_flat = I.flatten()
d_flat = D.flatten()

# 创建 DataFrame
data = pd.DataFrame({
    'P': p_flat,
    'I': i_flat,
    'D': d_flat
})

# 保存到 CSV
data.to_csv('/workspaces/PID/pid_sampled.csv', index=False)

print(f"采样完成，共 {len(data)} 条记录，保存至 /workspaces/PID/pid_sampled.csv")