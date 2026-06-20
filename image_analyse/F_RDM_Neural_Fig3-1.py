import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.io import savemat  # <--- 新增：用于保存为 .mat 文件
import scipy.io as sio
import os

# ================= 1. 全局图表高级设置 =================
# 启用 Seaborn 干净清爽的白底风格
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 载入 MATLAB 神经信号数据 =================
print("正在载入神经电生理数据...")
data_path = 'acc.mat'  # 请确保路径和文件名正确
data_pathlabel = 'Model_RDMs.mat'  # 请确保路径和文件名正确
try:
    mat_labels = sio.loadmat(data_pathlabel)
    short_labels = mat_labels['stimulus_labels']
    mat_data = sio.loadmat(data_path)
    X_ENTO = mat_data['X_ENTO']
    X_MVL = mat_data['X_MVL']
    print(f"✅ 成功载入数据！X_ENTO 维度: {X_ENTO.shape}, X_MVL 维度: {X_MVL.shape}")
except FileNotFoundError:
    print(f"❌ 找不到文件 {data_path}，请确保已从 MATLAB 导出该文件！")

# ================= 3. 计算 RDM (基于 1 - Pearson r) =================   cosine
print("正在计算 ENTO 和 MVL 的表征相异性矩阵 (RDM)...")
# metric='correlation' 在 scipy 中直接计算的就是 1 - Pearson相关系数 correlation
rdm_ento = squareform(pdist(X_ENTO, metric='correlation'))
rdm_mvl = squareform(pdist(X_MVL, metric='correlation'))

# ================= 4. 绘制顶刊级双面板对比图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
titles = ['ENTO RDM', 'MVL RDM']
rdms = [rdm_ento, rdm_mvl]

# # 自定义坐标轴的刻度标签位置 (由于我们有24个刺激，分3组，每组8个)
# tick_positions = [3.5, 11.5, 19.5]  # 分别对应第4, 12, 20个格子的中心
# tick_labels = ['Shape 1', 'Shape 2', 'Shape 3']

for i in range(2):
    # 使用 seaborn 画热力图，颜色越深(紫色)代表距离越近，越亮(黄色)代表差异越大
    sns.heatmap(rdms[i], cmap='viridis', square=True,
                # vmin=0, vmax=1.4,  # 锁定颜色范围以实现绝对公平的对比
                xticklabels=short_labels, yticklabels=short_labels,
                ax=axes[i])

    axes[i].set_title(titles[i], fontsize=16, fontweight='bold', pad=15)
    axes[i].tick_params(axis='x', rotation=90, labelsize=9)
    axes[i].tick_params(axis='y', rotation=0, labelsize=9)

    # 添加极其细微的白色网格线，划分每一个刺激格，提升顶刊图表的高级感
    axes[i].hlines(np.arange(1, len(short_labels)), *axes[i].get_xlim(), color='white', linewidth=0.5, alpha=0.5)
    axes[i].vlines(np.arange(1, len(short_labels)), *axes[i].get_ylim(), color='white', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.tight_layout()
save_path = "Neural_RDMs_Python.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 神经信号 RDM 绘制完成，图片已保存至 {save_path}！")

# ================= 5. 保存 RDM 为 .mat 文件供 MATLAB 使用 =================
mat_save_path = "Neural_RDMs.mat"

# 构建要保存的数据字典
Neural_data = {
    'rdm_ento': rdm_ento,
    'rdm_mvl': rdm_mvl
}

# 写入 .mat 文件
savemat(mat_save_path, Neural_data)
print(f"✅ 三大特征矩阵已成功打包存入 MATLAB 格式文件: {mat_save_path}")
plt.show()