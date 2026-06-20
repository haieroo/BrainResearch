import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.io import savemat  # <--- 新增：用于保存为 .mat 文件
import os

# ================= 导入你放在 tools_feature 文件夹里的子程序 =================
from tools_feature.color_feature import get_color_features
from tools_feature.shape_feature import get_shape_features
from tools_feature.texture_feature import get_texture_features

# ================= 1. 图表设置 =================
plt.style.use('seaborn-v0_8-white')
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 替换为你自己电脑上的真实路径
base_path = r"F:\peng\picture1"

# ================= 2. 依次调用子程序获取物理特征 =================
print("正在调用 tools_feature 提取颜色特征 (LAB空间)...")
color_features, color_names = get_color_features(base_path)

print("正在调用 tools_feature 提取形状特征 (全局HOG)...")
shape_features, shape_names = get_shape_features(base_path)

print("正在调用 tools_feature 提取纹理特征 (核心腐蚀 LBP+GLCM)...")
texture_features, texture_names = get_texture_features(base_path)

# 【极其严厉的安全校验】：一旦文件顺序不对齐，整个 RSA 结果将完全作废！
assert color_names == shape_names == texture_names, "❌ 严重错误：子程序输出的刺激文件顺序不一致！请检查 sorted 逻辑。"

# ================= 3. 计算计算机视觉模型的 RDM (相异度矩阵) =================
print("正在计算 Model RDMs...")
rdm_color = squareform(pdist(color_features, metric='euclidean'))
rdm_shape = squareform(pdist(shape_features, metric='euclidean'))
rdm_texture = squareform(pdist(texture_features, metric='euclidean'))

# ================= 4. 高颜值 RDM 可视化 =================  
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
titles = ['Model RDM: Color',
          'Model RDM: Shape',
          'Model RDM: Texture']
rdms = [rdm_color, rdm_shape, rdm_texture]


short_labels = [name.replace('.png', '').replace('.jpg', '') for name in color_names]

for i in range(3):
   
    sns.heatmap(rdms[i], cmap='viridis', square=True,
                xticklabels=short_labels, yticklabels=short_labels,
                cbar_kws={"shrink": .75}, ax=axes[i])

    axes[i].hlines(np.arange(1, len(short_labels)), *axes[i].get_xlim(), color='white', linewidth=0.5, alpha=0.5)
    axes[i].vlines(np.arange(1, len(short_labels)), *axes[i].get_ylim(), color='white', linewidth=0.5, alpha=0.5)

plt.tight_layout()


plt.show()