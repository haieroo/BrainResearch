import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.preprocessing import StandardScaler
import umap
from adjustText import adjust_text

# ================= 1. 图表美化设置 =================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['Arial', 'Times New Roman', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

base_path = r"F:\peng\picture1"
mask_dir = os.path.join(base_path, 'masks')

def cv_imread(file_path, flags=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)

features = []
image_names, true_shapes, shape_markers, original_colors = [], [], [], []
color_map = {'red': '#D62728', 'green': '#2CA02C', 'blue': '#1F77B4', 'yellow': '#FF7F0E'}

# 初始化 HOG 描述符 (适合 64x64 的标准网格)
hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

# ================= 2. 基于 HOG 的全局形状特征提取 =================
print("正在提取掩码的 HOG 方向梯度特征...")
valid_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

for filename in valid_files:
    mask_path = os.path.join(mask_dir, filename)
    mask = cv_imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: continue

    fname_lower = filename.lower()
    if 'bird' in fname_lower or 's1' in fname_lower:
        s_label, marker = 'Shape 1', 'o'
    elif 'duck' in fname_lower or 's2' in fname_lower:
        s_label, marker = 'Shape 2', 's'
    elif 'ostrich' in fname_lower or 's3' in fname_lower:
        s_label, marker = 'Shape 3', '^'
    else:
        continue

    c_label = 'gray'
    if 'red' in fname_lower: c_label = 'red'
    elif 'green' in fname_lower: c_label = 'green'
    elif 'blue' in fname_lower: c_label = 'blue'
    elif 'yellow' in fname_lower: c_label = 'yellow'

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 提取鸟的边界框，裁剪出主体，消除图片中鸟的位置偏移干扰
    x, y, w, h = cv2.boundingRect(binary_mask)
    if w >= 8 and h >= 8:
        cropped_mask = binary_mask[y:y + h, x:x + w]
    else:
        cropped_mask = binary_mask

    # 统一缩放为 64x64 的标准网格
    resized_mask = cv2.resize(cropped_mask, (64, 64))

    # 计算 HOG 特征
    hog_features = hog.compute(resized_mask).flatten()

    features.append(hog_features)
    image_names.append(filename)
    true_shapes.append(s_label)
    shape_markers.append(marker)
    original_colors.append(c_label)

features = np.array(features)

# ================= 3. 标准化与单一 UMAP 降维 =================
print("正在执行 UMAP 非线性流形降维...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 精调 UMAP 参数
reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42, metric='euclidean')
umap_2d = reducer.fit_transform(features_scaled)

# ================= 4. 高颜值单一面板可视化 =================

fig, ax = plt.subplots(figsize=(12, 10))
ax.grid(False)

jitter_map = {
    'red': (1, 1),     # 右上
    'blue': (-1, -1),  # 左下
    'green': (-1, 1),  # 左上
    'yellow': (1, -1), # 右下
    'gray': (0, 0)
}

# 动态计算扰动距离 
x_range = umap_2d[:, 0].max() - umap_2d[:, 0].min()
y_range = umap_2d[:, 1].max() - umap_2d[:, 1].min()
jitter_scale_x = max(x_range * 0.055, 0.015)
jitter_scale_y = max(y_range * 0.055, 0.015)

texts = []
for i in range(len(umap_2d)):
    color_val = color_map.get(original_colors[i], '#555555')
    dx, dy = jitter_map.get(original_colors[i], (0, 0))

    # 施加空间扰动
    plot_x = umap_2d[i, 0] + dx * jitter_scale_x
    plot_y = umap_2d[i, 1] + dy * jitter_scale_y

    # 散点大小 s 提升到 700，边框 linewidth 提升到 2.5
    ax.scatter(plot_x, plot_y, c=color_val, marker=shape_markers[i], s=1000,
               edgecolors='white', linewidths=2.5, zorder=3, alpha=0.9)

    # 提取精简标签，仅标注红色的点
    if original_colors[i] == 'red':
        short_name = image_names[i].replace('.png', '').replace('red', '').replace('blue', '').replace('green', '').replace('yellow', '').strip('_')
        short_name = short_name[:5] if len(short_name) >= 5 else short_name

        txt = ax.text(plot_x, plot_y, short_name, fontsize=18, fontweight='bold', color='#222222',
                      path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=4)
        texts.append(txt)

# 自动防重叠排版
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=1.5, alpha=0.7), expand_points=(1.5, 1.5))

# 留白设置
ax.set_xlim(umap_2d[:, 0].min() - x_range * 0.2, umap_2d[:, 0].max() + x_range * 0.2)
ax.set_ylim(umap_2d[:, 1].min() - y_range * 0.2, umap_2d[:, 1].max() + y_range * 0.2)

# SCI 规范坐标轴与标题
ax.set_xlabel('Dimension 1', fontsize=24, fontweight='bold', labelpad=20)
ax.set_ylabel('Dimension 2', fontsize=24, fontweight='bold', labelpad=20)
ax.set_title('Shape clustering', fontsize=28, fontweight='bold', pad=20)

# 坐标轴边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(2.0)
    spine.set_color('#222222')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)

# 构建高级图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Shape 1', markerfacecolor='#888888', markersize=20),
    Line2D([0], [0], marker='s', color='w', label='Shape 2', markerfacecolor='#888888', markersize=20),
    Line2D([0], [0], marker='^', color='w', label='Shape 3', markerfacecolor='#888888', markersize=20)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=15, frameon=True, shadow=True, borderpad=1.0)

plt.tight_layout()

