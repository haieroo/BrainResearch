import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================= 1. 图表美化设置 =================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['Arial', 'Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

base_path = r"F:\peng\picture1"
img_dir = os.path.join(base_path, 'bird picture1')
mask_dir = os.path.join(base_path, 'masks')


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)


a_means, b_means, true_colors, true_shapes = [], [], [], []

color_palette = {
    'red': '#D62728',
    'green': '#2CA02C',
    'blue': '#1F77B4',
    'yellow': '#FF7F0E'
}

# ================= 2. 显著性色彩特征提取 =================
print("正在提取 LAB 空间显著性色彩特征...")
for filename in os.listdir(img_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    img = cv_imread(img_path, cv2.IMREAD_COLOR)
    mask = cv_imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    # 提取颜色标签
    c_label = 'gray'
    if 'red' in filename.lower():
        c_label = 'red'
    elif 'green' in filename.lower():
        c_label = 'green'
    elif 'blue' in filename.lower():
        c_label = 'blue'
    elif 'yellow' in filename.lower():
        c_label = 'yellow'

    # 提取形状标签
    s_label = 'o'
    if 's1' in filename.lower():
        s_label = 'o'
    elif 's2' in filename.lower():
        s_label = 's'
    elif 's3' in filename.lower():
        s_label = '^'

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)

    pixels_lab = img_lab[mask > 128]

    if len(pixels_lab) > 0:
        a_vals = pixels_lab[:, 1]
        b_vals = pixels_lab[:, 2]

        chroma = np.sqrt(np.square(a_vals) + np.square(b_vals))
        threshold = np.percentile(chroma, 70)
        colorful_pixels = pixels_lab[chroma >= threshold]

        a_means.append(np.mean(colorful_pixels[:, 1]))
        b_means.append(np.mean(colorful_pixels[:, 2]))
        true_colors.append(c_label)
        true_shapes.append(s_label)

# ================= 3. 高颜值可视化绘制 =================
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid(False)

limit = max(max(np.abs(a_means)), max(np.abs(b_means))) + 15


ax.axvspan(0, limit, ymin=0.5, ymax=1, color='#FFD700', alpha=0.03)
ax.axvspan(-limit, 0, ymin=0.5, ymax=1, color='#90EE90', alpha=0.03)
ax.axvspan(-limit, 0, ymin=0, ymax=0.5, color='#87CEFA', alpha=0.03)
ax.axvspan(0, limit, ymin=0, ymax=0.5, color='#FFB6C1', alpha=0.03)

# 十字交叉基准线
ax.axhline(0, color='#555555', linestyle='--', linewidth=1.2, zorder=1)
ax.axvline(0, color='#555555', linestyle='--', linewidth=1.2, zorder=1)


for i in range(len(a_means)):
    ax.scatter(a_means[i], b_means[i],
               c=color_palette.get(true_colors[i], 'gray'),
               marker=true_shapes[i], s=900,
               edgecolors='white', linewidths=1.5,
               zorder=3, alpha=0.9)

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_xlabel('a* (Green-red axis)', fontsize=20, fontweight='bold', labelpad=20)
ax.set_ylabel('b* (Blue-yellow axis)', fontsize=20, fontweight='bold', labelpad=20)

ax.set_title('Color distribution', fontsize=24, fontweight='bold', pad=20)

for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color('#222222')

# ================= 4. 构建顶刊级别的复合图例 =================
legend_elements = [
  
    Line2D([0], [0], marker='o', color='w', label='Red', markerfacecolor='#D62728', markersize=18),
    Line2D([0], [0], marker='o', color='w', label='Green', markerfacecolor='#2CA02C', markersize=18),
    Line2D([0], [0], marker='o', color='w', label='Blue', markerfacecolor='#1F77B4', markersize=18),
    Line2D([0], [0], marker='o', color='w', label='Yellow', markerfacecolor='#FF7F0E', markersize=18),

    # 占位符，用来在图例里空出一行
    Line2D([0], [0], color='w', label=' '),

    Line2D([0], [0], marker='o', color='w', label='Shape 1', markerfacecolor='#888888', markersize=18),
    Line2D([0], [0], marker='s', color='w', label='Shape 2', markerfacecolor='#888888', markersize=18),
    Line2D([0], [0], marker='^', color='w', label='Shape 3', markerfacecolor='#888888', markersize=18)
]

ax.legend(handles=legend_elements, loc='best', fontsize=11, frameon=True, shadow=True, borderpad=1.2)

plt.tight_layout()
