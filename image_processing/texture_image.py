import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.preprocessing import StandardScaler
import umap
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from adjustText import adjust_text

# ================= 1. 图表美化设置 =================
# 采用纯净学术白底
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['Arial', 'Times New Roman', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

base_path = r"F:\peng\picture1"
img_dir = os.path.join(base_path, 'bird picture1')
mask_dir = os.path.join(base_path, 'masks')

def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)

texture_colors = {'T1': '#E63946', 'T2': '#457B9D'}
shape_markers = {'Shape 1': 'o', 'Shape 2': 's', 'Shape 3': '^'}

# ================= 2. 参数初始化 =================
n_points, radius = 16, 2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 核心腐蚀手术刀

feat_list_lbp, feat_list_glcm = [], []
image_names, true_shapes, true_textures = [], [], []

# ================= 3. 核心腐蚀 + LBP/GLCM 特征提取 =================
print("正在执行抗轮廓干扰的『核心腐蚀 + LBP微观 + GLCM宏观』特征提取...")
valid_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

for filename in valid_files:
    img = cv_imread(os.path.join(img_dir, filename), cv2.IMREAD_COLOR)
    mask = cv_imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None: continue

    fname_upper = filename.upper()
    if 'S1' in fname_upper: s_label = 'Shape 1'
    elif 'S2' in fname_upper: s_label = 'Shape 2'
    elif 'S3' in fname_upper: s_label = 'Shape 3'
    else: continue

    if 'T1' in fname_upper: t_label = 'T1'
    elif 'T2' in fname_upper: t_label = 'T2'
    else: continue

    # 提取 L* 通道
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = clahe.apply(lab_img[:, :, 0])

    # 核心手术：形态学腐蚀 (剥除外边缘干扰)
    pure_core_mask = cv2.erode(binary_mask, morph_kernel, iterations=2)
    if cv2.countNonZero(pure_core_mask) < 200:
        pure_core_mask = cv2.erode(binary_mask, morph_kernel, iterations=1)
        if cv2.countNonZero(pure_core_mask) < 200:
            pure_core_mask = binary_mask

    # --- 特征 1: LBP ---
    lbp = local_binary_pattern(l_channel, n_points, radius, method='uniform')
    masked_lbp = lbp[pure_core_mask > 0]
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(masked_lbp, bins=n_bins, range=(0, n_bins), density=True)
    feat_list_lbp.append(lbp_hist)

    # --- 特征 2: GLCM ---
    x, y, w, h = cv2.boundingRect(pure_core_mask)
    roi = l_channel[y:y + h, x:x + w] if (w >= 2 and h >= 2) else l_channel

    glcm = graycomatrix(roi, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
                        symmetric=True, normed=True)
    glcm_feats = np.concatenate([
        graycoprops(glcm, 'contrast').flatten(),
        graycoprops(glcm, 'homogeneity').flatten(),
        graycoprops(glcm, 'energy').flatten(),
        graycoprops(glcm, 'correlation').flatten()
    ])
    feat_list_glcm.append(glcm_feats)

    image_names.append(filename)
    true_shapes.append(s_label)
    true_textures.append(t_label)

# ================= 4. 标准化与特征融合 =================
scaler_lbp = StandardScaler()
scaler_glcm = StandardScaler()

norm_lbp = scaler_lbp.fit_transform(feat_list_lbp)
norm_glcm = scaler_glcm.fit_transform(feat_list_glcm)
final_features = np.hstack([norm_lbp, norm_glcm])

# ================= 5. 降维 =================
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42, metric='euclidean')
umap_2d = reducer.fit_transform(final_features)

# ================= 6. 高颜值单一面板可视化 =================
fig, ax = plt.subplots(figsize=(12, 10))
ax.grid(False)

x_range = umap_2d[:, 0].max() - umap_2d[:, 0].min()
y_range = umap_2d[:, 1].max() - umap_2d[:, 1].min()

texts = []
np.random.seed(42) 

for i in range(len(umap_2d)):
   
    jx = np.random.normal(0, x_range * 0.02)
    jy = np.random.normal(0, y_range * 0.02)
    plot_x, plot_y = umap_2d[i, 0] + jx, umap_2d[i, 1] + jy

    # 【散点大小 s 提升到 700，边框 linewidth 提升到 2.5
    ax.scatter(plot_x, plot_y, c=texture_colors[true_textures[i]],
               marker=shape_markers[true_shapes[i]], s=1000,
               edgecolors='white', linewidths=2.5, zorder=3, alpha=0.85)

    short_name = f"{true_shapes[i][:2]}_{true_textures[i]}"
    # 每 4 个点标注一次（即每种形状+纹理组合只标一个），避免画面太乱
    if i % 4 == 0:
        txt = ax.text(plot_x, plot_y, short_name, fontsize=18, fontweight='bold', color='#222222',
                      path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=4)
        texts.append(txt)

adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=1.5, alpha=0.7), expand_points=(1.5, 1.5))

# 留白设置
ax.set_xlim(umap_2d[:, 0].min() - x_range * 0.2, umap_2d[:, 0].max() + x_range * 0.2)
ax.set_ylim(umap_2d[:, 1].min() - y_range * 0.2, umap_2d[:, 1].max() + y_range * 0.2)

# SCI 规范坐标轴与标题
ax.set_xlabel('Dimension 1', fontsize=24, fontweight='bold', labelpad=20)
ax.set_ylabel('Dimension 2', fontsize=24, fontweight='bold', labelpad=20)
ax.set_title('Texture clustering', fontsize=28, fontweight='bold', pad=20)

# 坐标轴边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(2.0)
    spine.set_color('#222222')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)

# 构建高级图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Texture 1 (T1)', markerfacecolor='#E63946', markersize=20),
    Line2D([0], [0], marker='o', color='w', label='Texture 2 (T2)', markerfacecolor='#457B9D', markersize=20),
    Line2D([0], [0], color='w', label=' '), # 空行分隔
    Line2D([0], [0], marker='o', color='w', label='Shape 1', markerfacecolor='gray', markersize=20),
    Line2D([0], [0], marker='s', color='w', label='Shape 2', markerfacecolor='gray', markersize=20),
    Line2D([0], [0], marker='^', color='w', label='Shape 3', markerfacecolor='gray', markersize=20)
]
ax.legend(handles=legend_elements, loc='best', fontsize=14, frameon=True, shadow=True, borderpad=1.0)

plt.tight_layout()
