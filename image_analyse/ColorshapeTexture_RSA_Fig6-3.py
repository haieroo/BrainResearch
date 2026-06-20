import os

os.environ['TORCH_HOME'] = r'F:\soft\pycharm\PyTorch_Models'
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, sem, ttest_1samp, mannwhitneyu
import matplotlib.pyplot as plt
import glob

# 尝试导入 CORnet
try:
    import cornet

    HAS_CORNET = True
except ImportError:
    HAS_CORNET = False
    print("⚠️ 警告: 未找到 cornet 库。")


# ==========================================
# 1. 自动生成针对您刺激矩阵的特征选择性掩码
# ==========================================
def generate_exact_feature_masks():
    """
    针对您 3x2x4 刺激集完美优化的特征解耦掩码生成器（严格校准版）
    """
    shapes = ['S1', 'S2', 'S3']
    textures = ['T1', 'T2']
    colors = ['Blue', 'Green', 'Red', 'Yellow']

    labels = []
    # 严格按照您图片的自然排列逻辑构建24标签
    for s in shapes:
        for t in textures:
            for c in colors:
                labels.append(f"{s}_{t}_{c}")

    n_total = len(labels)
    color_mask = np.zeros((n_total, n_total), dtype=bool)
    shape_mask = np.zeros((n_total, n_total), dtype=bool)
    texture_mask = np.zeros((n_total, n_total), dtype=bool)

    parsed_attrs = []
    for label in labels:
        s, t, c = label.split('_')
        parsed_attrs.append({'shape': s, 'texture': t, 'color': c})

    for i in range(n_total):
        for j in range(i + 1, n_total):
            a1 = parsed_attrs[i]
            a2 = parsed_attrs[j]

            # 1. 【纯颜色】：形状、纹理死锁完全相同，只有颜色不同 -> 48个点 (完全正确)
            if a1['shape'] == a2['shape'] and a1['texture'] == a2['texture'] and a1['color'] != a2['color']:
                color_mask[i, j] = True
                color_mask[j, i] = True

            # 2. 【纯纹理】：形状、颜色死锁完全相同，只有纹理不同 -> 12个点 (完全正确，完美控制全局形状)
            if a1['shape'] == a2['shape'] and a1['color'] == a2['color'] and a1['texture'] != a2['texture']:
                texture_mask[i, j] = True
                texture_mask[j, i] = True

            # 3. 【纯形状】：颜色必须死锁相同；
            # 同时保留 T1 vs T1 以及 T2 vs T2 的对比，最大化消除局部表面纹理特异性带来的潜在干扰 -> 36个点
            if a1['color'] == a2['color'] and a1['texture'] == a2['texture'] and a1['shape'] != a2['shape']:
                shape_mask[i, j] = True
                shape_mask[j, i] = True

    return color_mask, shape_mask, texture_mask


# ==========================================
# 2. 数据与模型初始化加载
# ==========================================
def load_individual_brain_data(filepath):
    mat = sio.loadmat(filepath)
    bird_rdms_3d = mat['bird_rdms']
    ceiling_lower = mat['static_lower_bound'].item()
    valid_birds_rdms = []
    # 注意：此处保留完整的 24x24 矩阵形式，因为后续需要进行空间位置掩码提取
    for i in range(bird_rdms_3d.shape[2]):
        single_rdm = bird_rdms_3d[:, :, i]
        if not np.isnan(single_rdm).all():
            # 确保对角线为0
            np.fill_diagonal(single_rdm, 0)
            valid_birds_rdms.append(single_rdm)
    return valid_birds_rdms, ceiling_lower


path_ento = r'F:\peng\Data_picture_fixed_Analyse_picture\fenxi\picture\data\RSA\data\RDMs_ENTO.mat'
path_mvl = r'F:\peng\Data_picture_fixed_Analyse_picture\fenxi\picture\data\RSA\data\RDMs_MVL.mat'
ento_birds, ento_ceiling = load_individual_brain_data(path_ento)
mvl_birds, mvl_ceiling = load_individual_brain_data(path_mvl)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_folder = r"F:\peng\picture1\bird picture1"
img_paths = sorted(glob.glob(os.path.join(img_folder, "*.png")) + glob.glob(os.path.join(img_folder, "*.jpg")))
images_tensor = torch.stack([transform(Image.open(p).convert('RGB')) for p in img_paths])


class FeatureExtractor:
    def __init__(self, model, layers_dict):
        self.model = model.eval()
        self.features = {}
        self.hooks = []
        for name, module in layers_dict.items():
            self.hooks.append(module.register_forward_hook(self.get_hook(name)))

    def get_hook(self, name):
        def hook(model, input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            if out_tensor.dim() == 3:
                feat = out_tensor[:, 0, :]
            else:
                feat = out_tensor.view(out_tensor.size(0), -1)
            self.features[name] = feat.detach().cpu().numpy()

        return hook

    def extract(self, x):
        self.features.clear()
        device = next(self.model.parameters()).device
        x = x.to(device)
        with torch.no_grad(): self.model(x)
        return self.features.copy()

    def remove_hooks(self):
        for h in self.hooks: h.remove()


print("⏳ 正在初始化 CNN/Transformer 大军...")
alex_pre = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
vgg_pre = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
res_pre = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
alex_rand = models.alexnet(weights=None)
vit_pre = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

model_configs = {
    'AlexNet': {'model': alex_pre,
                'layers': {'1': alex_pre.features[0], '2': alex_pre.features[3], '3': alex_pre.features[6],
                           '4': alex_pre.features[8], '5': alex_pre.features[10], '6': alex_pre.classifier[1]}},
    'VGG-16': {'model': vgg_pre,
               'layers': {'1': vgg_pre.features[4], '2': vgg_pre.features[9], '3': vgg_pre.features[16],
                          '4': vgg_pre.features[23], '5': vgg_pre.features[30], '6': vgg_pre.classifier[0],
                          '7': vgg_pre.classifier[3], '8': vgg_pre.classifier[6]}},
    'ResNet-50': {'model': res_pre,
                  'layers': {'1': res_pre.maxpool, '2': res_pre.layer1, '3': res_pre.layer2, '4': res_pre.layer3,
                             '5': res_pre.layer4, '6': res_pre.avgpool}},
    'Random': {'model': alex_rand,
               'layers': {'1': alex_rand.features[0], '2': alex_rand.features[3], '3': alex_rand.features[6],
                          '4': alex_rand.features[8], '5': alex_rand.features[10], '6': alex_rand.classifier[1]}},
    'ViT-B/16': {'model': vit_pre, 'layers': {'1': vit_pre.conv_proj, '2': vit_pre.encoder.layers.encoder_layer_0,
                                              '3': vit_pre.encoder.layers.encoder_layer_3,
                                              '4': vit_pre.encoder.layers.encoder_layer_7,
                                              '5': vit_pre.encoder.layers.encoder_layer_11, '6': vit_pre.encoder.ln}}
}

if HAS_CORNET:
    cornet_model = cornet.cornet_s(pretrained=True)
    cornet_base = cornet_model.module if isinstance(cornet_model, torch.nn.DataParallel) else cornet_model
    model_configs['CORnet-S'] = {'model': cornet_base,
                                 'layers': {'1': cornet_base.V1, '2': cornet_base.V2, '3': cornet_base.V4,
                                            '4': cornet_base.IT, '5': cornet_base.decoder.avgpool,
                                            '6': cornet_base.decoder.linear}}

# 生成解耦所需的上三角特征掩码
color_m, shape_m, texture_m = generate_exact_feature_masks()
triu_idx = np.triu_indices(24, k=1)

# 将掩码矩阵限制在上三角，匹配电生理的一维输入计算
color_mask_triu = color_m[triu_idx]
shape_mask_triu = shape_m[triu_idx]
texture_mask_triu = texture_m[triu_idx]

feature_masks = {
    'Color': color_mask_triu,
    'Shape': shape_mask_triu,
    'Texture': texture_mask_triu
}

# ==========================================
# 3. 核心大计算：包含特征解耦的多维层级对齐
# ==========================================
multifeat_peak_results = {}

for model_name, config in model_configs.items():
    print(f"\n>> 正在处理 {model_name} (特征解耦计算中) ...")
    extractor = FeatureExtractor(config['model'], config['layers'])
    feats_dict = extractor.extract(images_tensor)
    extractor.remove_hooks()

    layer_names = list(feats_dict.keys())
    num_layers = len(layer_names)

    # 预计算每一层的模型 RDM (一维上三角形式)
    layer_rdms = [pdist(feats_dict[lname], metric='correlation') for lname in layer_names]

    multifeat_peak_results[model_name] = {}

    # 对三大特征维度分别进行独立的层级对齐分析
    for feat_name, mask_vector in feature_masks.items():

        def analyze_brain_for_feature(birds_list):
            best_layer_depths = []
            for bird_rdm_2d in birds_list:
                bird_rdm_triu = bird_rdm_2d[triu_idx]

                # 过滤出该特征维度的专用数据子集
                bird_sub = bird_rdm_triu[mask_vector]

                layer_rhos = []
                for l_rdm in layer_rdms:
                    l_sub = l_rdm[mask_vector]
                    rho, _ = spearmanr(bird_sub, l_sub)
                    layer_rhos.append(rho if not np.isnan(rho) else -1.0)

                # 记录每只鸟在控制变量下响应最高层级的相对深度 (0 到 1)
                best_layer_idx = np.argmax(layer_rhos) + 1
                best_layer_depths.append(best_layer_idx / num_layers)

            return best_layer_depths


        e_depths = analyze_brain_for_feature(ento_birds)
        m_depths = analyze_brain_for_feature(mvl_birds)

        # 检验 MVL 的编码深度是否显著深于 ENTO (单尾检验)
        stat, p_hier = mannwhitneyu(m_depths, e_depths, alternative='greater')
        if p_hier < 0.001:
            h_star = '***'
        elif p_hier < 0.01:
            h_star = '**'
        elif p_hier < 0.05:
            h_star = '*'
        # elif p_hier < 0.1:
        #     h_star = '†'
        else:
            h_star = 'ns'

        multifeat_peak_results[model_name][feat_name] = {
            'e_mean': np.mean(e_depths), 'e_sem': sem(e_depths),
            'm_mean': np.mean(m_depths), 'm_sem': sem(m_depths),
            'h_star': h_star
        }

# ==========================================
# 4. 绘图 2 终极进化版: 完美呼吸间距多色聚合柱状图
# ==========================================
print("\n⏳ 正在绘制 终极全特征聚合柱状图 (优化间距且包含完整统计标注)...")
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

models_list = list(model_configs.keys())
n_models = len(models_list)

fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)

# --- 📐 几何间距控制中心 (解决挨得太近的核心算法) ---
w = 0.09          # 单个柱子的宽度
intra_gap = 0.02  # 🚀 核心修改：同一条件对内部，ENTO 与 MVL 之间的合适空隙
inter_gap = 0.07  # 不同特征大类之间的明显间隔

# 精准计算 6 根柱子在一个模型周期内的绝对坐标偏移量（含脑区内小间隙）
off_c_ento = -2.5 * w - 0.5 * intra_gap - inter_gap
off_c_mvl  = -1.5 * w + 0.5 * intra_gap - inter_gap
off_s_ento = -0.5 * w - 0.5 * intra_gap
off_s_mvl  =  0.5 * w + 0.5 * intra_gap
off_t_ento =  1.5 * w - 0.5 * intra_gap + inter_gap
off_t_mvl  =  2.5 * w + 0.5 * intra_gap + inter_gap

x = np.arange(n_models)

# 提取并准备数据
c_e = [multifeat_peak_results[m]['Color']['e_mean'] for m in models_list]
c_e_s = [multifeat_peak_results[m]['Color']['e_sem'] for m in models_list]
c_m = [multifeat_peak_results[m]['Color']['m_mean'] for m in models_list]
c_m_s = [multifeat_peak_results[m]['Color']['m_sem'] for m in models_list]

s_e = [multifeat_peak_results[m]['Shape']['e_mean'] for m in models_list]
s_e_s = [multifeat_peak_results[m]['Shape']['e_sem'] for m in models_list]
s_m = [multifeat_peak_results[m]['Shape']['m_mean'] for m in models_list]
s_m_s = [multifeat_peak_results[m]['Shape']['m_sem'] for m in models_list]

t_e = [multifeat_peak_results[m]['Texture']['e_mean'] for m in models_list]
t_e_s = [multifeat_peak_results[m]['Texture']['e_sem'] for m in models_list]
t_m = [multifeat_peak_results[m]['Texture']['m_mean'] for m in models_list]
t_m_s = [multifeat_peak_results[m]['Texture']['m_sem'] for m in models_list]

# ==================== 🎨 绘制三大特征柱状图 ====================
# 💙 Color - 蓝色系
ax.bar(x + off_c_ento, c_e, w, yerr=c_e_s, capsize=2, label='Color (ENTO)', color='#A0C4DF', edgecolor='#1F77B4', linewidth=1.2)
ax.bar(x + off_c_mvl,  c_m, w, yerr=c_m_s, capsize=2, label='Color (MVL)',  color='#1F77B4', edgecolor='#1F77B4', linewidth=1.2)

# 💚 Shape - 绿色系
ax.bar(x + off_s_ento, s_e, w, yerr=s_e_s, capsize=2, label='Shape (ENTO)', color='#C2E6C9', edgecolor='#2CA02C', linewidth=1.2)
ax.bar(x + off_s_mvl,  s_m, w, yerr=s_m_s, capsize=2, label='Shape (MVL)',  color='#2CA02C', edgecolor='#2CA02C', linewidth=1.2)

# ❤️ Texture - 红色系
ax.bar(x + off_t_ento, t_e, w, yerr=t_e_s, capsize=2, label='Texture (ENTO)', color='#F9C2C2', edgecolor='#D62728', linewidth=1.2)
ax.bar(x + off_t_mvl,  t_m, w, yerr=t_m_s, capsize=2, label='Texture (MVL)',  color='#D62728', edgecolor='#D62728', linewidth=1.2)


# ==================== 📊 绘制横线与显著性标注 ====================
def draw_stats_bracket(ax, x1, x2, y1, y2, star_text):
    """ 在指定的两根柱子上方绘制标准的学术对比横线和符号 """
    # 计算横线的绝对高度
    h_line = max(y1, y2) + 0.05
    # 绘制对比桥线
    ax.plot([x1, x2], [h_line, h_line], color='black', linewidth=0.8)
    # 在线中央标注显著性文本
    display_text = 'n.s.' if star_text == 'ns' else star_text
    font_sz = 12 if display_text == 'n.s.' else 14
    ax.text((x1 + x2) / 2, h_line + 0.005, display_text, ha='center', va='bottom',
            fontsize=font_sz, fontweight='normal' if display_text == 'n.s.' else 'bold')


for i, model in enumerate(models_list):
    # 1. Color 维度的横线与标注
    star_c = multifeat_peak_results[model]['Color']['h_star']
    draw_stats_bracket(ax, x[i] + off_c_ento, x[i] + off_c_mvl, c_e[i] + c_e_s[i], c_m[i] + c_m_s[i], star_c)

    # 2. Shape 维度的横线与标注
    star_s = multifeat_peak_results[model]['Shape']['h_star']
    draw_stats_bracket(ax, x[i] + off_s_ento, x[i] + off_s_mvl, s_e[i] + s_e_s[i], s_m[i] + s_m_s[i], star_s)

    # 3. Texture 维度的横线与标注
    star_t = multifeat_peak_results[model]['Texture']['h_star']
    draw_stats_bracket(ax, x[i] + off_t_ento, x[i] + off_t_mvl, t_e[i] + t_e_s[i], t_m[i] + t_m_s[i], star_t)

    # # 绘制大组（网络模型）之间的淡灰色垂直分隔走廊
    # if i < n_models - 1:
    #     ax.axvline(x=i + 0.5, color='#E0E0E0', linestyle=':', linewidth=1.0)

# ==================== 🛠️ 细节控制与美化 ====================
ax.set_ylabel('Relative hierarchical depth', fontsize=18)
ax.set_title('Hierarchical alignment across multiple visual features', fontsize=20, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=16)
ax.set_ylim([0, 1.25])  # 留出顶部空间放标注
# ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0)  # 1.0轴线

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 图例放置在最右侧外部
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=False, fontsize=12)

plt.tight_layout()
fig.savefig("Fig2_Combined_Multifeature_Aligned_Perfect.pdf", dpi=300)
print("✅ 完美对齐柱间间距的高质量大图已成功导出！")
plt.show()
