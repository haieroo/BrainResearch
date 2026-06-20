import os
os.environ['TORCH_HOME'] = r'F:\soft\pycharm\PyTorch_Models'
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist
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
# 1. 数据与模型初始化加载
# ==========================================
def load_individual_brain_data(filepath):
    mat = sio.loadmat(filepath)
    bird_rdms_3d = mat['bird_rdms']
    ceiling_lower = mat['static_lower_bound'].item()
    valid_birds_rdms = []
    triu_idx = np.triu_indices(24, k=1)
    for i in range(bird_rdms_3d.shape[2]):
        single_rdm = bird_rdms_3d[:, :, i]
        if not np.isnan(single_rdm).all():
            valid_birds_rdms.append(single_rdm[triu_idx])
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
            # --- 修改部分开始 ---
            if out_tensor.dim() == 3: # 针对 Transformer [B, Seq_Len, Dim]
                # 方式A: 取 CLS Token (第0个序列) [推荐]
                feat = out_tensor[:, 0, :]
                # 方式B: 或者对所有 Patch 求平均 feat = out_tensor[:, 1:, :].mean(dim=1)
            else: # 针对 CNN [B, C, H, W]
                feat = out_tensor.view(out_tensor.size(0), -1)
            # --- 修改部分结束 ---
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


# 初始化模型字典
print("⏳ 正在初始化 CNN/Transformer 大军...")
alex_pre = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
vgg_pre = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
res_pre = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
alex_rand = models.alexnet(weights=None)

# 🌟 新增：加载最新的 Vision Transformer (ViT-B/16)
vit_pre = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

model_configs = {
    # AlexNet: 6个功能阶段 (5个Conv块 + 1个FC块)
    'AlexNet': {
        'model': alex_pre,
        'layers': {
            '1': alex_pre.features[0],  # Conv 1
            '2': alex_pre.features[3],  # Conv 2
            '3': alex_pre.features[6],  # Conv 3
            '4': alex_pre.features[8],  # Conv 4
            '5': alex_pre.features[10],  # Conv 5
            '6': alex_pre.classifier[1]  # FC 6 (第一个全连接层)
        }
    },

    # VGG-16: 8个功能阶段 (5个池化块 + 3个全连接层)
    'VGG-16': {
        'model': vgg_pre,
        'layers': {
            '1': vgg_pre.features[4],  # Pool 1
            '2': vgg_pre.features[9],  # Pool 2
            '3': vgg_pre.features[16],  # Pool 3
            '4': vgg_pre.features[23],  # Pool 4
            '5': vgg_pre.features[30],  # Pool 5
            '6': vgg_pre.classifier[0],  # FC 6
            '7': vgg_pre.classifier[3],  # FC 7
            '8': vgg_pre.classifier[6]  # FC 8 (输出层)
        }
    },

    # ResNet-50: 6个功能阶段 (初始池化 + 4个残差块 + 全局池化)
    'ResNet-50': {
        'model': res_pre,
        'layers': {
            '1': res_pre.maxpool,  # Initial Maxpool (捕捉底层物理特征)
            '2': res_pre.layer1,  # Stage 1 (Bottleneck 1)
            '3': res_pre.layer2,  # Stage 2 (Bottleneck 2)
            '4': res_pre.layer3,  # Stage 3 (Bottleneck 3)
            '5': res_pre.layer4,  # Stage 4 (Bottleneck 4)
            '6': res_pre.avgpool  # Global Avg Pool (高维语义表征)
        }
    },
# 🌟 将随机模型放在第四个位置
    'Random': {
        'model': alex_rand,
        'layers': {'1': alex_rand.features[0], '2': alex_rand.features[3], '3': alex_rand.features[6], '4': alex_rand.features[8], '5': alex_rand.features[10], '6': alex_rand.classifier[1]}
    },
    # ViT-B/16: 6个功能阶段 (输入切片投影 + 均匀抽取的4个编码块 + 最终层归一化)
    'ViT-B/16': {
        'model': vit_pre,
        'layers': {
            '1': vit_pre.conv_proj,  # Stage 1: Patch Embedding
            '2': vit_pre.encoder.layers.encoder_layer_0,  # Stage 2: Encoder Block 1
            '3': vit_pre.encoder.layers.encoder_layer_3,  # Stage 3: Encoder Block 4
            '4': vit_pre.encoder.layers.encoder_layer_7,  # Stage 4: Encoder Block 8
            '5': vit_pre.encoder.layers.encoder_layer_11,  # Stage 5: Encoder Block 12
            '6': vit_pre.encoder.ln  # Stage 6: Final LayerNorm (语义聚合)
        }
    }
}

if HAS_CORNET:
    cornet_model = cornet.cornet_s(pretrained=True)
    cornet_base = cornet_model.module if isinstance(cornet_model, torch.nn.DataParallel) else cornet_model
    # 更新后的 6 层配置
    model_configs['CORnet-S'] = {
        'model': cornet_base,
        'layers': {
            '1': cornet_base.V1,  # Stage 1: V1
            '2': cornet_base.V2,  # Stage 2: V2
            '3': cornet_base.V4,  # Stage 3: V4
            '4': cornet_base.IT,  # Stage 4: IT
            '5': cornet_base.decoder.avgpool,  # Stage 5: Global Average Pooling
            '6': cornet_base.decoder.linear  # Stage 6: Final Classification Layer
        }}

# ==========================================
# 2. 核心大计算：包含三重顶刊统计检验
# ==========================================
results = {}
peak_layer_results = {}  # 用于存储图2的数据

for model_name, config in model_configs.items():
    print(f"\n>> 正在处理 {model_name} ...")
    extractor = FeatureExtractor(config['model'], config['layers'])
    feats_dict = extractor.extract(images_tensor)
    extractor.remove_hooks()

    layer_names = list(feats_dict.keys())
    num_layers = len(layer_names)

    # 预计算每一层的模型 RDM
    layer_rdms = [pdist(feats_dict[lname], metric='correlation') for lname in layer_names]


    # ---------------------------------------------------------
    # 统计函数封装 (执行 第一重检验 & 第三重检验的数据准备)
    # ---------------------------------------------------------
    def analyze_brain(birds_list, ceiling):
        all_rhos = np.zeros((len(birds_list), num_layers))
        all_var_exps = np.zeros((len(birds_list), num_layers))
        best_layer_idx = []  # 每只鸟的最佳层级(1到N)

        ceiling_var = ceiling ** 2

        for b_idx, bird_rdm in enumerate(birds_list):
            for l_idx, l_rdm in enumerate(layer_rdms):
                rho, _ = spearmanr(bird_rdm, l_rdm)
                all_rhos[b_idx, l_idx] = rho
                all_var_exps[b_idx, l_idx] = (rho ** 2) / ceiling_var if rho > 0 else 0.0
                # 🌟 核心修改：使用“带符号的方差解释率 (Signed Variance Explained)”
                # 这样 Random 模型的正负噪声在平均时会自动抵消为 0，而真实的 CNN 表征依然保持正的方差比例
                # all_var_exps[b_idx, l_idx] = np.sign(rho) * ((rho ** 2) / ceiling_var)
            # 【第三重检验准备】：找到这只鸟对应的最高rho所在的层级 (1-based)
            # 之前: best_layer_idx.append(np.argmax(all_rhos[b_idx, :]) + 1)
            # 建议修改为计算在 0 到 1 之间的相对深度：
            best_layer_idx.append((np.argmax(all_rhos[b_idx, :]) + 1) / num_layers)

        m_vars = np.mean(all_var_exps, axis=0)
        sem_vars = sem(all_var_exps, axis=0)

        # 【第一重检验】：Fisher-z 转换后的单样本T检验 (rho > 0)
        stars = []
        for l_idx in range(num_layers):
            rhos_clipped = np.clip(all_rhos[:, l_idx], -0.9999, 0.9999)
            z_vals = np.arctanh(rhos_clipped)
            try:
                _, p_val = ttest_1samp(z_vals, 0.0, alternative='greater')
                if p_val < 0.001:
                    stars.append('***')
                elif p_val < 0.01:
                    stars.append('**')
                elif p_val < 0.05:
                    stars.append('*')
                else:
                    stars.append('')
            except:
                stars.append('')

        # 【第二重检验】：找出平均方差最高的 Peak Layer，测试是否显著低于 1.0 (天花板)
        peak_idx = np.argmax(m_vars)
        peak_var_list = all_var_exps[:, peak_idx]
        try:
            _, p_ceil = ttest_1samp(peak_var_list, 1.0, alternative='less')
            limit_flag = '†' if p_ceil < 0.05 else ''
        except:
            limit_flag = ''

        return m_vars, sem_vars, stars, peak_idx, limit_flag, best_layer_idx


    # 运行 ENTO 和 MVL 的分析
    e_m, e_s, e_stars, e_peak_idx, e_limit, e_best_layers = analyze_brain(ento_birds, ento_ceiling)
    m_m, m_s, m_stars, m_peak_idx, m_limit, m_best_layers = analyze_brain(mvl_birds, mvl_ceiling)

    # 【第三重检验】：Mann-Whitney U 检验 (MVL 层级是否深于 ENTO 层级)
    stat, p_hier = mannwhitneyu(m_best_layers, e_best_layers, alternative='greater')
    if p_hier < 0.001:
        h_star = '***'
    elif p_hier < 0.01:
        h_star = '**'
    elif p_hier < 0.05:
        h_star = '*'
    elif p_hier < 0.1:
        h_star = '†'
    else:
        h_star = 'ns'

    # 保存结果以供绘图
    results[model_name] = {
        'layer_names': layer_names,
        'e_m': e_m, 'e_s': e_s, 'e_stars': e_stars, 'e_peak_idx': e_peak_idx, 'e_limit': e_limit,
        'm_m': m_m, 'm_s': m_s, 'm_stars': m_stars, 'm_peak_idx': m_peak_idx, 'm_limit': m_limit
    }

    peak_layer_results[model_name] = {
        'e_mean': np.mean(e_best_layers), 'e_sem': sem(e_best_layers),
        'm_mean': np.mean(m_best_layers), 'm_sem': sem(m_best_layers),
        'h_star': h_star, 'max_layer': num_layers
    }

import scipy.io as sio

# ==========================================
# 导出数据至 MATLAB (修正命名逻辑版本)
# ==========================================
print("\n💾 正在将计算结果导出至 MATLAB 格式...")

matlab_data = {
    'layer_wise_results': {},
    'summary_results': {
        'model_names': list(peak_layer_results.keys()),
        'ento_rel_depth_mean': [peak_layer_results[m]['e_mean'] for m in peak_layer_results],
        'ento_rel_depth_sem': [peak_layer_results[m]['e_sem'] for m in peak_layer_results],
        'mvl_rel_depth_mean': [peak_layer_results[m]['m_mean'] for m in peak_layer_results],
        'mvl_rel_depth_sem': [peak_layer_results[m]['m_sem'] for m in peak_layer_results],
        'h_stars': [peak_layer_results[m]['h_star'] for m in peak_layer_results]
    }
}

for model_name, d in results.items():
    # 核心修正：同时替换 '-' 和 '/' 为 '_'，确保 MATLAB 兼容性
    safe_name = model_name.replace('-', '_').replace('/', '_')

    matlab_data['layer_wise_results'][safe_name] = {
        'layer_names': d['layer_names'],
        'ento_mean': d['e_m'],
        'ento_sem': d['e_s'],
        'ento_stars': d['e_stars'],
        'mvl_mean': d['m_m'],
        'mvl_sem': d['m_s'],
        'mvl_stars': d['m_stars']
    }

save_file = "RSA_Alignment_Results_for_MATLAB.mat"
sio.savemat(save_file, matlab_data)
print(f"✅ 数据已保存！在 MATLAB 中请访问: data.layer_wise_results.ViT_B_16")


# ==========================================
# 3. 绘图 1: 第一与第二重检验 (方差解释率与天花板鸿沟)
# ==========================================
print("\n⏳ 正在绘制 图 1: 方差解释比例与表征鸿沟...")
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

num_models = len(model_configs)
fig1, axes1 = plt.subplots(1, num_models, figsize=(3.5 * num_models, 5), sharey=True)
if num_models == 1: axes1 = [axes1]
fig1.subplots_adjust(wspace=0)

c_ento, c_mvl ='#4C72B0','#C44E52'     #'#3498db', '#e74c3c' 1.65, 1.85  1.25, 1.45
gray_bottom, gray_top = 1.65, 1.85

for i, (model_name, d) in enumerate(results.items()):
    ax = axes1[i]
    x = np.arange(len(d['layer_names']))
    ax.axhspan(gray_bottom, gray_top, color='lightgray', alpha=0.5, zorder=0)

    # 绘制带误差棒折线
    ax.errorbar(x, d['e_m'], yerr=d['e_s'], fmt='-o', color=c_ento, linewidth=2.5, capsize=4,
                label='ENTO' if i == 0 else "")
    ax.errorbar(x, d['m_m'], yerr=d['m_s'], fmt='-o', color=c_mvl, linewidth=2.5, capsize=4,
                label='MVL' if i == 0 else "")
    ax.axhline(1.0, color='black', linestyle=':', linewidth=2, zorder=1)

    # 【打星号 (第一重) & 打十字架 (第二重)】
    for j in range(len(x)):
        # ENTO 的星星和十字架
        if d['e_stars'][j]:
            text_str = d['e_stars'][j]
            if j == d['e_peak_idx'] and d['e_limit'] == '†': text_str += '†'
            ax.text(x[j], gray_bottom + 0.02, text_str, color=c_ento, ha='center', va='bottom', fontsize=14,
                    fontweight='bold')

        # MVL 的星星和十字架
        if d['m_stars'][j]:
            text_str = d['m_stars'][j]
            if j == d['m_peak_idx'] and d['m_limit'] == '†': text_str += '†'
            ax.text(x[j], gray_bottom + 0.10, text_str, color=c_mvl, ha='center', va='bottom', fontsize=14,
                    fontweight='bold')
#, rotation=20 if 'CORnet' in model_name else 0
    ax.set_xticks(x)
    ax.set_xticklabels(d['layer_names'], fontsize=16)
    ax.set_title(model_name, fontsize=20,  pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
#, fontweight='bold'
    if i == 0:
        ax.set_ylabel("Proportion brain variance explained", fontsize=18)
        ax.spines['left'].set_linewidth(1.5)
        # ax.set_ylim([-0.05, 1.55])
        ax.set_ylim([-0.05, 2])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=16)
    else:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', left=False)

fig1.text(0.5, -0.05, 'Network layer', ha='center', fontsize=18)
plt.tight_layout()
fig1.savefig("Fig1_Variance_Explained.pdf", dpi=300, bbox_inches='tight')

# ==========================================
# 4. 绘图 2: 第三重检验 (层级递进柱状图)
# ==========================================
print("⏳ 正在绘制 图 2: 层级对应分析 (Peak Layer Analysis)...")
fig2, ax2 = plt.subplots(figsize=(10, 6))
models = list(peak_layer_results.keys())
x_pos = np.arange(len(models))
width = 0.35

e_means = [peak_layer_results[m]['e_mean'] for m in models]
e_sems = [peak_layer_results[m]['e_mean'] for m in models]
m_means = [peak_layer_results[m]['m_mean'] for m in models]
m_sems = [peak_layer_results[m]['m_sem'] for m in models]
h_stars = [peak_layer_results[m]['h_star'] for m in models]

rects1 = ax2.bar(x_pos - width / 2, e_means, width, yerr=[peak_layer_results[m]['e_sem'] for m in models], capsize=5,
                 label='ENTO', color=c_ento, edgecolor='black', linewidth=1.5)
rects2 = ax2.bar(x_pos + width / 2, m_means, width, yerr=m_sems, capsize=5, label='MVL', color=c_mvl, edgecolor='black',
                 linewidth=1.5)

for i, star in enumerate(h_stars):
    if star != 'ns':
        max_y = max(e_means[i] + peak_layer_results[models[i]]['e_sem'], m_means[i] + m_sems[i])
        ax2.text(x_pos[i], max_y + 0.2, star, ha='center', va='bottom', fontsize=18,  color='black')
#fontweight='bold',, fontweight='bold', fontweight='bold'
ax2.set_ylabel('Relative hierarchical depth', fontsize=18)
ax2.set_title('Hierarchical correspondence across models', fontsize=20,  pad=20)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, fontsize=16)
ax2.legend(frameon=False, fontsize=14)

max_layer_all = max([peak_layer_results[m]['max_layer'] for m in models])
# ax2.set_ylim([0, max_layer_all + 1.5])
# 直接把它删掉或注释掉，替换成下面这行：
ax2.set_ylim([0, 1.2])  # 因为相对深度最大是 1.0，设为 1.2 刚好能给上面的星号留出完美的空间
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
fig2.savefig("Fig2_Hierarchical_Correspondence.pdf", dpi=300)

print("\n✅ 所有统计学检验与顶刊双图绘制完毕！")
plt.show()