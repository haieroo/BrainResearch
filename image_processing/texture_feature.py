# texture_feature.py
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def get_texture_features(base_path):
    img_dir = os.path.join(base_path, 'bird picture1')
    mask_dir = os.path.join(base_path, 'masks')

    n_points, radius = 16, 2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    feat_list_lbp, feat_list_glcm = [], []
    file_names = []

    # 【核心防御】：强制排序
    valid_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    for filename in valid_files:
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None: continue

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = clahe.apply(lab_img[:, :, 0])

        pure_core_mask = cv2.erode(binary_mask, morph_kernel, iterations=2)
        if cv2.countNonZero(pure_core_mask) < 200:
            pure_core_mask = cv2.erode(binary_mask, morph_kernel, iterations=1)
            if cv2.countNonZero(pure_core_mask) < 200:
                pure_core_mask = binary_mask

        # 提取 LBP
        lbp = local_binary_pattern(l_channel, n_points, radius, method='uniform')
        masked_lbp = lbp[pure_core_mask > 0]
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(masked_lbp, bins=n_bins, range=(0, n_bins), density=True)
        feat_list_lbp.append(lbp_hist)

        # 提取 GLCM
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
        file_names.append(filename)

    # 标准化并融合 LBP 和 GLCM
    scaler_lbp = StandardScaler()
    scaler_glcm = StandardScaler()
    norm_lbp = scaler_lbp.fit_transform(feat_list_lbp)
    norm_glcm = scaler_glcm.fit_transform(feat_list_glcm)

    final_features = np.hstack([norm_lbp, norm_glcm])
    # final_features = np.hstack([norm_glcm])
    return final_features, file_names