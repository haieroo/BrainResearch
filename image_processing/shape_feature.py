# shape_feature.py
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def get_shape_features(base_path):
    mask_dir = os.path.join(base_path, 'masks')

    features = []
    file_names = []

    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    # 【核心防御】：同样强制排序
    valid_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    for filename in valid_files:
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if mask is None: continue

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        x, y, w, h = cv2.boundingRect(binary_mask)
        if w >= 8 and h >= 8:
            cropped_mask = binary_mask[y:y + h, x:x + w]
        else:
            cropped_mask = binary_mask

        resized_mask = cv2.resize(cropped_mask, (64, 64))
        hog_features = hog.compute(resized_mask).flatten()

        features.append(hog_features)
        file_names.append(filename)

    features_array = np.array(features)

    # 对 HOG 特征进行标准化，确保距离计算不受绝对数值影响
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    return features_scaled, file_names