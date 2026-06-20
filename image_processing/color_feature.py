# color_feature.py
import cv2
import numpy as np
import os


def get_color_features(base_path):
    img_dir = os.path.join(base_path, 'bird picture1')
    mask_dir = os.path.join(base_path, 'masks')

    features = []
    file_names = []

    # 【核心防御】：强制按字母顺序排序，确保 24 张图的顺序绝对一致
    valid_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    for filename in valid_files:
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None: continue

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

            mean_a = np.mean(colorful_pixels[:, 1])
            mean_b = np.mean(colorful_pixels[:, 2])

            features.append([mean_a, mean_b])
            file_names.append(filename)

    return np.array(features), file_names