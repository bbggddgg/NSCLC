import numpy as np
import cv2
from skimage.registration import phase_cross_correlation  # 确保正确导入
from PIL import Image
import matplotlib.pyplot as plt
import os

# 读取图像
example_root = r"/path/to/dataset"
pred_img_path = os.path.join(example_root, "test_case", "tissue", "fusion_map_HD_heatmap.png")
ck_img_path = os.path.join(example_root, "test_case", "tissue", "ck.png")

pred_img = np.array(Image.open(pred_img_path).convert('RGB'))
ck_img = np.array(Image.open(ck_img_path).convert('RGB'))


# 1. 二值化预测图像（掩码图）
def binarize_prediction_image(img):
    """
    将预测图像（RGB分类掩码）二值化，提取前景区域
    """
    blue_color = np.array([0, 21, 126])
    green_color = np.array([53, 183, 119])

    # 创建掩码，保留非蓝色和非绿色的像素
    mask = ~(np.all(img == blue_color, axis=-1) | np.all(img == green_color, axis=-1))
    binary_img = np.zeros(img.shape[:2], dtype=np.uint8)
    binary_img[mask] = 1
    return binary_img


# 2. 二值化CK染色图像（可调整阈值）
def binarize_ck_image(img, threshold=128):
    """
    将CK染色图像二值化，提取染色区域
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转为灰度图
    _, binary_img = cv2.threshold(img_gray, threshold, 1, cv2.THRESH_BINARY)  # 二值化
    binary_img = 1 - binary_img  # 反转二值化结果
    return binary_img


# 3. 调整图像尺寸
def resize_images(img1, img2):
    """
    将两幅图像调整为相同的最小尺寸
    """
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h), interpolation=cv2.INTER_NEAREST)
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_NEAREST)
    return img1_resized, img2_resized


# 二值化图像
pred_binary = binarize_prediction_image(pred_img)
ck_binary = binarize_ck_image(ck_img, threshold=100)

# 调整尺寸
pred_binary, ck_binary = resize_images(pred_binary, ck_binary)

# 使用scikit-image的相位相关配准方法
shift, error, diffphase = phase_cross_correlation(pred_binary, ck_binary, upsample_factor=10)

# 打印偏移量
print(f"Computed shift: {shift}, Error: {error}")

# 4. 平移图像（使用 cv2.warpAffine 代替 np.roll）
def translate_image(img, shift):
    """
    平移图像，避免边界环绕
    """
    matrix = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])  # X 和 Y 方向的偏移量
    shifted_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT)
    return shifted_img


# 对原始预测图像进行平移
pred_aligned = translate_image(pred_img, shift)

# 可视化结果
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(pred_img)
axes[0].set_title("Original Prediction Image")

axes[1].imshow(ck_img)
axes[1].set_title("CK Image")

axes[2].imshow(pred_aligned)
axes[2].set_title("Aligned Prediction Image")

plt.show()
