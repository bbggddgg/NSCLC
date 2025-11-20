import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    example_root = r"/path/to/dataset"
    root = os.path.join(example_root, "ki67_case", "tissue", "PATCHES", "level0")
    output_path = os.path.join(example_root, "test_case", "tissue", "ki67_results.csv")

    names = os.listdir(root)
    x_list = []
    y_list = []
    ratio_list = []

    for name in names:
        # 提取坐标
        name = name.replace(".jpg", "")
        x, y = name.split("_")[-2:]
        path = os.path.join(root, name + ".jpg")

        # 读取图像
        img = cv2.imread(path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换为 HSV 色彩空间

        # 定义棕色的 HSV 范围 (需要根据实际图像微调)
        # lower_brown = np.array([10, 50, 20])  # 棕色的下界 (Hue, Saturation, Value)
        # upper_brown = np.array([30, 255, 200])  # 棕色的上界

        # 定义扩展的棕色 HSV 范围
        lower_brown = np.array([0, 20, 20])  # 棕色的下界 (Hue, Saturation, Value)
        upper_brown = np.array([40, 255, 150])  # 棕色的上界，降低亮度以包含偏黑棕色

        # 提取棕色区域
        brown_mask = cv2.inRange(img_hsv, lower_brown, upper_brown)
        brown_pixels = np.sum(brown_mask > 0)  # 棕色像素计数
        total_pixels = brown_mask.size  # 图像总像素

        # 计算棕色占比
        brown_ratio = brown_pixels / total_pixels
        ratio_list.append(brown_ratio)

        x_list.append(x)
        y_list.append(y)

        # # 并排展示原图和掩膜图
        # plt.figure(figsize=(10, 5))  # 设置画布大小
        #
        # plt.subplot(1, 2, 1)  # 原图
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title("Original Image")
        # plt.axis('off')
        #
        # plt.subplot(1, 2, 2)  # 掩膜图
        # plt.imshow(brown_mask, cmap='gray')
        # plt.title("Brown Mask")
        # plt.axis('off')
        #
        # plt.suptitle(f"Brown Ratio = {brown_ratio:.4f}")  # 总标题显示棕色比例
        # plt.tight_layout()
        # plt.show()

        # print(f"Processed {name}: Brown Ratio = {brown_ratio:.4f}")

    # 保存结果到 CSV 文件
    df = pd.DataFrame({
        "x": x_list,
        "y": y_list,
        "ratio": ratio_list
    })

    df.to_csv(output_path, index=False)
    print("Processing complete. Results saved to:", output_path)

