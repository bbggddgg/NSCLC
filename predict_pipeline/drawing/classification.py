#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/9 17:42
# @Author  : YangChenghan
# @File    : classification.py
# @Description : 这个函数是用来画分类图热图
import math
import os
from gettext import translation

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from matplotlib.cm import ScalarMappable

# plt.rcParams['font.size'] = 18
plt.rcParams['font.sans-serif'] = ['SimHei']
print(plt.rcParams['font.sans-serif'])

OPENSLIDE_PATH = os.environ.get('OPENSLIDE_PATH', r'/path/to/openslide/bin')
try:
    if hasattr(os, 'add_dll_directory'):
        # Python >= 3.8 on Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except Exception:
    pass

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

def drawPieChart(ax, df, label_column, label_names, title, colors):
    """
    Parameters
    ----------
    ax : 画布
    df : pandas DataFrame，包含预测结果 (x, y, label)
    label_column : 标签列名 (e.g., 'label')
    label_names : 标签对应的名称, [normal, tumor1, tumor2]
    title : 图名
    colors : 颜色集 [class1, class2, class3,...,class_n]

    Returns
    -------
    无
    """
    # 统计标签数量
    label_counts = df[label_column].value_counts().sort_index()
    print(label_counts)

    # 筛选出有标签的类别和对应的颜色
    non_zero_label_counts = label_counts[label_counts > 0]
    non_zero_label_names = [label_names[i] for i in non_zero_label_counts.index]
    non_zero_colors = [colors[i] for i in non_zero_label_counts.index]

    # 绘制饼图
    pos = 0
    if non_zero_label_counts.iloc[0] / non_zero_label_counts.sum() < 0.1:
        pos = 0.15

    patches, texts, autotexts = ax.pie(
        x=non_zero_label_counts,
        colors=non_zero_colors,
        autopct='%.2f%%',
        radius=1.1,
        startangle=90,
        pctdistance=0.8,
        explode=(pos,) + (0,) * (len(non_zero_label_counts) - 1)
    )

    # 自定义每个类别的百分比文本距离
    offsets = [0.8, 0.7, 0.95, 0.8]  # 根据类别设置不同距离

    for autotext, offset in zip(autotexts, offsets):
        x, y = autotext.get_position()  # 获取当前位置
        autotext.set_position((x * offset, y * offset))  # 按比例调整距离

    # 设置字体属性
    font_family = "SimHei"
    font_size = 26
    for text in texts:
        text.set_fontfamily(font_family)
        text.set_fontsize(font_size)
    for autotext in autotexts:
        autotext.set_fontfamily(font_family)
        autotext.set_fontsize(font_size)

    # 添加图例并设置字体
    ax.legend(
        patches,
        non_zero_label_names,
        loc='center left',
        bbox_to_anchor=(0.15, -0.15, 0, 0),
        frameon=False,
        prop={'family': font_family, 'size': font_size}
    )

    # 获取当前Axes的位置并调整位置
    pos = ax.get_position()
    new_pos = [pos.x0, pos.y0 + 0.1, pos.width, pos.height]
    ax.set_position(new_pos)


def drawWSI(ax, wsi_path, level=6, title="WSI"):
    """

    Parameters
    ----------
    ax : 画布
    wsi_path : 原始WSI路径
    level : 缩放等级
    title : 图名

    Returns
    -------
    无

    """
    slide = openslide.OpenSlide(wsi_path)
    level = len(slide.level_dimensions)-1
    slide_np = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))

    ax.imshow(slide_np)
    ax.axis('off')
    return slide.level_dimensions[0], level

def save_subfig(fig, ax, save_path):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, bbox_inches=extent, transparent=True, dpi=400)

def drawHeatmap(ax, zoom, z_shape, z_patch_size, threshold, csv_path, save_path,  title='Tissue Area', colors=None):
    """

    Parameters
    ----------
    ax : 画布
    shape : 原始WSI大小
    csv_path : 预测结果存放csv(x, y, label)
    patch_size : 图像块大小
    title : 图名
    colors : 颜色集[class1, class2, class3,...,background_color]

    Returns
    -------
    无

    """

    # 背景色
    background_color = colors[0]

    print(f"z_shape: {z_shape}")
    probs_map = np.full(z_shape+(3,), background_color, dtype=np.uint8)

    df = pd.read_csv(csv_path)

    label_list = []
    for index, row in df.iterrows():
        pred = [row[0], row[1], row[2]]
        label = pred.index(max(pred))
        # label = 1 if pred[label] < threshold else label
        if label == 0 and pred[label] < threshold:
            label = 1 if row[1] > row[2] else 2

        label_list.append(label)

        center_x = int(row['x'] // zoom)
        center_y = int(row['y'] // zoom)

        left_x = center_x - z_patch_size // 2
        left_y = center_y - z_patch_size // 2

        color = colors[label+1]
        probs_map[left_x:left_x+z_patch_size, left_y:left_y+z_patch_size] = color

    probs_map = np.transpose(probs_map, axes=[1, 0, 2])

    img = Image.fromarray(probs_map)
    img.save(save_path, transparent=True)

    # print(probs_map.shape)
    df['label'] = label_list
    im = ax.imshow(probs_map, cmap='rainbow')
    ax.axis('off')

    return df
    # ax.set_title(title)

def draw_map(wsi_path, csv_path, o_patch_size, label_names, colors, save_path=None, threshold=0.6, mask_level=-1):
    """

    Parameters
    ----------
    wsi_path : 原始WSI路径
    csv_path : 预测结果存放csv(x, y, label)
    label_names : 标签对应的名称, [normal, tumor1, tumor2]
    colors : 颜色集[class1, class2, class3,...,class_n, background_color]

    Returns
    -------
    热力图

    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1, 1]})
    ax_wsi = axs[0]
    ax_heatmap = axs[1]
    ax_pie = axs[2]

    o_shape, z_level = drawWSI(ax_wsi,  wsi_path, title="original WSIs")
    # o_patch_size = 256
    # patch_size = 256
    zoom = 2 ** mask_level

    mask_w, mask_h = o_shape[0] // zoom, o_shape[1] // zoom
    z_shape = (mask_w, mask_h)
    z_patch_size = o_patch_size // zoom

    df = drawHeatmap(ax=ax_heatmap, zoom=zoom, z_shape=z_shape, z_patch_size=z_patch_size, threshold=threshold,
                     csv_path=csv_path, save_path=save_path.replace(".png", '_HD_heatmap.png'), title="classification map", colors=colors)



    pie_colors = [mcolors.rgb2hex((r / 255, g / 255, b / 255)) for r, g, b in colors[1:]]

    drawPieChart(ax_pie, df, 'label', label_names, "pie chart", pie_colors)

    save_subfig(fig, ax_wsi, save_path.replace(".png", '_wsi.png'))
    save_subfig(fig, ax_heatmap, save_path.replace(".png", '_heatmap.png'))
    save_subfig(fig, ax_pie, save_path.replace(".png", '_pie.png'))
    # fig.subplots_adjust(top=1.0)
    # plt.tight_layout()
    plt.savefig(save_path, transparent=True, dpi=400)
    # plt.show()






