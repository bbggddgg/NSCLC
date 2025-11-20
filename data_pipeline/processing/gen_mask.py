#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 18:22
# @Author  : YangChenghan
# @File    : gen_mask.py
# @Description : 这个函数是用来balabalabala自己写

import json
import logging
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters.thresholding import threshold_otsu
import cv2

OPENSLIDE_PATH = os.environ.get('OPENSLIDE_PATH', r'/path/to/openslide/bin')

try:
    if hasattr(os, 'add_dll_directory'):
        # Python >= 3.8 on Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except:
    pass


def tissue_mask_gen(wsi_path, npy_path, level=2, RGB_min=50):
    """
    生成组织掩膜

    Parameters
    ----------
    wsi_path : WSI路径
    npy_path : tissue保存路径
    level : 取样的级别
    RGB_min : OSTU 阈值

    Returns
    -------
    无

    Examples
    -------
    tissue_mask_gen(wsi_path=wsi_path, npy_path=tissue_path, level=2, RGB_min=0)
    tissue_np = np.transpose(np.load(tissue_path))
    plt.imshow(tissue_np, cmap='gray')
    plt.show()

    """
    slide = openslide.OpenSlide(wsi_path)
    if (len(slide.level_dimensions)-1) < level:
        return False

    print(f"在level {level}下WSI的(宽，高): {slide.level_dimensions[level]}")
    # factor = int(slide.level_dimensions[0][0] // slide.level_dimensions[level][0])

    logging.info(f"在level {level}下WSI的(W, H): {slide.level_dimensions[level]}")
    img_RGB = np.transpose(np.array(
        slide.read_region((0, 0), level,
                          slide.level_dimensions[level]).convert('RGB')),
        axes=[1, 0, 2])

    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])

    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    # print(f"tissue_mask.shape: {tissue_mask.shape}")

    np.save(npy_path, tissue_mask)


def roi_mask_gen(wsi_path, json_path, npy_path, level):
    slide = openslide.open_slide(wsi_path)
    w, h = slide.level_dimensions[level]

    # (W, H, C) --> (H, W, C); Image --> Array
    mask_roi = np.zeros((h, w))
    factor = slide.level_downsamples[level]

    with open(json_path) as f:
        rois = json.load(f)

    for roi in rois:
        name = roi['name']
        vertices = np.array(roi['vertices']) / factor
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(mask_roi, [vertices], (255))

    mask_roi = mask_roi[:] > 127
    mask_roi = np.transpose(mask_roi)
    # mask_tissue = np.load(tissue_npy_path)

    # mask_roi = mask_roi & mask_tissue

    np.save(npy_path, mask_roi)

