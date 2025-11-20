#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 19:59
# @Author  : YangChenghan
# @File    : gen_spot.py
# @Description : 这个函数是用来balabalabala自己写
import math
import os
import numpy as np
import logging

def no_repeated_sampled(npy_path, txt_path, number=300, level=4, patch_size=1024, ratio=0.8):
    """
    平移切割WSI，采样中心点，保证之后裁剪的图像块不会重复

    Parameters
    ----------
    npy_path : 掩膜文件路径
    txt_path : 采样点保存路径
    number : 采样数量
    level : 采样等级
    patch_size : 采样图像块大小
    ratio : 阈值（采样 if S‘ >= patch_size**2 * ratio ）

    Returns
    -------
    无

    Examples
    -------
    tumor_npy_path = "test_image/NPYs/tumor1/A161023-C2021-05-24_17_48_03_tumor.npy"
    tumor_txt_path = "test_image/COORDS/tumor1/A161023-C2021-05-24_17_48_03_tumor.txt"
    number = np.Infinity
    no_repeated_sampled(npy_path=tumor_npy_path, txt_path=tumor_txt_path, number=number)

    """
    sample_mask = np.load(npy_path)
    slide_h, slide_w = sample_mask.shape

    # print(f'patch_size:{patch_size}')
    # print(f'mask_shape:{slide_h} {slide_w}')

    mask_patch_size = patch_size // (2 ** level)
    area = math.ceil(mask_patch_size * mask_patch_size * ratio)
    # print(f"area: {area}")

    logging.info(f"裁剪图像块大小为:{patch_size}, 掩膜(H, W) = {slide_h},{slide_w}), "
                 f"缩放图像块大小:{mask_patch_size * mask_patch_size}, 置信面积为:{area}")

    X_idcs, Y_idcs = [], []
    for i in range(0, slide_h - mask_patch_size, mask_patch_size):
        for j in range(0, slide_w - mask_patch_size, mask_patch_size):
            tumor_mask_np = sample_mask[i:(i + mask_patch_size),
                            j: (j + mask_patch_size)]

            if (tumor_mask_np == True).sum() >= int(area):
                x, y = int(i + mask_patch_size // 2), int(j + mask_patch_size // 2)
                X_idcs.append(x)
                Y_idcs.append(y)

    center_points = np.stack(np.vstack((X_idcs, Y_idcs)), axis=1)
    logging.debug(f"总共中心点数量: {len(center_points)}")
    # print(f"中心点数量: {len(center_points)}")

    random_seed = 42
    np.random.seed(random_seed)
    if center_points.shape[0] > number:
        sampled_points = center_points[np.random.randint(center_points.shape[0], size=number), :]
    else:
        sampled_points = center_points
    sampled_points = (sampled_points * 2 ** level).astype(np.int32)
    mask_name = os.path.split(npy_path)[-1].split(".")[0]
    name = np.full((sampled_points.shape[0], 1), mask_name)
    center_points = np.hstack((name, sampled_points))

    # print(f"取样中心点数量: {len(center_points)}")
    logging.info(f"取样中心点数量: {len(center_points)}")
    with open(txt_path, 'w') as f:
        np.savetxt(f, center_points, fmt="%s", delimiter=",")


def get_mask_sampled(npy_path, txt_path, number, mask_level, patch_size):
    mask_tissue = np.load(npy_path)
    mask_patch_size = patch_size // (2 ** mask_level)
    slide_h, slide_w = mask_tissue.shape

    X_idcs, Y_idcs = [], []
    for i in range(0, slide_h):
        for j in range(0, slide_w):
            if mask_tissue[i, j] == True:
                x, y = int(i + mask_patch_size // 2), int(j + mask_patch_size // 2)
                X_idcs.append(x)
                Y_idcs.append(y)

    center_points = np.stack(np.vstack((X_idcs, Y_idcs)), axis=1)
    random_seed = 42
    np.random.seed(random_seed)
    if center_points.shape[0] > number:
        sampled_points = center_points[np.random.randint(center_points.shape[0], size=number), :]
    else:
        sampled_points = center_points
    sampled_points = (sampled_points * 2 ** mask_level).astype(np.int32)
    mask_name = os.path.split(npy_path)[-1].split(".")[0]
    name = np.full((sampled_points.shape[0], 1), mask_name)
    center_points = np.hstack((name, sampled_points))

    # print(f"取样中心点数量: {len(center_points)}")
    logging.info(f"取样中心点数量: {len(center_points)}")
    with open(txt_path, 'w') as f:
        np.savetxt(f, center_points, fmt="%s", delimiter=",")

