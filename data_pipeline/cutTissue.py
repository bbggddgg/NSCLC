#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 10:16
# @Author  : YangChenghan
# @File    : cutTissue.py
# @Description : 这个函数是用来balabalabala自己写

import logging
import os
import time

import processing
import numpy as np
import matplotlib.pyplot as plt

def process(wsi_path, npy_path, txt_path, patch_folder, number, mask_level, patch_level, patch_size, RGB_min):

    # wsi_name = os.path.split(wsi_path)[-1]
    flag = processing.tissue_mask_gen(wsi_path, npy_path, mask_level, RGB_min)
    if flag == False:
        logging.info(f"WSI: {wsi_path} 无法在level {mask_level}下生成组织掩膜")
        return False

    image = np.transpose(np.load(npy_path))
    plt.imshow(image, cmap='gray')
    plt.title(wsi_path)
    plt.show()

    # processing.no_repeated_sampled(npy_path, txt_path, number, mask_level, patch_size, 0.5)
    processing.get_mask_sampled(npy_path, txt_path, number, mask_level, patch_size)

    if isinstance(patch_level, int):
        processing.patch_gen(wsi_path, patch_folder, txt_path, patch_size, patch_level, type='tissue')
    else:
        for cur_level in patch_level:
            cur_patch_folder = os.path.join(patch_folder, f'level{cur_level}')
            if not os.path.exists(cur_patch_folder):
                os.makedirs(cur_patch_folder)
            logging.info(f"当前倍镜level为：{cur_level}, 文件夹为:{cur_patch_folder}")
            processing.patch_gen_bylevel(wsi_path, cur_patch_folder, txt_path, patch_size, cur_level, type='tissue')

def queryWSI(root, folder_name=None):
    if folder_name is None:
        folder_name = os.listdir(root)

    for case in folder_name:

        start = time.time()
        case_path = os.path.join(root, case)
        wsi_path = os.path.join(case_path, '1.tif')
        tissue_folder = os.path.join(case_path, 'tissue')
        patch_folder = os.path.join(tissue_folder, 'PATCHES')

        if not os.path.exists(tissue_folder):
            os.makedirs(tissue_folder)
        if not os.path.exists(patch_folder):
            os.makedirs(patch_folder)

        npy_path = os.path.join(tissue_folder, 'data.npy')
        txt_path = os.path.join(tissue_folder, 'coords.txt')

        number = np.Infinity

        mask_level = 7
        patch_level = [0, 2]
        patch_size = 256
        RGB_min = 0

        process(wsi_path=wsi_path, npy_path=npy_path, txt_path=txt_path, patch_folder=patch_folder,
                number=number, mask_level=mask_level, patch_level=patch_level, patch_size=patch_size, RGB_min=RGB_min)

        print(f'单线程情况下：{wsi_path} 完成, 花费{time.time()-start} s')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    root = r"/path/to/dataset"
    folder_name = ['sample_case_ki67', 'sample_case_ck']
    queryWSI(root, folder_name)

