#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 18:16
# @Author  : YangChenghan
# @File    : colorWSI.py
# @Description : 这个函数是用来balabalabala自己写

import os
import shutil
import time
import logging

import staintools
from PIL import Image
from tqdm import tqdm
import multiprocessing
import processing


def process_patch(normalizer, patch_path, color_path):
    patch = staintools.read_image(patch_path)
    patch = normalizer(patch)
    pil_image = Image.fromarray(patch)
    pil_image.save(color_path)


def color_WSI(root, normalizer, max_workers):
    cases = [name for name in os.listdir(root)]
    for case in cases:
        wsi_folder = os.path.join(root, case)
        normal_patches_folder = os.path.join(wsi_folder, 'PATCHES', 'normal')
        normal_color_folder = os.path.join(wsi_folder, 'COLOR', 'normal')

        tumor_patches_folder = os.path.join(wsi_folder, 'PATCHES', 'tumor')
        tumor_color_folder = os.path.join(wsi_folder, 'COLOR', 'tumor')

        if not os.path.exists(normal_color_folder):
            os.makedirs(normal_color_folder)
        else:
            shutil.rmtree(normal_color_folder)
            os.makedirs(normal_color_folder)

        if not os.path.exists(tumor_color_folder):
            os.makedirs(tumor_color_folder)
        else:
            shutil.rmtree(tumor_color_folder)
            os.makedirs(tumor_color_folder)

        start = time.time()
        pool = multiprocessing.Pool(max_workers)
        logging.info(f'---------------------------- color images: {normal_color_folder} start ----------------------------')

        for patch_name in tqdm(os.listdir(normal_patches_folder)):
            patch_path = os.path.join(normal_patches_folder, patch_name)
            color_path = os.path.join(normal_color_folder, patch_name)
            pool.apply_async(process_patch, args=(normalizer, patch_path, color_path))

        logging.info(f'color images: {normal_color_folder} finish. cost time :{time.time() - start}')

        start = time.time()
        for patch_name in tqdm(os.listdir(tumor_patches_folder)):
            patch_path = os.path.join(tumor_patches_folder, patch_name)
            color_path = os.path.join(tumor_color_folder, patch_name)
            pool.apply_async(process_patch, args=(normalizer, patch_path, color_path))

        pool.close()
        pool.join()

        logging.info(f'color images: {tumor_color_folder} finish. cost time :{time.time() - start}')
        logging.info(f"----------"*8)


