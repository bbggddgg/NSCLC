#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/20 17:15
# @Author  : YangChenghan
# @File    : gen_patches.py
# @Description : 这个函数是用来balabalabala自己写
import json
import os
import shutil
import time

from PIL import Image
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import logging
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

def get_patches(json_file_path, wsi_file_path, patch_size=256,save_file_folder=''):
    with open(json_file_path, 'r') as f:
        file = json.load(f)

    start = time.time()
    WSI = openslide.open_slide(wsi_file_path)

    logging.info(f"读取WSI花费 {time.time() - start} s")

    for it in file:
        points = it['points']
        name = it['name']

        save_patch_folder = os.path.join(save_file_folder, name)
        if not os.path.exists(save_patch_folder):
            os.makedirs(save_patch_folder)
        else:
            shutil.rmtree(save_patch_folder)
            os.makedirs(save_patch_folder)

        start = time.time()
        logging.info(f'ROI: {name}, 开始切割图像块')

        for p in tqdm(points):
            x, y = p
            patch = WSI.read_region((x,y), 0, (patch_size,patch_size))
            save_patch_path = os.path.join(save_patch_folder, '{}_{}.png'.format(x,y))
            patch.save(save_patch_path)
            # print(save_patch_path)
        logging.info(f'ROI:{name}, 花费时间{time.time() - start} s')
            # print(x, y)
