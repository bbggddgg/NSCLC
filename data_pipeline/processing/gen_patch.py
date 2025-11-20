#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 20:24
# @Author  : YangChenghan
# @File    : gen_patch.py
# @Description : 这个函数是用来balabalabala自己写
import concurrent.futures
import os
import logging

from pandas.core.array_algos.transforms import shift

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

def process_patch(line, slide, save_folder, filename, level, patch_size,type):
    pid, x_center, y_center = line.strip('\n').split(',')
    # x = int(int(x_center) - patch_size / 2) * scale
    # y = int(int(y_center) - patch_size / 2) * scale
    x = int(int(x_center) - patch_size / 2)
    y = int(int(y_center) - patch_size / 2)
    img = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
    patch_path = os.path.join(save_folder, f'{filename.replace(".tif", "_" + type)}_{x}_{y}' + '.jpg')
    img.save(patch_path)

def patch_gen(wsi_path, save_folder, txt_path, patch_size=256, level=0, type="tumor", shfit=[0,0]):

    infile = open(txt_path)
    slide = openslide.open_slide(wsi_path)
    filename = (os.path.split(wsi_path)[-1])

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, line in enumerate(infile):
        pid, x_center, y_center = line.strip('\n').split(',')
        x = int(int(x_center) - patch_size // 2)+shift[0]
        y = int(int(y_center) - patch_size // 2)+shift[1]
        img = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
        patch_path = os.path.join(save_folder, f'{filename.replace(".tif", "_" + type)}_{x}_{y}' + '.jpg')

        # print(patch_path)
        logging.debug(patch_path)
        img.save(patch_path)


def patch_gen_bylevel(wsi_path, save_folder, txt_path, patch_size=256, level=0, type="tumor", shfit=[0,0]):
    infile = open(txt_path)
    slide = openslide.open_slide(wsi_path)
    filename = (os.path.split(wsi_path)[-1]).replace(".tif", "")
    # print(filename)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    for i, line in enumerate(infile):
        pid, x_center, y_center = line.strip('\n').split(',')
        bias = int(0.5 * (2**level) * patch_size)

        left_x = int(x_center) - bias + shfit[0]
        left_y = int(y_center) - bias + shfit[1]

        img = slide.read_region((left_x, left_y), level, (patch_size, patch_size)).convert('RGB')
        # patch_path = os.path.join(save_folder, f'{filename.replace(".tif", "_" + type)}_level{level}_{x_center}_{y_center}' + '.png')
        patch_name = f"{filename}_{type}_level{level}_{x_center}_{y_center}.jpg"
        patch_path = os.path.join(save_folder, patch_name)
        # print(patch_path)
        # logging.debug(patch_path)
        img.save(patch_path)

