#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 15:31
# @Author  : YangChenghan
# @File    : drawgraph.py
# @Description : 这个函数是用来balabalabala自己写
import os

import drawing

def draw(csv_path, wsi_path, save_path):
    # label_names = ["存活肿瘤", "间质", "坏死与角化", "其他"]
    label_names = ["viable tumor", "stroma", "necrosis or keratinization", "others"]
    color_list = [(228, 227, 225), (193, 18, 33), (53, 183, 119), (248, 230, 32), (48, 104, 141)]

    threshold = 0.5
    mask_level = 7
    o_patch_size = 256
    drawing.draw_map(wsi_path, csv_path, o_patch_size, label_names, color_list, save_path, threshold, mask_level)

    save_folder_path = os.path.split(save_path)[0]
    plot = drawing.WSIPlot(csv_path, wsi_path, o_patch_size, threshold, mask_level)
    plot.gen_notes(save_folder_path)
    plot.gen_asap(save_folder_path)

if __name__ == '__main__':

    example_root = r"/path/to/dataset"
    csv_path = os.path.join(example_root, "test_case", "tissue", "MMT-Net_pred.csv")
    wsi_path = os.path.join(example_root, "test_case", "slide.tif")
    save_path = os.path.join(example_root, "test_case", "tissue", "slide_map.png")
    draw(csv_path, wsi_path, save_path)




