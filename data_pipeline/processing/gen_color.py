#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 10:53
# @Author  : YangChenghan
# @File    : color_patch.py
# @Description : 这个函数是用来对WSI统一染色
import time

import staintools
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv
def standard_transfrom(standard_img,method = 'M'):
    if method == 'V':
        stain_method = staintools.StainNormalizer(method='macenko')
        stain_method.fit(standard_img)
    else:
        stain_method = staintools.StainNormalizer(method='vahadane')
        stain_method.fit(standard_img)


    return stain_method


class Stain():
    def __init__(self, standard_img, method='M'):
        if method == 'M':
            standard_img = staintools.LuminosityStandardizer.standardize(standard_img)
            self.stain_method = staintools.StainNormalizer(method='macenko')
            self.stain_method.fit(standard_img)
        else:
            standard_img = staintools.LuminosityStandardizer.standardize(standard_img)
            self.stain_method = staintools.StainNormalizer(method='vahadane')
            self.stain_method.fit(standard_img)

    def __call__(self, x):
        # x = np.array(x)
        x = staintools.LuminosityStandardizer.standardize(x)
        x = self.stain_method.transform(x)
        return x

