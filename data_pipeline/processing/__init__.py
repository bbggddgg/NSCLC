#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/18 17:14
# @Author  : YangChenghan
# @File    : __init__.py.py
# @Description : 这个函数是用来balabalabala自己写

# import os
# OPENSLIDE_PATH = r'/path/to/openslide/bin'
#
# try:
#     if hasattr(os, 'add_dll_directory'):
#         # Python >= 3.8 on Windows
#         with os.add_dll_directory(OPENSLIDE_PATH):
#             import openslide
#     else:
#         import openslide
# except:
#     pass

from .gen_roi import ROI
from .gen_mask import roi_mask_gen, tissue_mask_gen
from .gen_spot import no_repeated_sampled, get_mask_sampled
from .gen_patch import patch_gen, patch_gen_bylevel
# from .gen_color import Stain
# from .convertTiff import mds2tiff
