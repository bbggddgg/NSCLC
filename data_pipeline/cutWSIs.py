#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 17:08
# @Author  : YangChenghan
# @File    : cutWSIs.py
# @Description : 这个函数是用来balabalabala自己写

import logging
import os
import time
import numpy as np
import processing

class WholeSlideImage():
    """
    用于切割WSI的类
    """
    def __init__(self, wsi_path, note_path, roi_folder, mask_level, patch_level, patch_size, ratio=0.5):

        self.wsi_path = wsi_path
        self.wsi_name = os.path.split(self.wsi_path)[-1]
        self.note_path = note_path
        self.mask_level = mask_level
        self.roi_folder = roi_folder
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.ratio = ratio
        self.roi = None

    def detach_patch(self, area_name='tumor'):
        """
        切割图像块
        Parameters
        ----------
        area_name :

        Returns
        -------

        """
        area_folder = os.path.join(self.roi_folder, area_name)
        if not os.path.exists(area_folder):
            logging.info('不存在 {} 区域'.format(area_name))
            return False
        else:
            logging.info('开始切割 {} 区域'.format(area_name))

            json_path = os.path.join(area_folder, 'data.json')
            npy_path = os.path.join(area_folder, 'data.npy')
            txt_path = os.path.join(area_folder, 'coords.txt')
            patch_folder = os.path.join(area_folder, 'PATCHES')

            processing.roi_mask_gen(self.wsi_path, json_path, npy_path, self.mask_level)
            logging.info('{} 区域掩膜生成完成'.format(area_name))

            processing.no_repeated_sampled(npy_path, txt_path, np.Infinity, self.mask_level, self.patch_size, ratio=self.ratio)
            logging.info('{} 区域采样完成'.format(area_name))

            logging.info('选取的倍镜为{}'.format(self.patch_level))
            if isinstance(self.patch_level, int):
                processing.patch_gen(self.wsi_path, patch_folder, txt_path, self.patch_size, self.patch_level, type=area_name)
            else:
                for cur_level in self.patch_level:
                    cur_patch_folder = os.path.join(patch_folder, f'level{cur_level}')
                    if not os.path.exists(cur_patch_folder):
                        os.makedirs(cur_patch_folder)

                    processing.patch_gen_bylevel(self.wsi_path, cur_patch_folder, txt_path, self.patch_size, cur_level, type=area_name)
                    logging.info("当前倍镜level为：{}切割完成".format(cur_level))

            logging.info("{} 区域切割完成\n".format(area_name))
            return True


    def process(self):
        """
        提取切片区域，以及切割各类图像块
        Returns
        -------

        """
        # 获取各区域的json文件
        # logging.info(f"当前文件为：{self.wsi_path}")
        self.roi = processing.ROI(self.note_path)
        self.roi.getArea()
        logging.info("note文件转换json格式 -- 成功\n")
        # 获取各区域的图像块
        nk_exist = self.detach_patch(area_name='necrosis_and_keratinize')
        stroma_exist = self.detach_patch(area_name='stroma')
        tumor_exist = self.detach_patch(area_name='viable_tumor')

        return self.wsi_name, nk_exist, stroma_exist, tumor_exist


def queryWSI(root, mask_level, patch_level, patch_size, ratio=0.8):
    """

    Parameters
    ----------
    root : 切片根路径
    mask_level : 掩膜层级
    patch_level : 多少倍镜下切图
    patch_size : 图像块大小
    ratio : 区域占比

    Returns
    -------

    """
    file_folder = os.listdir(root)
    for name in file_folder:
        start = time.time()
        logging.info("开始处理 {}".format(name))

        wsi_path = os.path.join(root, name, '1.tif')
        note_path = os.path.join(root, name, 'notes')

        if os.path.exists(note_path):

            roi_path = os.path.join(root, name)

            wsi = WholeSlideImage(wsi_path=wsi_path, note_path=note_path, roi_folder=roi_path,
                                  mask_level=mask_level, patch_level=patch_level, patch_size=patch_size, ratio=ratio)
            wsi.process()
        else:
            print(f"{wsi_path} 没有勾画")

        logging.info("处理 {} 完成, 花费{} 分钟".format(name, (time.time()-start) / 60))
        logging.info("----------"*8)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    root = r"/path/to/dataset/test"
    mask_level = 6
    patch_level = [0, 1, 2]
    patch_size = 256
    queryWSI(root, mask_level, patch_level, patch_size, ratio=0.8)


