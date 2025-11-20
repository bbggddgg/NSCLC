#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 15:47
# @Author  : YangChenghan
# @File    : cvs2xml.py
# @Description : 这个函数是用来balabalabala自己写
import math
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
import xml.dom.minidom
OPENSLIDE_PATH = os.environ.get('OPENSLIDE_PATH', r'/path/to/openslide/bin')
try:
    if hasattr(os, 'add_dll_directory'):
        # Python >= 3.8 on Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except Exception:
    pass


class WSIPlot():
    def __init__(self, predict_csv_path, wsi_path, patch_size, threshold=0.5, mask_level=7):
        self.slide = openslide.open_slide(wsi_path)
        self.shape = self.slide.level_dimensions[0]
        self.predict_df = pd.read_csv(predict_csv_path, sep=',')
        self.patch_size = patch_size
        self.mask_patchsize = self.patch_size
        self.mask_level = mask_level
        self.zoom = 2**mask_level
        self.threshold = threshold
        self.coordinates = []

    def gen_mask(self, y_label=0):

        w, h = self.shape[0] // self.zoom, self.shape[1] // self.zoom

        mask_roi = np.zeros((w, h))

        for index, row in self.predict_df.iterrows():
            pred = [row[0], row[1], row[2]]
            label = pred.index(max(pred))
            # label = 3 if pred[label] < self.threshold else label
            if label == 0 and pred[label] < self.threshold:
                label = 1 if row[1] > row[2] else 2

            if label == y_label:
                # center_x = int(row['x'] + self.patch_size// 2) // self.patch_size
                # center_y = int(row['y'] + self.patch_size // 2) // self.patch_size

                center_x = int(row['x']) // self.zoom
                center_y = int(row['y']) // self.zoom
                mask_roi[center_x, center_y] = 255

        mask_roi = np.transpose(mask_roi)
        mask_roi = np.uint8(mask_roi)

        return mask_roi

    def gen_level_mask(self, y_label=0):
        if self.mask_level > 8 or self.mask_level < 0:
            raise ValueError("mask_level must be in range 0-7")

        # self.zoom = 2**mask_level
        mask_w, mask_h = self.shape[0] // self.zoom, self.shape[1] // self.zoom
        self.mask_patchsize = self.patch_size // self.zoom
        mask_roi = np.zeros((mask_w, mask_h))

        for index, row in self.predict_df.iterrows():
            pred = [row[0], row[1], row[2]]
            label = pred.index(max(pred))
            # label = 3 if pred[label] < self.threshold else label
            if label == 0 and label < self.threshold:
                label = 1 if row[1] > row[2] else 2

            if label == y_label:
                center_x = int(row['x']) // self.zoom
                center_y = int(row['y']) // self.zoom

                left_x = center_x - self.mask_patchsize // 2
                left_y = center_y - self.mask_patchsize // 2
                mask_roi[left_x:left_x+self.mask_patchsize, left_y:left_y+self.mask_patchsize] = 255

        mask_roi = np.transpose(mask_roi)
        mask_roi = np.uint8(mask_roi)
        return mask_roi

    def gen_coords(self, mask, roi_peakpum=3):

        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        all_coordinates = []

        for coord_array in contours:
            # eps = 0.00001 * cv2.arcLength(coord_array, True)
            # approx_array = cv2.approxPolyDP(coord_array, eps, True)
            if len(coord_array) >= max(5, roi_peakpum):

                coordinates = []
                for point in coord_array:
                    x_point = (point[0][0] * self.zoom)
                    y_point = (point[0][1] * self.zoom)

                    # coordinates.append({"X": x_point-5, "Y": y_point-5})
                    # coordinates.append({"X": x_point+min(128, self.patch_size), "Y": y_point+min(128,self.patch_size)})
                    coordinates.append({"X": x_point, "Y": y_point})
                    # coordinates.append({"X": x_point+5, "Y": y_point+5})
                all_coordinates.append(coordinates)

        return all_coordinates

    def draw_roiplot(self, coordinates, save_folder_path):
        # color_list = ['#FF0000', '#00CC66', '#FFFF00']

        # 定义三类轮廓颜色 (RGB格式)
        color_list = [
            (255, 0, 0),  # 肿瘤 - 红色
            (0, 204, 102),  # 间质 - 绿色
            (255, 255, 0)  # 坏死 - 黄色
        ]
        # 设置图像的大小
        # plt.figure(figsize=(12, 12))  # 调整图像大小，单位是英寸

        slide_np = np.array(self.slide.read_region((0, 0), self.mask_level, self.slide.level_dimensions[self.mask_level]).convert('RGB'))

        for index in range(len(coordinates)):
            cur_coordinates = coordinates[index]
            cur_color = color_list[index]

            for coord in cur_coordinates:
                scaled_contour = []
                for point in coord:
                    x, y= point['X'], point['Y']
                    # print(x, y)
                    x = int(round(x / self.zoom))
                    y = int(round(y / self.zoom))
                    scaled_contour.append((x, y))

                if len(scaled_contour) >= 3:
                    # 转换为OpenCV格式 (N,1,2)
                    contour_np = np.array([scaled_contour], dtype=np.int32)
                    # 绘制轮廓（线宽=3，填充内部）
                    cv2.drawContours(
                        slide_np,
                        [contour_np],
                        contourIdx=-1,
                        color=cur_color,
                        thickness=1,
                        lineType=cv2.LINE_AA  # 抗锯齿
                    )

        save_path = os.path.join(save_folder_path, 'proj.png')
        cv2.imwrite(save_path, cv2.cvtColor(slide_np, cv2.COLOR_RGB2BGR))

        # 可选：显示图像
        # plt.imshow(slide_np)
        # plt.axis('off')
        # plt.show()

        # plt.axis('off')
        # plt.imshow(slide_np)
        # plt.savefig(os.path.join(save_folder_path, 'proj.png'), dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.show()
        # plt.figure(figsize=(12, 12))
        # plt.imshow(slide_np)
        # plt.axis('off')  # 隐藏坐标轴
        # plt.tight_layout()




    def gen_notes(self, save_folder):

        # ["viable_tumor", "stroma", "necrosis_and_keratinize"]
        tumor = self.gen_mask(y_label=0)
        stroma = self.gen_mask(y_label=1)
        necrosis = self.gen_mask(y_label=2)

        Image.fromarray(tumor).save(os.path.join(save_folder, 'tumor.png'))
        Image.fromarray(stroma).save(os.path.join(save_folder, 'stroma.png'))
        Image.fromarray(necrosis).save(os.path.join(save_folder, 'necrosis.png'))

        coordinates = []
        if len(self.coordinates) == 0:
            coordinates = [self.gen_coords(tumor, 5),
                               self.gen_coords(stroma, 5),
                               self.gen_coords(necrosis, 5)]

            self.coordinates = coordinates


        name_list = ["存活肿瘤", "间质", "角化和坏死"]
        color_list = ['4294901760', '4278222976', '4294967040']

        self.draw_roiplot(coordinates, save_folder)

        root = ET.Element("Annotations", Unit="", Scale="1")

        num = 1
        for index in range(len(coordinates)):
            cur_coordinates = coordinates[index]
            cur_color = color_list[index]
            cur_name = name_list[index]

            for item in cur_coordinates:

                # scale = 10

                if len(item) in range(0, 20):
                    scale = "5"
                elif len(item) in range(20, 50):
                    scale = "10"
                elif len(item) in range(50, 100):
                    scale = "20"
                else:
                    scale = "40"

                annotation = ET.SubElement(root, "Annotation", DetailVisible="0", FontFamily="Arial", FontItalic="0",
                                           FontSize="12", Visible="-1", Width="2", Selected="0", Measurement="0",
                                           FontBold="0", Subtype="0", FontUnderline="0", Type=f"Pointn",
                                           Color=cur_color,
                                           GUID=f"-1")

                metadata = ET.SubElement(annotation, "Metadata", Scale=scale, ArcLength="-1", Path="",
                                         Name=f"标注 {num}",
                                         Detail=f"{cur_name}", Angle="-1", Radius="-1", Length="-1", Area="-1")
                num += 1
                for coord in item:
                    point = ET.SubElement(annotation, "P", X=str(coord['X']), Y=str(coord['Y']))

        tree = ET.ElementTree(root)
        xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
        xml_str = xml.dom.minidom.parseString(xml_str)
        pretty_xml = xml_str.toprettyxml()

        # 写入文件
        save_path = os.path.join(save_folder, 'notes')
        with open(save_path, "w", encoding='utf-8') as file:
            file.write(pretty_xml)

    def gen_asap(self, save_folder):
        tumor = self.gen_mask(y_label=0)
        stroma = self.gen_mask(y_label=1)
        necrosis = self.gen_mask(y_label=2)

        Image.fromarray(tumor).save(os.path.join(save_folder, 'tumor.png'))
        Image.fromarray(stroma).save(os.path.join(save_folder, 'stroma.png'))
        Image.fromarray(necrosis).save(os.path.join(save_folder, 'necrosis.png'))

        coordinates = []
        if len(self.coordinates) == 0:
            coordinates = [self.gen_coords(tumor, 5),
                           self.gen_coords(stroma, 5),
                           self.gen_coords(necrosis, 5)]

            self.coordinates = coordinates

        else:
            coordinates = self.coordinates

        # coordinates = [self.gen_coords(tumor, 5),
        #                self.gen_coords(stroma, 5),
        #                self.gen_coords(necrosis, 5)]

        name_list = ["tumor", "stroma", "necrosis"]
        color_list = ['#FF0000', '#00CC66', '#FFFF00']

        root = ET.Element("ASAP_Annotations")
        annotations = ET.SubElement(root, "Annotations")

        num = 1

        for index in range(len(coordinates)):
            cur_coordinates = coordinates[index]
            cur_color = color_list[index]
            cur_name = name_list[index]

            for item in cur_coordinates:
                annotation = ET.SubElement(annotations, "Annotation", Name=f"Annotation {num}", Type="Polygon",
                                           PartOfGroup=cur_name, Color=cur_color)
                coordinates_element = ET.SubElement(annotation, "Coordinates")
                num += 1
                for j in range(len(item)):
                    coord = item[j]
                    point = ET.SubElement(coordinates_element, "Coordinate", Order=str(j), X=str(coord['X']),
                                          Y=str(coord['Y']))

        annotation_groups = ET.SubElement(root, "AnnotationGroups")
        for index in range(len(name_list)):
            name = name_list[index]
            color = color_list[index]
            group = ET.SubElement(annotation_groups, "Group", Name=name, PartOfGroup="None", Color=color)
            attributes = ET.SubElement(group, "Attributes")

        xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')
        xml_str = xml.dom.minidom.parseString(xml_str)
        pretty_xml = xml_str.toprettyxml()

        # Write to the file
        save_path = os.path.join(save_folder, 'asap.xml')
        with open(save_path, "w") as file:
            file.write(pretty_xml)

