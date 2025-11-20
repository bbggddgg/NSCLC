#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/18 17:15
# @Author  : YangChenghan
# @File    : gen_roi.py
# @Description : 这个函数是用来balabalabala自己写
import shutil

import xmltodict
import json
import os
import numpy as np
import xml.etree.ElementTree as ET

class ROI:
    def __init__(self, notePath, jsonData=None):
        self.jsonData = jsonData
        self.notePath = notePath
        self.folderPath = os.path.split(notePath)[0]
        self.labeldict = {
            '4294901760': 'viable_tumor',
            '4294967040': 'necrosis_and_keratinize',
            '4278222976': 'stroma'
        }
        # self.colorDict = dict()
        self.areaDict = dict()
        self.roiDict = dict()
        if self.jsonData is None:
            self.readnote()
        else:
            with open(self.jsonData, "r") as f:
                self.jsonData = json.load(f)

    def readnote(self):
        with open(self.notePath, 'r', encoding='utf-8') as xml_file:
            xml_data = xml_file.read()
        self.jsonData = xmltodict.parse(xml_data)
        with open(os.path.join(self.folderPath, 'notes.json'), 'w', encoding='utf-8') as json_file:
            json.dump(self.jsonData, json_file, indent=4)

    def getArea(self):
        for item in self.jsonData['Annotations']['Annotation']:
            label = item['@Color']
            label_name = self.labeldict[label]

            if label_name not in self.areaDict:
                self.areaDict[label_name] = []
                self.roiDict[label_name] = []

            self.areaDict[label_name].append(item)

        for area in self.areaDict:
            areaPath = os.path.join(self.folderPath, area)

            if not os.path.exists(areaPath):
                os.makedirs(areaPath)
            else:
                shutil.rmtree(areaPath)
                os.makedirs(areaPath)

            for item in self.areaDict[area]:
                metadata = item['Metadata']
                name = metadata['@Name']
                vertices = [(int(item["@X"]), int(item["@Y"])) for item in item["P"]]

                self.roiDict[area].append({'name': name, 'vertices': vertices})

            with open(os.path.join(areaPath, 'data.json'), 'w', encoding='utf-8') as json_file:
                json.dump(self.roiDict[area], json_file, indent=4)

        with open(os.path.join(self.folderPath, 'ROIs.json'), 'w', encoding='utf-8') as json_file:
            json.dump(self.roiDict, json_file, indent=4)



