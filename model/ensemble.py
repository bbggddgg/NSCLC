#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 19:21
# @Author  : YangChenghan
# @File    : ensemble_model.py
# @Description : 这个函数是用来balabalabala自己写
import os

import torch
import torch.nn as nn
from torchvision import models

class ensemble_r50(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ensemble_r50, self).__init__()
        # Load pre-trained ResNet50 models
        self.resnet_backbone_40x = models.resnet50(pretrained=pretrained)
        self.resnet_backbone_20x = models.resnet50(pretrained=pretrained)
        self.resnet_backbone_10x = models.resnet50(pretrained=pretrained)

        # Remove avgpool and fc
        self.resnet_backbone_40x = nn.Sequential(*list(self.resnet_backbone_40x.children())[:-2])
        self.resnet_backbone_20x = nn.Sequential(*list(self.resnet_backbone_20x.children())[:-2])
        self.resnet_backbone_10x = nn.Sequential(*list(self.resnet_backbone_10x.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3 * 2048, num_classes)

    def forward(self, x):
        features_40x = self.resnet_backbone_40x(x['40x'])
        features_20x = self.resnet_backbone_20x(x['20x'])
        features_10x = self.resnet_backbone_10x(x['10x'])
        fused_features = torch.cat([features_40x, features_20x, features_10x], dim=1)
        fused_features = self.avgpool(fused_features)
        fused_features = torch.flatten(fused_features, 1)
        fused_features = self.classifier(fused_features)

        return fused_features

class ensemble_r18(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ensemble_r18, self).__init__()
        # Load pre-trained ResNet50 models
        self.resnet_backbone_40x = models.resnet18(pretrained=pretrained)
        self.resnet_backbone_20x = models.resnet18(pretrained=pretrained)
        self.resnet_backbone_10x = models.resnet18(pretrained=pretrained)

        # Remove avgpool and fc
        self.resnet_backbone_40x = nn.Sequential(*list(self.resnet_backbone_40x.children())[:-2])
        self.resnet_backbone_20x = nn.Sequential(*list(self.resnet_backbone_20x.children())[:-2])
        self.resnet_backbone_10x = nn.Sequential(*list(self.resnet_backbone_10x.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3 * 512, num_classes)

    def forward(self, x):
        features_40x = self.resnet_backbone_40x(x['40x'])
        features_20x = self.resnet_backbone_20x(x['20x'])
        features_10x = self.resnet_backbone_10x(x['10x'])
        fused_features = torch.cat([features_40x, features_20x, features_10x], dim=1)
        fused_features = self.avgpool(fused_features)
        fused_features = torch.flatten(fused_features, 1)
        fused_features = self.classifier(fused_features)

        return fused_features


# if __name__ == '__main__':
#     os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#
#     model = ensemble_r18(num_classes=3)
#     print(model)
#     image40x = torch.randn(16, 3, 256, 256)
#     image20x = torch.randn(16, 3, 256, 256)
#     image10x = torch.randn(16, 3, 256, 256)
#
#     data = {'40x':image40x, '20x': image20x, '10x': image10x}
#
#     model.eval()
#     output = model(data)
#     print(output.shape)
#     print(output)
