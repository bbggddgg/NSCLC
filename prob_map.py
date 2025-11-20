 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 18:39
# @Author  : YangChenghan
# @File    : prob_map.py
# @Description : 这个函数是用来balabalabala自己写


import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.data import DataLoader
from argparse import Namespace, ArgumentParser
import os
import json


# ---------------------------------------------
from model import *
from utils import *

class FeatureExtractorResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的resnet18模型
        self.resnet = models.resnet18(pretrained=False)
        # 移除resnet的最后两层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        return (x,)  # 返回一个元组

def choose_model(model_name, num_classes, ckpt_path):
    from mmselfsup.models import ResNet
    if model_name == 'signle_model':

        backbone = ResNet(depth=18)
        net = ResNet18(backbone, num_classes)

    elif model_name == "torch_model":
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif model_name == 'MultiAttenTransModel':
        backbone_40x = ResNet(depth=18)
        backbone_10x = ResNet(depth=18)
        net = MultiAttenTransModel(backbone_40x, backbone_10x)

    elif model_name == 'MultiAttenTransModel_torch':
        backbone_40x = FeatureExtractorResNet()
        backbone_10x = FeatureExtractorResNet()
        net = MultiAttenTransModel(backbone_40x, backbone_10x)

    elif model_name == 'MultiTransModel':
        backbone_40x = ResNet(depth=18)
        backbone_10x = ResNet(depth=18)
        net = MultiTransModel(backbone_40x, backbone_10x)

    elif model_name == 'MultiAttenModel':
        backbone_40x = ResNet(depth=18)
        backbone_10x = ResNet(depth=18)
        net = MultiAttenModel(backbone_40x, backbone_10x)

    elif model_name == 'MultiModel':
        backbone_40x = ResNet(depth=18)
        backbone_10x = ResNet(depth=18)
        net = MultiModel(backbone_40x, backbone_10x)

    elif model_name == 'ensemble_r50':
        net = ensemble_r50(num_classes=num_classes, pretrained=False)

    elif model_name == 'ensemble_r18':
        net = ensemble_r18(num_classes=num_classes, pretrained=False)

    elif model_name == 'msbp_model':
        net = resnet_msbp(exp_mode='ResNet_MSBP', nr_classes=num_classes, pretrained=False)

    elif model_name == 'msdnet_model':
        net = msdnet(pretrained=False, nr_class=num_classes)

    elif model_name == 'res2net50_model':
        net = res2net50_v1b(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        net = None

    net.load_state_dict(torch.load(ckpt_path))
    net = torch.nn.DataParallel(net)

    return net


def run(args):
    with open(args.config_path, 'r') as f:
        mapJson = json.load(f)

    model_name = mapJson['model']
    patches_path = mapJson['patches_path']
    ckpt_path = mapJson['ckpt_path']
    num_classes = mapJson['num_classes']
    save_csv_path = mapJson['save_csv_path']
    label_names = mapJson['label_names']
    # strategy = mapJson['strategy']
    net = choose_model(model_name, num_classes, ckpt_path)


    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if model_name.startswith("single"):
        ds_tissue = WSIPatchDataset(patches_path=patches_path, transform=transform)
    else:
        ds_tissue = MultiWSIPatchDataset(patches_path=patches_path, transform=transform)

    dl_tissue = DataLoader(ds_tissue, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    data_for_map(net, dl_tissue, save_csv_path, label_names)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    parser = ArgumentParser(description='Draw map')
    parser.add_argument('--config_path', default='config/map/22-00243-7-map.json', metavar='CNN_PATH', type=str,
                        help='Path to the config file in jsonformat')
    parser.add_argument('--batch_size', default=512, metavar='CNN_PATH', type=int,
                        help='Path to the config file in jsonformat')
    parser.add_argument('--num_workers', default=8, metavar='CNN_PATH', type=int,
                        help='Path to the config file in jsonformat')
    args = parser.parse_args()
    run(args)