#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 15:10
# @Author  : YangChenghan
# @File    : test.py
# @Description : 这个函数是用来balabalabala自己写

import os

import torch
from torch import nn
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import json

# ---------------------------------------------
import pandas as pd
from model import *
import utils


class FeatureExtractorResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的resnet18模型
        self.resnet = models.resnet18(pretrained=False)
        # 移除resnet的最后两层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        return (x,)  # 返回一个元组，兼容 MultiAttenTransModel 接口


def ROC_Matrix(eval_csv, title, label_names, save_folder):

    eval_df = pd.read_csv(eval_csv)

    true_label = eval_df['label'].tolist()
    pred_data = eval_df.iloc[:, :len(label_names)]
    pred_label = pred_data.to_numpy().argmax(axis=1).tolist()
    print(true_label)
    print(pred_label)
    nm = utils.norm(true_label, pred_label)
    nm.to_csv(os.path.join(save_folder, title + ' norm2.csv'))
    # 混淆矩阵
    matrix = utils.confusionMatrix(title, label_names, true_label, pred_label)
    matrix.savefig(os.path.join(save_folder, title + ' matrix2.png'), transparent=True)
    matrix.show()
    # 如果需要 ROC 再打开下面两行
    # roc = utils.ROC(title, label_names, true_label, pred_data)
    # roc.savefig(os.path.join(save_folder, title + ' ROC.png'), transparent=True)


# ========= 关键兼容函数：把训练时的 state_dict 映射回测试模型 =========
def load_ckpt_compat(net: nn.Module, ckpt_path: str) -> nn.Module:
    """
    兼容从 torchkeras.KerasModel + LogitsOnly + DataParallel 训练得到的权重：
    - 去掉 'module.' 前缀
    - 去掉 'backbone.' 前缀（训练时 LogitsOnly.backbone 里包着真正模型）
    - 兼容可能的外层 dict: state_dict / model / model_state_dict 等
    """
    state = torch.load(ckpt_path, map_location="cpu")

    # 可能是 {'state_dict': {...}} 这种形式，展开一下
    if isinstance(state, dict):
        for key in ["state_dict", "model", "model_state", "model_state_dict"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format in {ckpt_path}, got type: {type(state)}")

    new_state = {}
    for k, v in state.items():
        new_k = k

        # 去掉 DataParallel 的前缀
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        # 去掉 LogitsOnly.backbone 的前缀
        if new_k.startswith("backbone."):
            new_k = new_k[len("backbone."):]

        new_state[new_k] = v

    missing, unexpected = net.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[load_ckpt_compat] Warning: missing {len(missing)} keys, e.g.: {missing[:5]}")
    if unexpected:
        print(f"[load_ckpt_compat] Warning: unexpected {len(unexpected)} keys, e.g.: {unexpected[:5]}")

    return net


def choose_model(model_name, num_classes, ckpt_path):
    from mmselfsup.models import ResNet

    # ========= 按名称构建结构（保持你原来的写法）=========
    if model_name == 'signle_model':
        backbone = ResNet(depth=18)
        net = ResNet18(backbone, num_classes)

    elif model_name == "torch_model":
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif model_name == "resnet50_model":
        net = models.resnet50(pretrained=False)
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
        raise ValueError(f"Unknown model_name: {model_name}")

    # ========= 关键：用兼容函数加载 ckpt，然后再 DataParallel =========
    # net = load_ckpt_compat(net, ckpt_path)
    # net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(ckpt_path))
    net = torch.nn.DataParallel(net)

    return net


def run(args):
    with open(args.config_path, 'r') as f:
        evalJson = json.load(f)

    model_name = evalJson['model']
    data_path = evalJson['data_path']
    ckpt_path = evalJson['ckpt_path']
    label_names = evalJson['label_names']
    num_classes = evalJson['num_classes']
    save_csv_path = evalJson['save_csv_path']
    # strategy = evalJson['strategy']

    net = choose_model(model_name, num_classes, ckpt_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    if model_name.startswith("ensemble"):
        print("多倍镜数据")
        ds_eval = utils.EnsembleWSIDataset(data_path=data_path, transform=transform)
    elif model_name.startswith("Multi"):
        print("10倍和40倍数据")
        ds_eval = utils.BothWSIDataset(data_path=data_path, transform=transform)
    else:
        ds_eval = utils.WSIDataset(data_path=data_path, transform=transform)

    dl_eval = DataLoader(ds_eval,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=False)

    utils.data_for_compare(model=net,
                           dataloader=dl_eval,
                           save_csv_path=save_csv_path,
                           label_names=label_names)

    title = os.path.split(save_csv_path)[-1]
    save_folder = os.path.split(save_csv_path)[0]
    ROC_Matrix(save_csv_path, title, label_names, save_folder)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    parser = ArgumentParser(description='Eval model')
    parser.add_argument('--config_path',
                        default='./config/sl2model/MultiAttenTransModel/exp0.json',
                        metavar='CNN_PATH',
                        type=str,
                        help='Path to the config file in jsonformat')
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='batch size for eval')
    parser.add_argument('--num_workers',
                        default=8,
                        type=int,
                        help='num_workers for dataloader')
    args = parser.parse_args()
    run(args)
