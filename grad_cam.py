#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/19 10:40
# @Author  : YangChenghan
# @File    : grad_cam.py
# @Description : 这个函数是用来balabalabala自己写
import os

from PIL import Image
from matplotlib import pyplot as plt, gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from model import *
from torch import nn
from torchvision import transforms
from torchvision import models
import cv2
import torch
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, input_tensor_40x, input_tensor_10x, target_layer):
        image_size = (input_tensor_40x.shape[2], input_tensor_40x.shape[3])
        self.model.zero_grad()

        x = {'40x': input_tensor_40x, '10x': input_tensor_10x}
        output, features_40x, features_10x = self.model(x)

        if target_layer == '40x':
            features = features_40x
        elif target_layer == '10x':
            features = features_10x

        features.register_hook(self.save_gradient)

        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(output.device)

        one_hot_output[0][output.argmax()] = 1
        print(f"one_hot_output: {one_hot_output}")
        # 确保output是标量
        output = (output * one_hot_output).sum()

        # 确保output需要计算梯度
        if not output.requires_grad:
            output.requires_grad_(True)

        output.backward()

        gradients = self.gradient[0].cpu().data.numpy()

        if gradients.ndim == 2:
            gradients = gradients.reshape(gradients.shape[1], -1, int(gradients.shape[0]**0.5))

        weights = np.mean(gradients, axis=(1, 2))
        features = features[0].cpu().data.numpy()

        if features.ndim == 2:
            features = features.reshape(features.shape[1], -1, int(features.shape[0]**0.5))

        cam = np.zeros(features.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * features[i, :, :]

        print(f"np.max(cam) {np.max(cam)}, np.min(cam) {np.min(cam)}")
        cam = cam - np.min(cam)  # 将最小值平移为0
        cam = cv2.resize(cam, image_size)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)  # 归一化到 [0, 1] 的范围内
        return cam


def show_cam_on_image(img, cam, use_rgb=True, save_path=None):
    img = np.array(img)
    # cam_float = cam.astype(np.float32)  # 确保是 float32 的归一化热图，值在 0~1
    cam_float = (cam * 255).astype(np.float32)
    cam_vis = np.uint8(cam_float)  # 用于生成 heatmap 显示
    heatmap = cv2.applyColorMap(cam_vis, cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    output_image = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[5, 0.5, 5, 0.4])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(img)
    ax0.set_aspect('equal')

    ax1 = plt.subplot(gs[2])
    ax1.imshow(output_image)
    ax1.set_aspect('equal')

    # ✅ 设置色温条位置和大小
    cax = inset_axes(
        ax1,
        width="6%",
        height="100%",
        loc='right',
        bbox_to_anchor=(0.2, 0., 1, 1),
        bbox_transform=ax1.transAxes,
        borderpad=0,
    )

    # ✅ 直接用 ScalarMappable 创建色温条，彻底避免 imshow 的透明图层问题
    norm = Normalize(vmin=0, vmax=255)
    sm = ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])  # 仅用于 colorbar，不显示图像
    plt.colorbar(sm, cax=cax)

    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.show()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"

    model_name = "MultiAttenTransModel"
    num_classes = 3
    ckpt_path = "/path/to/checkpoint.pt"
    model = choose_model(model_name, num_classes, ckpt_path)
    model = model.cuda()

    data_path ="/path/to/image.jpg"
    x40 = data_path
    x10 = data_path.replace('level0', 'level2')
    image40 = Image.open(x40).convert('RGB')
    image10 = Image.open(x10).convert('RGB')

    image40_tensor = transform(image40)
    image40_tensor = torch.unsqueeze(image40_tensor, dim=0)
    image40_tensor = image40_tensor.cuda()

    image10_tensor = transform(image10)
    image10_tensor = torch.unsqueeze(image10_tensor, dim=0)
    image10_tensor = image10_tensor.cuda()

    grad_cam = GradCAM(model)

    cam_40x = grad_cam(image40_tensor, image10_tensor, '40x')
    cam_10x = grad_cam(image40_tensor, image10_tensor, '10x')
    save_path = x40+".cam.png"
    show_cam_on_image(image40, cam_40x, use_rgb=True, save_path=save_path)

    save_path = x10 + ".cam.png"
    show_cam_on_image(image10, cam_10x, use_rgb=True, save_path=save_path)



