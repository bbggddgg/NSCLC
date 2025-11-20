#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 15:28
# @Author  : YangChenghan
# @File    : Loss.py
# @Description : 这个函数是用来balabalabala自己写

import torch
from torch.autograd import Variable
import torch.nn.functional as F

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num=3, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].to(predict.device)  # 确保alpha和预测在相同设备上
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class InfoNCELossCalculator:
    def __init__(self, args):
        self.args = args

    def __call__(self, features):
        labels = self._generate_labels()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def _generate_labels(self):
        batch_size = self.args.batch_size
        n_views = self.args.n_views

        labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        return labels