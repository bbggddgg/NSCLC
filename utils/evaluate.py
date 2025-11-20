#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/10 14:18
# @Author  : YangChenghan
# @File    : evaluate.py
# @Description : 这个函数是用来生成评估集所需数据

import pandas as pd
import torch
from tqdm import tqdm


def data_for_compare(model, dataloader, save_csv_path, label_names):
    pred_list = []
    true_list = []
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for images, y in tqdm(dataloader, desc="Model is predicting, please wait"):
            # 将数据转到GPU
            if isinstance(images, dict):
                images = {key: images[key].cuda() for key in images}
            else:
                images = images.cuda()
            # print(y.tolist())
            # 将图片传入到模型当中就，得到预测的值pred
            preds = model(images)
            pred_softmax = torch.softmax(preds,1).cpu().numpy()

            pred_list.extend(pred_softmax.tolist())
            true_list.extend(y.tolist())

    df_pred = pd.DataFrame(data=pred_list, columns=label_names)
    df_pred['label'] = true_list
    df_pred.to_csv(save_csv_path, encoding='gbk', index=False)

def data_for_map(model, dataloader, save_csv_path, label_names):
    x_point_list = []
    y_point_list = []
    label_list = []
    pred_list = []
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for images, x_list, y_list in tqdm(dataloader):

            if isinstance(images, dict):
                images = {key: images[key].cuda() for key in images}
            else:
                images = images.cuda()
            # images = images.cuda()
            preds = model(images)

            pred_softmax = torch.softmax(preds, 1).cpu().numpy()
            pred_list.extend(pred_softmax.tolist())
            # max_pred = torch.max(preds, dim=1).cpu().numpy()
            # labels = torch.argmax(preds, dim=1).cpu().numpy()

            x_points = [int(x) for x in x_list]
            y_points = [int(y) for y in y_list]

            x_point_list.extend(x_points)
            y_point_list.extend(y_points)

    df_pred = pd.DataFrame(data=pred_list, columns=label_names)
    df_pred['x'] = x_point_list
    df_pred['y'] = y_point_list
    # 保存为CSV文件
    df_pred.to_csv(save_csv_path, encoding='gbk', index=False)


