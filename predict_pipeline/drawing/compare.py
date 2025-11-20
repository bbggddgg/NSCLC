#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 18:21
# @Author  : YangChenghan
# @File    : compare.py
# @Description : 这个函数是用来绘制模型比较各图表的
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
def norm(true_label, predict_label):
    '''
        常用指标：精度，查准率，召回率，F1-Score
    '''
    # 精度，准确率， 预测正确的占所有样本种的比例
    accuracy = accuracy_score(true_label, predict_label)
    print("精度: ", accuracy)

    def get_norm(average):
        precision = precision_score(true_label, predict_label, average=average)
        recall = recall_score(true_label, predict_label, average=average)
        f1 = f1_score(true_label, predict_label, average=average)

        return precision, recall, f1

    Weighted_precision, Weighted_recall, Weighted_f1_score = get_norm('weighted')
    Macro_precision, Macro_recall, Macro_f1_score = get_norm('macro')
    Micro_precision, Micro_recall, Micro_f1_score = get_norm('micro')


    print('------Weighted------')
    print('Weighted precision', Weighted_precision)
    print('Weighted recall', Weighted_recall)
    print('Weighted f1-score', Weighted_f1_score)

    print('------Macro------')
    print('Macro precision', Macro_precision)
    print('Macro recall', Macro_recall)
    print('Macro f1-score', Macro_f1_score)

    print('------Micro------')
    print('Micro precision', Micro_precision)
    print('Micro recall', Micro_recall)
    print('Micro f1-score', Micro_f1_score)

    df = pd.DataFrame({
        "accuracy": accuracy,
        'Weighted precision': Weighted_precision,
        'Weighted recall': Weighted_recall,
        'Weighted f1-score': Weighted_f1_score,
        'Macro precision': Macro_precision,
        'Macro recall': Macro_recall,
        'Macro f1-score': Macro_f1_score,
        'Micro precision': Micro_precision,
        'Micro recall': Micro_recall,
        'Micro f1-score': Micro_f1_score,
    }, index=[0])

    return df


def confusionMatrix(title, label_names, true_label, predict_label):


    confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])
    plt.rcParams['font.size'] = 14  # 设置全局字体大小
    plt.figure(figsize=(12, 8))
    plt.imshow(confusion, cmap=plt.cm.Blues)  # 使用imshow替代matshow
    plt.colorbar()
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            plt.annotate(confusion[j, i], xy=(i, j), ha='center', va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(len(label_names)), label_names)
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.title(title + "  Confusion Matrix")
    return plt
    # plt.tight_layout()  # 使用tight_layout调整布局
    # plt.show()



def ROC(title, label_names, true_label, predict_data):
    n_classes = len(label_names)
    binarize_predict = to_categorical(true_label, num_classes=n_classes, dtype="int")
    predict_score = predict_data.to_numpy()
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(n_classes-1)
    for i in range(n_classes):
        print(i)
        fpr[i], tpr[i], _ = roc_curve(binarize_predict[:, i], [score_i[i] for score_i in predict_score])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])

    # print("roc_auc = ",roc_auc)

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2

    plt.rcParams['font.size'] = 14  # 设置全局字体大小
    plt.figure(figsize=(12, 8))

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.4f})'.format(label_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' Multi-class receiver operating characteristic ')
    plt.legend(loc="lower right")
    # plt.show()
    return plt


def Model_ROC(names, target_loc, predict_locs, colors, pos_label=1):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 14  # 设置全局字体大小
    # 读取目标标签数据
    target_data = pd.read_csv(target_loc, sep=" ", names=["loc", "type"])
    true_label = [i for i in target_data["type"]]

    for (name, predict_loc, colorname) in zip(names, predict_locs, colors):
        predict_data = pd.read_csv(predict_loc)
        predict_score = predict_data.iloc[:, pos_label].to_numpy()

        print(true_label)
        print(predict_score)
        # 计算 ROC 曲线的数据
        fpr, tpr, thresholds = roc_curve(true_label, predict_score, pos_label=pos_label)

        # 绘制 ROC 曲线
        plt.plot(fpr, tpr, lw=3, label='{} (AUC={:.4f})'.format(name, auc(fpr, tpr)), color=colorname)

        # 绘制对角虚线
        plt.plot([0, 1], [0, 1], '--', lw=3, color='grey')

        # 设置坐标轴范围和刻度标签
        # plt.axis('scaled')
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # 设置坐标轴标签和标题
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC Curve', fontsize=25)

        # 设置图例的位置和字体大小
        plt.legend(loc='lower right', fontsize=15)

        # 调整子图的宽度
        # ax = plt.gca()
        # ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # 调整布局和外边距
        # plt.tight_layout()

    return plt

def Model_ROC_with_inset(names, target_loc, predict_locs, colors, pos_label=1, zoom_in=True):
    fig, ax = plt.subplots(figsize=(12, 8))

    # 读取目标标签数据
    target_data = pd.read_csv(target_loc, sep=" ", names=["loc", "type"])
    true_label = [i for i in target_data["type"]]

    for (name, predict_loc, colorname) in zip(names, predict_locs, colors):
        predict_data = pd.read_csv(predict_loc)
        predict_score = predict_data.iloc[:, pos_label].to_numpy()

        # 计算 ROC 曲线的数据
        fpr, tpr, thresholds = roc_curve(true_label, predict_score, pos_label=pos_label)

        # 绘制 ROC 曲线
        ax.plot(fpr, tpr, lw=3, label='{:15} (AUC = {:.4f})'.format(name, auc(fpr, tpr)), color=colorname)

    # 设置坐标轴范围和刻度标签
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if zoom_in:
        # 创建Zoom In图
        axins = ax.inset_axes([0.2, 0.6, 0.3, 0.2])
        axins.set_xlim(0.0, 0.2)
        axins.set_ylim(0.9, 1.0)
        for (name, predict_loc, colorname) in zip(names, predict_locs, colors):
            predict_data = pd.read_csv(predict_loc)
            predict_score = predict_data.iloc[:, pos_label].to_numpy()
            fpr, tpr, thresholds = roc_curve(true_label, predict_score, pos_label=pos_label)
            axins.plot(fpr, tpr, lw=2, color=colorname)

        # 创建虚线框
        rect = Rectangle((0.0, 0.9), 0.2, 0.1, fill=None, alpha=1, color='gray', linestyle='dashed')
        ax.add_patch(rect)

        # 连接Zoom In图和虚线框
        con1 = ConnectionPatch(xyA=(0.0, 0.9), xyB=(0.0, 0.9), coordsA="data", coordsB="data",
                               axesA=axins, axesB=ax, color="gray", linestyle='dashed')
        con2 = ConnectionPatch(xyA=(0.2, 1.0), xyB=(0.2, 1.0), coordsA="data", coordsB="data",
                               axesA=axins, axesB=ax, color="gray", linestyle='dashed')
        ax.add_artist(con1)
        ax.add_artist(con2)

    # 绘制对角虚线
    ax.plot([0, 1], [0, 1], '--', lw=3, color='grey')

    # 设置坐标轴标签和标题
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_title('ROC curve on Testing Dataset', fontsize=25)

    # 设置刻度标签的字体大小
    ax.tick_params(axis='both', which='major', labelsize=15)

    # 设置图例的位置和字体大小
    ax.legend(loc='lower right', fontsize=15)

    # plt.show()
    return plt




