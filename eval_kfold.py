#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

import utils
from torchvision import transforms

@torch.no_grad()
def evaluate_fold(net: nn.Module, dl: DataLoader, num_classes: int):
    """对一个 DataLoader 做评估，返回各种指标"""
    net.eval()
    device = next(net.parameters()).device

    all_logits, all_labels = [], []

    for batch in dl:
        # 兼容 (x, y) / (x, y, path) / dict / (dict, y)
        if isinstance(batch, (list, tuple)):
            x_raw, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            # 纯 dict 的情况：自己从中取出 img 和 label
            x_raw = batch.get("img") or batch.get("image")
            y = batch.get("label")
        else:
            x_raw, y = batch

        # 把 x, y 搬到 device；x 可能是 tensor，也可能是 dict({'img_40x':..., ...})
        if isinstance(x_raw, dict):
            x = {k: v.to(device, non_blocking=True) for k, v in x_raw.items()}
        else:
            x = x_raw.to(device, non_blocking=True)

        y = y.to(device, non_blocking=True)

        logits = net(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = logits.argmax(axis=1)

    # ===== 基础指标 =====
    acc = accuracy_score(labels, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    prec_u, rec_u, f1_u, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    # ===== 特异性 / 敏感性（多分类：一对其余） =====
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)
    eps = 1e-12
    sens_per_class = tp / (tp + fn + eps)  # = recall / TPR
    spec_per_class = tn / (tn + fp + eps)  # = TNR
    support = cm.sum(axis=1).astype(float)

    sens_macro = float(np.mean(sens_per_class))
    spec_macro = float(np.mean(spec_per_class))
    sens_weighted = float(np.average(sens_per_class, weights=support))
    spec_weighted = float(np.average(spec_per_class, weights=support))

    metrics = {
        "acc": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "precision_micro": prec_u,
        "recall_micro": rec_u,
        "f1_micro": f1_u,
        "specificity_macro": spec_macro,
        "specificity_weighted": spec_weighted,
        "sensitivity_macro": sens_macro,
        "sensitivity_weighted": sens_weighted,
        "cm": cm,
        "sens_per_class": sens_per_class,
        "spec_per_class": spec_per_class,
        "support": support,
    }
    return metrics


def save_fold_metrics_csv(metrics: dict, save_path: str):
    row = {
        "accuracy": metrics["acc"],
        "prec_w": metrics["precision_weighted"],
        "rec_w": metrics["recall_weighted"],
        "f1_w": metrics["f1_weighted"],
        "prec_m": metrics["precision_macro"],
        "rec_m": metrics["recall_macro"],
        "f1_m": metrics["f1_macro"],
        "prec_u": metrics["precision_micro"],
        "rec_u": metrics["recall_micro"],
        "f1_u": metrics["f1_micro"],
        "spec_m": metrics["specificity_macro"],
        "spec_w": metrics["specificity_weighted"],
        "sens_m": metrics["sensitivity_macro"],
        "sens_w": metrics["sensitivity_weighted"],
    }
    pd.DataFrame([row]).to_csv(save_path, index=False)


def save_fold_confusion_csv(metrics: dict, save_path: str):
    cm = metrics["cm"]
    df_cm = pd.DataFrame(cm)
    df_cm.to_csv(save_path, index=False)


def main(args: Namespace):
    """
    使用【训练时的配置 folder5.json】和 KFold 划分，
    在每一折的“验证集”上评估已经训练好的 fold 权重。
    """
    # ==== 读“训练用”的 config（包含 data_path_train / num_folds 等） ====
    with open(args.config_path, "r") as f:
        cnn = json.load(f)

    num_classes     = cnn["num_classes"]
    data_path_train = cnn["data_path_train"]      # ✅ 训练时的 txt
    model_name      = cnn["model_name"]
    backbone_name   = cnn["backbone_name"]
    pretrained      = cnn["pretrained"].lower() == "true"
    freeze          = cnn["freeze"].lower() == "true"

    batch_size  = cnn["batch_size"]
    num_workers = cnn["num_workers"]
    k           = int(cnn["num_folds"])          # 和训练时一致

    ckpt_dir = args.ckpt_dir                     # exp?_k5 的目录

    # transforms：和训练里的 train/valid 保持一致
    transform = transforms.Compose([
        transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    ds_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    def _make_dataset(path, _tfm):
        if model_name.startswith("ensemble"):
            return utils.EnsembleWSIDataset(data_path=path, transform=_tfm)
        elif model_name.startswith("Multi"):
            return utils.BothWSIDataset(data_path=path, transform=_tfm)
        else:
            return utils.WSIDataset(data_path=path, transform=_tfm)

    # 用 train.txt 重新构造 dataset，并用 KFold 还原每一折的 train/val 划分
    ds_train_aug  = _make_dataset(data_path_train, transform)
    ds_eval_norm  = _make_dataset(data_path_train, ds_transform)

    n_samples = len(ds_train_aug)
    indices   = list(range(n_samples))

    splitter  = KFold(n_splits=k, shuffle=True, random_state=42)
    split_iter = splitter.split(indices)

    print(f"使用 num_folds={k}，只评估 KFold 的验证集，ckpt_dir = {ckpt_dir}")

    all_acc = []
    all_f1_macro = []

    for fold, (tr, va) in enumerate(split_iter, 1):
        print(f"\n===== Eval Fold {fold}/{k} (validation set) =====")

        dl_valid = DataLoader(
            Subset(ds_eval_norm, va),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

        # 构建模型并加载该折权重
        net_f = utils.choose_model(
            model_name, backbone_name, pretrained,
            num_classes=num_classes, freeze=freeze
        ).cuda()

        ckpt_path_f = os.path.join(ckpt_dir, f"{model_name}_fold{fold}.pt")
        print(f"Loading checkpoint: {ckpt_path_f}")
        state = torch.load(ckpt_path_f, map_location="cuda")
        net_f.load_state_dict(state)

        # 评估该折的“验证集”
        metrics = evaluate_fold(net_f, dl_valid, num_classes)
        print({
            "acc": metrics["acc"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "sens_macro": metrics["sensitivity_macro"],
            "spec_macro": metrics["specificity_macro"],
        })

        # 保存 CSV：就保存在 ckpt_dir 里
        metrics_csv = os.path.join(ckpt_dir, f"fold{fold}_metrics_val_eval.csv")
        cm_csv      = os.path.join(ckpt_dir, f"fold{fold}_confusion_val_eval.csv")
        save_fold_metrics_csv(metrics, metrics_csv)
        save_fold_confusion_csv(metrics, cm_csv)
        print(f"saved metrics -> {metrics_csv}")
        print(f"saved confusion -> {cm_csv}")

        all_acc.append(metrics["acc"])
        all_f1_macro.append(metrics["f1_macro"])

    # 汇总 K 折“验证集”的平均表现
    if all_f1_macro:
        mean_acc = float(np.mean(all_acc))
        std_acc  = float(np.std(all_acc))
        mean_f1  = float(np.mean(all_f1_macro))
        std_f1   = float(np.std(all_f1_macro))
        print("\n==== KFold validation summary (eval) ====")
        print(f"acc:      {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"

    args = Namespace(
        # ⚠ 这里要换成你“训练时做 5 折”的那份 config
        config_path="config/sl2model/MultiAttenTransModel/folder5.json",
        # ⚠ 这里是存有 MultiAttenTransModel_fold1~5.pt 的目录
        ckpt_dir="",
    )
    main(args)
