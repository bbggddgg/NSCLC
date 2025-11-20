import torch
import wandb
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from argparse import Namespace, ArgumentParser
# from loading import WSIDataset
from sklearn.model_selection import KFold
import os
import json
import torchmetrics

#------------------------------------
import utils
import torchkeras
from torchkeras.kerascallbacks import WandbCallback

utils.seed_everything(42)

import numpy as np
import torch
import wandb
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from argparse import Namespace, ArgumentParser
from sklearn.model_selection import KFold
import os
import json
import torchmetrics

import utils
import torchkeras
from torchkeras.kerascallbacks import WandbCallback

utils.seed_everything(42)

transform = transforms.Compose([
    transforms.ColorJitter(64.0/255, 0.75, 0.25,  0.04),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ds_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def compute_metrics(true_label, pred_label, num_classes):
    # 计算准确度
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    accuracy_score = accuracy(torch.tensor(pred_label), torch.tensor(true_label))

    # 计算 F1-score
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
    f1_score = f1(torch.tensor(pred_label), torch.tensor(true_label))

    # 计算敏感度（Recall）
    sensitivity = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro')
    sensitivity_score = sensitivity(torch.tensor(pred_label), torch.tensor(true_label))

    # 计算特异性（Specificity）
    specificity = torchmetrics.Specificity(task='multiclass', num_classes=num_classes, average='macro')
    specificity_score = specificity(torch.tensor(pred_label), torch.tensor(true_label))

    return accuracy_score.item(), f1_score.item(), sensitivity_score.item(), specificity_score.item()

def run(args):
    with open(args.config_path, 'r') as f:
        cnn = json.load(f)

    num_classes = cnn['num_classes']  # 分类数
    norm = cnn['norm']  # 评价指标
    batch_size = cnn['batch_size']
    sampler_method = cnn['sampler']  # 未使用
    lr = cnn['lr']
    data_path_train = cnn['data_path_train']
    data_path_valid = cnn['data_path_valid']
    epochs = cnn['epochs']
    save_ckpt_folder_path = cnn['save_ckpt_folder_path']
    num_workers = cnn['num_workers']
    decayRate = cnn['decayRate']
    num_folds = cnn['num_folds']
    patience = cnn['patience']
    use_wandb = cnn['use_wandb']
    warmup = cnn['warmup']
    pretrained = cnn['pretrained'].lower() == 'true'
    freeze = cnn['freeze'].lower() == 'true'
    model_name = cnn['model_name']
    backbone_name = cnn['backbone_name']

    net = utils.choose_model(
        model_name, backbone_name, pretrained, num_classes=num_classes, freeze=freeze
    )
    net = torch.nn.DataParallel(net)

    # 损失函数选择 + 包装（确保输入为 logits Tensor）
    if cnn['loss_fn'] == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    elif cnn['loss_fn'] == 'MultiCEFocalLoss':
        loss_fn = utils.MultiCEFocalLoss(class_num=num_classes)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # optim选择
    if cnn['optim'] == 'Adam':
        optim = torch.optim.Adam(net.parameters(), lr)
    else:
        optim = torch.optim.SGD(net.parameters(), lr, momentum=0.8)

    # scheduler选择
    if cnn['scheduler'] == 'ExponentialLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decayRate)
    elif cnn['scheduler'] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, gamma=0.1, last_epoch=-1)
    elif cnn['scheduler'] == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='max', factor=0.1, patience=5, verbose=True)
    elif cnn['scheduler'] == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.001, max_lr=0.01, cycle_momentum=False)
    else:
        lr_scheduler = None

    # 指标 + 监控字段（torchmetrics 会接收 logits/probs，内部做 argmax）
    if norm == "f1-score":
        metrics_dict = {"f1-score": torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')}
        monitor_name, monitor_mode = 'val_f1-score', 'max'
    else:
        metrics_dict = {"acc": torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)}
        monitor_name, monitor_mode = 'val_acc', 'max'

    model = torchkeras.KerasModel(
        net, loss_fn=loss_fn, optimizer=optim, lr_scheduler=lr_scheduler, metrics_dict=metrics_dict
    )

    print(f"模型 {model_name} 结构如下：")
    print(model)

    # wandb（非CV通用 callback；KFold 时每折单独创建）
    if use_wandb == 'True':
        config = Namespace(batch_size=batch_size, lr=lr)
        project = cnn['project']
        wandb_cb = WandbCallback(project=project,
                                 config=config.__dict__,
                                 name=None,
                                 save_code=True,
                                 save_ckpt=False)
        wandb.login()
    else:
        wandb_cb = None

    shuffle = True

    # 使用 KFold 交叉验证
    print(f"使用 {num_folds} 折 KFold 交叉验证")
    k = int(num_folds)

    # 数据集类型与 transform（训练增强 / 验证标准化）
    def _make_dataset(path, _tfm):
        if model_name.startswith("ensemble"):
            return utils.EnsembleWSIDataset(data_path=path, transform=_tfm)
        elif model_name.startswith("Multi"):
            return utils.BothWSIDataset(data_path=path, transform=_tfm)
        else:
            return utils.WSIDataset(data_path=path, transform=_tfm)

    ds_train_aug = _make_dataset(data_path_train, transform)
    ds_eval_norm = _make_dataset(data_path_train, ds_transform)

    n_samples = len(ds_train_aug)
    indices = list(range(n_samples))

    splitter = KFold(n_splits=k, shuffle=True, random_state=42)
    split_iter = splitter.split(indices)

    # 结果目录
    file_num = len(os.listdir(save_ckpt_folder_path))
    exp_dir = os.path.join(save_ckpt_folder_path, f"exp{file_num}_k{k}")
    os.makedirs(exp_dir, exist_ok=True)

    fold_scores = []

    for fold, (tr, va) in enumerate(split_iter, 1):
        print(f"\n===== Fold {fold}/{k} =====")

        dl_train = DataLoader(Subset(ds_train_aug, tr), batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)
        dl_valid = DataLoader(Subset(ds_eval_norm, va), batch_size=batch_size,
                              num_workers=num_workers, shuffle=False)

        # —— 每折重建：模型/损失/优化器/调度器/指标（前向统一 logits）——
        net_f = utils.choose_model(
            model_name, backbone_name, pretrained, num_classes=num_classes, freeze=freeze
        )
        net_f = torch.nn.DataParallel(net_f)

        if cnn['loss_fn'] == 'CrossEntropyLoss':
            loss_fn_f = nn.CrossEntropyLoss()
        elif cnn['loss_fn'] == 'MultiCEFocalLoss':
            loss_fn_f = utils.MultiCEFocalLoss(class_num=num_classes)
        else:
            loss_fn_f = nn.CrossEntropyLoss()

        optim_f = torch.optim.Adam(net_f.parameters(), lr)

        lr_scheduler_f = torch.optim.lr_scheduler.StepLR(optim_f, 10, gamma=0.1, last_epoch=-1)

        metrics_dict_f = {"acc": torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)}

        model_f = torchkeras.KerasModel(
            net_f, loss_fn=loss_fn_f, optimizer=optim_f,
            lr_scheduler=lr_scheduler_f, metrics_dict=metrics_dict_f
        )

        if use_wandb == 'True':
            config_f = Namespace(batch_size=batch_size, lr=lr, fold=fold, n_splits=k)
            wandb_cb_f = WandbCallback(project=cnn['project'],
                                       config=config_f.__dict__,
                                       name=f"{model_name}-fold{fold}",
                                       save_code=True,
                                       save_ckpt=False)
            wandb.login()
        else:
            wandb_cb_f = None

        ckpt_path_f = os.path.join(exp_dir, f"{model_name}_fold{fold}.pt")

        model_f.fit(
            train_data=dl_train,
            val_data=dl_valid,
            epochs=epochs,
            patience=patience,
            ckpt_path=ckpt_path_f,
            monitor=monitor_name,
            mode=monitor_mode,
            callbacks=[wandb_cb_f],
            save_best_only=True,
            warmup=True if warmup == 'True' else False
        )

        # 每折结束打印该折最佳指标
        true_label = [label for label in va]
        pred_label = model_f.predict(dl_valid)  # 此处使用模型对验证集进行预测
        accuracy, f1, sensitivity, specificity = compute_metrics(true_label, pred_label, num_classes)

        print(f"[Fold {fold}] Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, Sensitivity = {sensitivity:.4f}, Specificity = {specificity:.4f}")

        fold_scores.append(f1)

    # 最终汇总
    if len(fold_scores) > 0:
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"\nFinal KFold results - F1 mean = {mean_score:.4f}, std = {std_score:.4f}")



if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    parser = ArgumentParser(description='Train model')
    parser.add_argument('--config_path', default='./config/sl2model/MultiAttenTransModel/test_exp0.json', metavar='CNN_PATH', type=str,
                        help='Path to the config file in jsonformat')
    args = parser.parse_args()
    run(args)
