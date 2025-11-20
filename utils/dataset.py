import os
import sys
import numpy as np
import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.dataset import T_co
from torchvision import transforms  # noqa
from tqdm import tqdm

np.random.seed(42)

# 单倍镜 40x
class WSIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(WSIDataset, self).__init__()

        self.data_path = data_path
        self.transform = transform
        self.pre_process()

    def pre_process(self):

        self.items = pd.read_csv(self.data_path, header=None, sep=' ')
        self.num_images = len(self.items)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        path, label = self.items.iloc[idx]
        image = Image.open(path).convert('RGB')
        label = np.array(label, dtype=int)
        label = int(label)
        try:
            if self.transform:
                image = self.transform(image)
        except:
            print(f'error image:{path}')
        return image, label

# 三倍镜 40x 20x 10x
class EnsembleWSIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(EnsembleWSIDataset, self).__init__()

        self.data_path = data_path
        self.transform = transform
        self.pre_process()

    def pre_process(self):

        self.items = pd.read_csv(self.data_path, header=None, sep=' ')
        x40_column = self.items.iloc[:, 0]
        x20_column = x40_column.str.replace('level0', 'level1')
        x10_column = x40_column.str.replace('level0', 'level2')
        self.items.insert(1, "x20", x20_column)
        self.items.insert(2, "x10", x10_column)

        self.num_images = len(self.items)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        x40, x20, x10, label = self.items.iloc[idx]
        # x10 = x40.replace('level0', 'level2')
        image40 = None
        image20 = None
        image10 = None
        try:
            image40 = Image.open(x40).convert('RGB')
            image20 = Image.open(x20).convert('RGB')
            image10 = Image.open(x10).convert('RGB')
            label = np.array(label, dtype=int)
            label = int(label)
            if self.transform:
                image40 = self.transform(image40)
                image20 = self.transform(image20)
                image10 = self.transform(image10)
        except:
            print(f'error image:{x40}')
        # return x40, x10, label
        return {'40x': image40, '20x': image20, '10x': image10}, label

# 10x + 40x
class BothWSIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(BothWSIDataset, self).__init__()

        self.data_path = data_path
        self.transform = transform
        self.pre_process()

    def pre_process(self):

        self.items = pd.read_csv(self.data_path, header=None, sep=' ')
        x40_column = self.items.iloc[:, 0]
        x10_column = x40_column.str.replace('level0', 'level2')
        self.items.insert(1, "x10", x10_column)

        self.num_images = len(self.items)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        x40, x10, label = self.items.iloc[idx]
        # x10 = x40.replace('level0', 'level2')
        image40 = None
        image10 = None
        try:
            image40 = Image.open(x40).convert('RGB')
            image10 = Image.open(x10).convert('RGB')
            label = np.array(label, dtype=int)
            label = int(label)
            if self.transform:
                image40 = self.transform(image40)
                image10 = self.transform(image10)
        except:
            print(f'error image:{x40}')
        # return x40, x10, label
        return {'40x': image40, '10x': image10}, label

# 20x + 40x
class Both2WSIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(Both2WSIDataset, self).__init__()

        self.data_path = data_path
        self.transform = transform
        self.pre_process()

    def pre_process(self):

        self.items = pd.read_csv(self.data_path, header=None, sep=' ')
        x40_column = self.items.iloc[:, 0]
        x10_column = x40_column.str.replace('level0', 'level1')
        self.items.insert(1, "x10", x10_column)

        self.num_images = len(self.items)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        x40, x10, label = self.items.iloc[idx]
        # x10 = x40.replace('level0', 'level2')
        image40 = None
        image10 = None
        try:
            image40 = Image.open(x40).convert('RGB')
            image10 = Image.open(x10).convert('RGB')
            label = np.array(label, dtype=int)
            label = int(label)
            if self.transform:
                image40 = self.transform(image40)
                image10 = self.transform(image10)
        except:
            print(f'error image:{x40}')
        # return x40, x10, label
        return {'40x': image40, '10x': image10}, label

# 单倍镜预测使用
class WSIPatchDataset(Dataset):

    def __init__(self,patches_path, transform=None):
        self.patches_path = patches_path
        self.transform = transform
        self.pre_process()

    def pre_process(self):
        self.items = []
        for filename in os.listdir(self.patches_path):
            path = os.path.join(self.patches_path, filename)
            filename = filename.split('.')[0]
            x, y = filename.split('_')[-2:]
            self.items.append((path, x, y))
            self.num_images = len(self.items)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        path, x, y = self.items[idx]
        # print(f"filename:{path}, point({x}, {y})")
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, x, y

# 多倍镜预测使用
class MultiWSIPatchDataset(Dataset):

    def __init__(self,patches_path, transform=None):
        self.patches_path = patches_path
        self.transform = transform
        self.pre_process()

    def pre_process(self):
        self.items = []
        for filename in os.listdir(self.patches_path):
            path_40x = os.path.join(self.patches_path, filename)
            path_10x = path_40x.replace('level0', 'level2')
            filename = filename.split('.')[0]
            x, y = filename.split('_')[-2:]
            self.items.append((path_40x, path_10x, x, y))
            self.num_images = len(self.items)


    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        path_40x, path_10x, x, y = self.items[idx]
        # path_10x = path_40x.replace('level0', 'level2')
        image40 = Image.open(path_40x).convert('RGB')
        image10 = Image.open(path_10x).convert('RGB')

        if self.transform:
            image40 = self.transform(image40)
            image10 = self.transform(image10)

        return {'40x': image40, '10x': image10}, x, y
