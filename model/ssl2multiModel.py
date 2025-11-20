import torch
from torch import nn
from torchvision import models
from .transformer import Transformer

# 多倍镜多尺度融合 + transformer层
class MultiAttenTransModel(nn.Module):
    def __init__(self, resnet_backbone_40x, resnet_backbone_10x, num_feats=512, output_class=3, num_heads=8):
        super(MultiAttenTransModel, self).__init__()
        self.resnet_backbone_40x = resnet_backbone_40x
        self.resnet_backbone_10x = resnet_backbone_10x
        self.num_feats = num_feats
        self.output_class = output_class
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 多通道空间注意力的卷积层
        self.spatial_attention = nn.MultiheadAttention(embed_dim=num_feats, num_heads=num_feats//num_heads, dropout=0.0)
        self.trans = Transformer(num_classes=output_class, input_dim=num_feats * 2)

    def forward(self, x):
        # 分别通过 ResNet 骨干提取特征
        features_40x = self.resnet_backbone_40x(x['40x'])[0]
        features_10x = self.resnet_backbone_10x(x['10x'])[0]



        batch_size, c, _, _ = features_40x.shape
        features_40x = features_40x.view(batch_size, -1, c)
        features_10x = features_10x.view(batch_size, -1, c)

        multi_attended_features_40x = self.spatial_attention(features_10x, features_10x, features_40x)
        attended_features_40x = multi_attended_features_40x[0] + features_40x
        fused_features = torch.cat([attended_features_40x, features_10x], dim=2)

        # 线性层进行分类
        output = self.trans(fused_features)
        # return output, attended_features_40x, features_10x
        return output

# 多倍镜多尺度融合 + Atten层
class MultiAttenModel(nn.Module):
    def __init__(self, resnet_backbone_40x, resnet_backbone_10x, num_feats=512, output_class=3, num_heads=8):
        super(MultiAttenModel, self).__init__()
        self.resnet_backbone_40x = resnet_backbone_40x
        self.resnet_backbone_10x = resnet_backbone_10x
        self.num_feats = num_feats
        self.output_class = output_class
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 多通道空间注意力的卷积层
        self.spatial_attention = nn.MultiheadAttention(embed_dim=num_feats, num_heads=num_feats//num_heads, dropout=0.0)
        self.fc = nn.Linear(2*num_feats, output_class)

    def forward(self, x):
        # 分别通过 ResNet 骨干提取特征
        features_40x = self.resnet_backbone_40x(x['40x'])[0]
        features_10x = self.resnet_backbone_10x(x['10x'])[0]


        batch_size, c, w, h = features_40x.shape
        features_40x = features_40x.view(batch_size, -1, c)
        features_10x = features_10x.view(batch_size, -1, c)

        multi_attended_features_40x = self.spatial_attention(features_10x, features_10x, features_40x)
        attended_features_40x = multi_attended_features_40x[0] + features_40x
        fused_features = torch.cat([attended_features_40x, features_10x], dim=2)
        fused_features = fused_features.view(batch_size, 2*c, w, h)
        fused_features = self.avgpool(fused_features)
        fused_features = torch.flatten(fused_features, 1)
        # 线性层进行分类
        output = self.fc(fused_features)
        return output

# 多倍镜 + transformer层
class MultiTransModel(nn.Module):
    def __init__(self, resnet_backbone_40x, resnet_backbone_10x, num_feats=512, output_class=3):
        super(MultiTransModel, self).__init__()
        self.resnet_backbone_40x = resnet_backbone_40x
        self.resnet_backbone_10x = resnet_backbone_10x
        self.num_feats = num_feats
        self.output_class = output_class
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.trans = Transformer(num_classes=output_class, input_dim=num_feats * 2)

    def forward(self, x):
        # 分别通过 ResNet 骨干提取特征
        features_40x = self.resnet_backbone_40x(x['40x'])[0]
        features_10x = self.resnet_backbone_10x(x['10x'])[0]

        batch_size, c, _, _ = features_40x.shape
        features_40x = features_40x.view(batch_size, -1, c)
        features_10x = features_10x.view(batch_size, -1, c)
        fused_features = torch.cat([features_40x, features_10x], dim=2)

        # 线性层进行分类
        output = self.trans(fused_features)
        return output

# 多倍镜 + fc层
class MultiModel(nn.Module):
    def __init__(self, resnet_backbone_40x, resnet_backbone_10x, num_feats=512, output_class=3):
        super(MultiModel, self).__init__()
        self.resnet_backbone_40x = resnet_backbone_40x
        self.resnet_backbone_10x = resnet_backbone_10x
        self.num_feats = num_feats
        self.output_class = output_class
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2*num_feats, output_class)

    def forward(self, x):
        # 分别通过 ResNet 骨干提取特征
        features_40x = self.resnet_backbone_40x(x['40x'])[0]
        features_10x = self.resnet_backbone_10x(x['10x'])[0]

        fused_features = torch.cat([features_40x, features_10x], dim=1)
        fused_features = self.avgpool(fused_features)
        fused_features = torch.flatten(fused_features, 1)
        # 线性层进行分类
        output = self.fc(fused_features)

        return output

