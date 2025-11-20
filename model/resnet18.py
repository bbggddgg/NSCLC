import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ResNet18, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)[0]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x