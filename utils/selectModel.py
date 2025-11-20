
import torch

from model import *
from .fixedSeed import seed_everything

seed_everything(42)

class Wide_ResNet18_Weights():

    SIMCLR_40x = "../model/backbone/simclr/resnet18_40x.pth"
    SIMCLR_20x = "../model/backbone/simclr/resnet18_20x.pth"
    SIMCLR_10x = "../model/backbone/simclr/resnet18_10x.pth"
    SIMSIAM_40x = "../model/backbone/simsiam/resnet18_40x.pth"
    MOCOV3_40x = "../model/backbone/mocov3/resnet18_40x.pth"


model_urls = {
    'simclr_resnet18_10x': Wide_ResNet18_Weights.SIMCLR_10x,
    'simclr_resnet18_20x': Wide_ResNet18_Weights.SIMCLR_20x,
    'simclr_resnet18_40x': Wide_ResNet18_Weights.SIMCLR_40x,
    'simsiam_resnet18_40x': Wide_ResNet18_Weights.SIMSIAM_40x,
    'mocov3_resnet18_40x': Wide_ResNet18_Weights.MOCOV3_40x,
}

def load_pretrained_weights(backbone, path, pretrained=True):
    # 加载预训练模型
    if pretrained:
        pretrained_model = torch.load(path)
        backbone.load_state_dict(pretrained_model['state_dict'])
    return backbone

def freeze_model(model, freeze=False):
   # 冻结模型
    if freeze:
        for param in model.backbone.parameters():
            param.requires_grad = False
    # 打印模型参数
    for name, param in model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')

    return model


def choose_model(model_name, backbone_name, pretrained=True, num_classes=3, freeze=False):
    # 选择模型
    print("backbone_name:", backbone_name)

    if model_name =='signle_model':
        from mmselfsup.models import ResNet
        backbone = None
        if backbone_name == 'simclr_resnet18':
            backbone = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_40x'], pretrained)
        elif backbone_name == 'simsiam_resnet18':
            backbone = load_pretrained_weights(ResNet(depth=18), model_urls['simsiam_resnet18_40x'], pretrained)
        elif backbone_name == 'mocov3_resnet18':
            backbone = load_pretrained_weights(ResNet(depth=18), model_urls['mocov3_resnet18_40x'], pretrained)
        else:
            pass
        model = ResNet18(backbone, num_classes=num_classes)
        model = freeze_model(model, freeze)

    elif model_name == 'MultiAttenTransModel':
        from mmselfsup.models import ResNet

        backbone_40x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_40x'], pretrained)
        backbone_10x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_10x'], pretrained)

        model =MultiAttenTransModel(backbone_40x, backbone_10x, output_class=num_classes)
        model = freeze_model(model, freeze)

    elif model_name == 'MultiAttenModel':
        from mmselfsup.models import ResNet

        backbone_40x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_40x'], pretrained)
        backbone_10x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_10x'], pretrained)

        model =MultiAttenModel(backbone_40x, backbone_10x, output_class=num_classes)
        model = freeze_model(model, freeze)

    elif model_name == 'MultiTransModel':
        from mmselfsup.models import ResNet

        backbone_40x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_40x'], pretrained)
        backbone_10x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_10x'], pretrained)

        model = MultiTransModel(backbone_40x, backbone_10x, output_class=num_classes)
        model = freeze_model(model, freeze)

    elif model_name == 'MultiModel':
        from mmselfsup.models import ResNet

        backbone_40x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_40x'], pretrained)
        backbone_10x = load_pretrained_weights(ResNet(depth=18), model_urls['simclr_resnet18_10x'], pretrained)

        model = MultiModel(backbone_40x, backbone_10x, output_class=num_classes)
        model = freeze_model(model, freeze)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'msbp_model':
        model = resnet_msbp(exp_mode='ResNet_MSBP', nr_classes=num_classes, pretrained=pretrained)

    elif model_name == 'res2net50_model':
        model = res2net50_v1b(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'msdnet_model':
        model = msdnet(pretrained=pretrained, nr_class=num_classes)

    elif model_name == 'ensemble_r50':
        model = ensemble_r50(num_classes=num_classes, pretrained=pretrained)

    elif model_name == 'ensemble_r18':
        model = ensemble_r18(num_classes=num_classes, pretrained=pretrained)

    return model



if __name__ == '__main__':

    net = choose_model("signle_model", "simclr_resnet18")
    print(net)





