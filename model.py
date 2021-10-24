import torch
import torchvision
import torch.nn as nn
from losses import ContentLoss, StyleLoss


class Transfer(nn.Module):
    def __init__(self, style_img, content_img, device):
        super(Transfer, self).__init__()
        self.style_img = style_img
        self.content_img = content_img
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_losses = []
        self.style_losses = []
        basenet = torchvision.models.vgg19(pretrained=True).features.to(device)
        self.basenet = self.build_model(basenet)
        self.device = device

    def build_model(self, net):
        i = 1
        normalization = Normalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), device=self.device)
        model = nn.Sequential(normalization)

        for layer in list(net):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                model.add_module(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                # 注意这里需要将inplace修改为False
                model.add_module(name, nn.ReLU(inplace=False))
                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                model.add_module(name, layer)

            if isinstance(layer, nn.BatchNorm2d):
                name = "" + str(i)
                model.add_module(name, layer)

            if name in self.content_layers:
                target_feature = model(self.content_img)
                content_loss = ContentLoss(target_feature)
                model.add_module("content_loss_" + str(i), content_loss)
                self.content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(self.style_img)
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_" + str(i), style_loss)
                self.style_losses.append(style_loss)

            if i == 6:
                return model


class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std
