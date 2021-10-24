import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # 必须要用detach来分离出target，否则会计算目标值的梯度
        self.target = target.detach()
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        self.loss = self.criterion(inputs, self.target)
        return inputs


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.target = self.gram(target).detach()
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        self.G = self.gram(inputs)
        self.loss = self.criterion(self.G, self.target)
        return inputs


class GramMatrix(nn.Module):
    def forward(self, inputs):
        a, b, c, d = inputs.size()
        features = inputs.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
