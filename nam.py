import torch.nn as nn
import torch
from apex import amp


class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        amp.register_float_function(torch, 'sigmoid')
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual

        return x


class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        self.Channel_Att = Channel_Att(channels)


    def forward(self, x):
        x_out1 = self.Channel_Att(x)
        return x_out1