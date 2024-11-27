# %%
import torch.nn as nn
import torch
from apex import amp
from torchvision import models
import collections
from nam import NAM


class MLAENet(nn.Module):
    def __init__(self, load_weights=False):
        super(MLAENet, self).__init__()
        amp.register_float_function(torch, 'sigmoid')
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = DC()
        self.conv0 = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1,stride=1)
        self.conv1 = nn.Conv2d(in_channels=512,out_channels=192,kernel_size=1,stride=1)
        self.c0 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1,padding=1)
        self.c1 =nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1,padding=1)
        self.c3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1,padding=1)

        self.at0 = NAM(channels=192)
        self.at1 = NAM(channels=64)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            # 10 convlution *(weight, bias) = 20 parameters
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.c0(x)
        x = nn.functional.interpolate(x, scale_factor=2,mode="bilinear")
        x = self.c1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.c2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.c3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)





def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                #layers += [conv2d, nn.BatchNorm2d(v), FR.FReLU()]
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                #layers += [conv2d, FR.FReLU()]
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers2(cfg, in_channels=3, batch_norm=False,kernel_size=3):

    layers = []
    for v in cfg:
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=kernel_size, stride=2)]
        else:
            conv2d_0 = nn.Conv2d(in_channels,in_channels//4, kernel_size=1)
            conv2d = nn.Conv2d(in_channels//4, v[0], kernel_size=kernel_size,
                               padding=v[1], dilation=v[1])
            #conv2d_1 = nn.Conv2d(in_channels//2, v[0], kernel_size=1)
            layers += [conv2d_0,conv2d,nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


class DB(nn.Module):
    def __init__(self, in_channels,out_channels = 0,d=1):
        super(DB, self).__init__()
        if out_channels == 0:
            out_channels = in_channels//2
        self.at=NAM(channels=out_channels)
        self.b0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.b1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1)
        self.b2 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3,dilation=d,padding=d,stride=1)

    def forward(self, x):
        identity = x
        identity = self.b0(x)
        #identity = self.at(identity)
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x = torch.cat((x1, x2), dim=1) + identity
        return x


class DB_1(nn.Module):
    def __init__(self, in_channels,out_channels = 0,d=1):
        super(DB_1, self).__init__()
        if out_channels == 0:
            out_channels = in_channels//2
        self.at=NAM(channels=out_channels)
        self.b0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.b1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,dilation=d,padding=d,stride=1)

    def forward(self, x):
        identity = x
        identity = self.b0(x)
        x = self.b1(x)
        x = x + identity
        return x


class DC(nn.Module):
    def __init__(self):
        super(DC, self).__init__()
        self.D1=nn.Sequential(DB(in_channels=512,out_channels=512,d=1),
                              DB(in_channels=512, d=1),
                              DB(in_channels=256, out_channels=256, d=1),
                              DB_1(in_channels=256, d=1),
                              DB_1(in_channels=128, d=1),
                              )

        self.D2=nn.Sequential(DB(in_channels=512,out_channels=512,d=1),
                              DB(in_channels=512, d=1),
                              DB(in_channels=256, out_channels=256, d=1),
                              DB_1(in_channels=256, d=2),
                              DB_1(in_channels=128, d=5),

                              )

        self.D3=nn.Sequential(DB(in_channels=512,out_channels=512,d=1),
                              DB(in_channels=512, d=1),
                              DB(in_channels=256, out_channels=256, d=1),
                              DB_1(in_channels=256, d=3),
                              DB_1(in_channels=128, d=7),
                              )
        self.at = NAM(channels=64)

    def forward(self, x):
        x1 = self.D1(x)
        x2 = self.D2(x)
        x3 = self.D3(x)
        x1 = self.at(x1)
        x2 = self.at(x2)
        x3 = self.at(x3)

        return torch.cat((x1, x2,x3), dim=1)
