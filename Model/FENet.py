import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.GEM import GEM_Att
from Model.mpvcov import CovpoolLayer, SqrtmLayer, TriuvecLayer
from Tool.Dropblock import DropBlock2D, LinearScheduler


def conv_(ic, oc, k=1, s=1, g=1, act=True):
    return nn.Sequential(
        nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=(k - 1) // 2, groups=g,
                  bias=False),
        nn.BatchNorm2d(oc),
        nn.Mish(inplace=True) if act else nn.Sequential(),
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class Block(nn.Module):
    def __init__(self, ic):
        super(Block, self).__init__()
        self.conv1 = conv_(ic, ic)
        self.conv2 = conv_(ic, ic, 3, g=32)
        self.conv3 = conv_(ic, ic, act=False)
        self.act = nn.Mish(inplace=True)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x[:, :res.size(1)] = x[:, :res.size(1)] + res
        return self.act(x + res)


class Stage(nn.Module):
    def __init__(self, ic, block_num, csp_act=True, down=True):
        super(Stage, self).__init__()
        csp_ic = ic
        base_ic = ic
        self.downsample = conv_(ic, ic, 3, 2, g=32) if down else None
        self.conv_csp_1 = conv_(ic, csp_ic, act=csp_act)
        self.conv_csp_2 = conv_(csp_ic, csp_ic, act=csp_act)
        self.conv_block = conv_(ic, base_ic)
        self.layers = nn.ModuleList()
        for _ in range(block_num):
            self.layers.append(Block(base_ic))
        self.att = GEM_Att(base_ic, base_ic//32)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x_csp = self.conv_csp_1(x)
        x_csp = self.conv_csp_2(x_csp)
        x = self.conv_block(x)
        for layer in self.layers:
            x = layer(x)
        return channel_shuffle(torch.cat([x_csp, x], dim=1), 2)


class CSPResNeXt50(nn.Module):
    def __init__(self, ic, block_num, num_classes):
        super(CSPResNeXt50, self).__init__()
        self.stem = nn.Sequential(conv_(3, ic[0], 7, 2),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = Stage(ic[0], block_num[0], down=False)
        self.stage2 = Stage(ic[1], block_num[1], csp_act=False)
        self.stage3 = Stage(ic[2], block_num[2], csp_act=False)
        self.stage4 = Stage(ic[3], block_num[3], csp_act=False)
        
        self.layer_reduce = nn.Conv2d(ic[4], 256, 1)
        self.fc_isqrt = nn.Linear(int(256 * (256 + 1) / 2), num_classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=0, block_size=7),
            start_value=0.,
            stop_value=0.3,
            nr_steps=5e3
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.0003)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dropblock(self.stage3(x))
        x = self.dropblock(self.stage4(x))
        x = self.layer_reduce(x)
        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 5)
        x = TriuvecLayer(x)
        return self.fc_isqrt(self.dropout(x.view(x.size(0), -1)))


def Model(num_classes):
    ic = [128, 256, 512, 1024, 2048]
    num_layers = [3, 3, 5, 2]
    return CSPResNeXt50(ic, num_layers, num_classes)