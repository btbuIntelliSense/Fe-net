import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, kk=(3, 3), ss=(2, 2), pp=(1,1)):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.kk = kk
        self.ss = ss
        self.pp = pp

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), self.kk,self.ss,self.pp).pow(1. / self.p)


class GEM_Att(nn.Module):
    def __init__(self, c, ac, k=7):
        super(GEM_Att, self).__init__()
        p = (k - 1) // 2

        self.gem = GeM()
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, ac, 1),
            nn.Conv2d(ac, ac, k, padding=p),
            nn.Conv2d(ac, c, 1))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gem(x)
        y = F.interpolate(y, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
        y = self.conv(y)
        return x * self.sigmoid(y)
