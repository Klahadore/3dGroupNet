import torch    
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv3d, Dropout, MaxPool3d, ConvTranspose3d, Linear
from torch import optim
from torch.utils.data import DataLoader
from dataset import SegDataset
import lightning as L
from einops.layers.PyTorch import Rearrange

import gconv.gnn as gnn
from gconv.gnn import GAvgSpatialPool3d, GAdaptiveMaxSpatialPool3d, GConvSE3, GLiftingConvSE3                                                              


class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, out_shape, p=0.5, lifting=False):
        super(Down, self).__init__()
        self.lifting = lifting
        if self.lifting:
            self.C1 = GLiftingConvSE3(in_channels, out_channels)
        else:
            self.C1 = GConvSE3(in_channels, out_channels)
        
        self.C2 = GConvSE3(out_channels, out_channels)
        
        self.pool = GAdaptiveMaxSpatialPool3d(*out_shape)


    def forward(self, x, H_previous):
        x, H1 = self.C1(x) if self.lifting else self.C1(x, H_previous)
        x = F.relu(x)
        x, H2 = self.C2(x, H1)
        x = F.relu(x)
        skip_con = x.clone()
        x = self.pool(x)
        return x, skip_con, H2
    
class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='billinear', align_corners=True)
        self.rearrange1 = Rearrange("b o g h w d -> b (o g) h w d)") 
        self.rearrange2 = Rearrange("b (o g) h w d -> b o g h w d", o=self.out_channels) 

        self.C1 = GConvSE3(out_channels, out_channels)
        self.C2 = GConvSE3(out_channels, out_channels)

        def forward(self, x, skip_con, H_previous):
            x = self.rearrange1(x)
            x = self.upsample(x)
            x = self.rearrange2(x)

            x = torch.cat([skip_con, x], dim=1)

            x, H1 = self.C1(x,  H_previous)
            x, H2 = self.C2(x, H1)

            return x, H2

