import torch    
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv3d, Dropout, MaxPool3d, ConvTranspose3d, Linear
from torch import optim
from torch.utils.data import DataLoader
from dataset import SegDataset
import lightning as L
from einops.layers.torch import Rearrange, Reduce

import gconv.gnn as gnn
from gconv.gnn import GMaxGroupPool, GConvSE3, GLiftingConvSE3, GSeparableConvSE3, GReLU                                                  
from gconv.geometry.groups import so3 as R

from metrics import calculate_metrics




import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce, Rearrange

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, lifting=False):
        super(Down, self).__init__()
        self.lifting = lifting

        
        if self.lifting:
            self.C1 = GLiftingConvSE3(in_channels, out_channels, kernel_size=3, padding='same')
        else:
            self.C1 = GSeparableConvSE3(in_channels, out_channels, kernel_size=3, padding='same')
        
        
        self.C2 = GSeparableConvSE3(out_channels, out_channels, kernel_size=3, padding='same')
        
        
        self.pool = Reduce("b o g (h h2) (w w2) (d d2) -> b o g h w d", reduction='max', h2=2, w2=2, d2=2)

    def forward(self, x, H_previous):
        if self.lifting:
            x, H1 = self.C1(x)
        else:
            x, H1 = self.C1(x, H_previous)
        
        x = F.leaky_relu(x)
        
        x, H2 = self.C2(x, H1)
       # print(H2.shape)
        x = F.leaky_relu(x)
        
        skip_con = x.clone()
        x = self.pool(x)
        
        return x, H2, skip_con

        #self.C2 = GSeparableConvSE3(2*out_channels, out_channels, kernel_size=3,padding='same', permute_output_grid=False)

        # self.concat_x = Rearrange("b i g h w d -> b i g h w d")

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.rearrange1 = Rearrange("b o g h w d -> b (o g) h w d") 
        self.rearrange2 = Rearrange("b (o g) h w d -> b o g h w d", g=4) 

        self.ResidualConv = GSeparableConvSE3(out_channels, out_channels, kernel_size=3, padding='same')
        self.C1 = GSeparableConvSE3(in_channels, out_channels, kernel_size=3, padding='same')
        self.C2 = GSeparableConvSE3(out_channels, out_channels, kernel_size=3, padding='same')
        self.C3 = GSeparableConvSE3(out_channels, out_channels, kernel_size=3, padding='same')


    def forward(self, x, H_skip, skip_con, H_previous):
        x, H1 = self.C1(x,  H_previous)
        x = F.leaky_relu(x)
        
# x_residual, H_residual = self.Residual_Conv(skip_con, H, H_out), and H_residual == H_out
        skip_con, H_skip = self.ResidualConv(skip_con, H_skip, out_H=H1)

        x = self.rearrange1(x)
        x = self.upsample(x)
        x = self.rearrange2(x)

        x, H2 = self.C2(x+skip_con, H1)
        x = F.leaky_relu(x)
        x, H3 = self.C3(x, H2)
        return x, H3

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.C1 = GSeparableConvSE3(in_channels, out_channels, 3, padding='same')
        self.C2 = GSeparableConvSE3(out_channels, out_channels, 3, padding='same')

    def forward(self, x, H_prev):
        x, H1 = self.C1(x, H_prev)
        x = F.leaky_relu(x)
        x, H2 = self.C2(x, H1)
        x = F.leaky_relu(x)
        return x, H2



class GroupUnet3d(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.down1 = Down(3, 8, lifting=True)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)

        self.bottleneck = Bottleneck(64, 128)

        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.up4 = Up(16, 8)


        # self.pool = GMaxGroupPool()
        self.pool = Reduce("b c g h w d -> b c h w d", reduction="max")

        self.out = Conv3d(8, 4, kernel_size=(1,1,1), stride=1)

    def forward(self, x):
        x, H1, S1 = self.down1(x, None)
        x, H2, S2 = self.down2(x, H1)
        x, H3, S3 = self.down3(x, H2)
        x, H4, S4 = self.down4(x, H3)

        x, H5 = self.bottleneck(x, H4)

        x, H6 = self.up1(x, H4, S4, H5)
        x, H7 = self.up2(x, H3, S3, H6)
        x, H8 = self.up3(x, H2, S2, H7)
        x, _ = self.up4(x, H1, S1, H8)

        x = self.pool(x)
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        y = torch.argmax(y, dim=1)
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        print(loss.item())
        self.log('training_loss', loss)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.log('val_loss', loss)
        
        metrics = calculate_metrics(y_hat, y)
        print("F1 score", metrics['f1_score'])
        self.log('train_accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log('train_precision', metrics['precision'], on_epoch=True, prog_bar=True)
        self.log('train_recall', metrics['recall'], on_epoch=True, prog_bar=True)
        self.log('train_f1_score', metrics['f1_score'], on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer



        

if __name__ == "__main__":
    ...
