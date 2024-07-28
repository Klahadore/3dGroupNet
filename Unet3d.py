"""
Preston Schmittou

This is a normal, 3d Unet that is used as a baseline.
"""

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv3d, Dropout, MaxPool3d, ConvTranspose3d, Linear
from torch import optim
from torch.utils.data import DataLoader
from dataset import SegDataset
import lightning as L

from metrics import calculate_metrics



class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, p=0.5):
        super(Down, self).__init__()
        self.C1 = Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding='same')
        self.dropout = Dropout(p=p)
        self.C2 = Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding='same')
        self.pool = MaxPool3d(kernel_size=(2,2,2), stride=2)
    
    def forward(self, x):
        x = self.C1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = F.relu(x)
        skip_con = x.clone()
        x = self.pool(x)

        return x, skip_con

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, p=0.5):
        super(Up, self).__init__()

        self.upscale = ConvTranspose3d(in_channels, out_channels, kernel_size=(2,2,2), stride=2)

        self.C1 = Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding='same')
        self.dropout = Dropout(p=p)
        self.C2 = Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding='same')

    def forward(self, x, skip_conn):
        x = self.upscale(x)
        #print("upscale", x.shape)
        x = torch.cat([skip_conn, x], dim=1)
       # print('cat', x.shape)
        x = self.C1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = F.relu(x)

        return x


class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.C1 = Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding='same')
        self.dropout = Dropout(p=0.5)
        self.C2 = Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding='same')
    
    def forward(self, x):
        x = self.C1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = F.relu(x)
        return x

        


class Unet(L.LightningModule):
    def __init__(self):
        super(Unet, self).__init__()

        self.down1 = Down(3, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)

        self.bottleneck = Bottleneck(64, 128)

        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.up4 = Up(16, 8)
        

        self.out = Conv3d(8, 4, kernel_size=(1,1,1), stride=1)
        # self.out = Up(64, 4)
        
    def forward(self, x):
     #   print(x.shape)
        x, s1 = self.down1(x)
     #   print("1", x.shape)
        x, s2 = self.down2(x)
     #   print(x.shape)
        x, s3 = self.down3(x)
    #    print(x.shape)
        x, s4 = self.down4(x)
     #   print(x.shape)
        
        x = self.bottleneck(x)
    #    print(x.shape)
        
        x = self.up1(x, s4)
     #   print(x.shape)
        x = self.up2(x, s3)
     #   print(x.shape)
        x = self.up3(x, s2)
     #   print(x.shape)
        x = self.up4(x, s1)
      #  print("g", x.shape)

        x = self.out(x)
       # print('h', x.shape)
     #   print(x.shape)

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
        self.log('train_accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log('train_precision', metrics['precision'], on_epoch=True, prog_bar=True)
        self.log('train_recall', metrics['recall'], on_epoch=True, prog_bar=True)
        self.log('train_f1_score', metrics['f1_score'], on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer