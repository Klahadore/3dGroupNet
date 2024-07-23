"""
Preston Schmittou

This is a normal, 3d Unet that is used as a baseline.
"""

import torch
import numpy as np
from torch.nn import functional as F
from torch.nn import Conv3d, Dropout, MaxPool3d, ConvTranspose3d, Linear
from torch import optim
from torch.utils.data import DataLoader
from dataset import SegDataset
from metrics import calculate_auc, calculate_f1
"""
Lets define the dimensionality:

Original input dim:
(128, 128, 128, 3)

C1: 64

"""

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.C1 = Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding='same')
        self.dropout = Dropout(p=0.1)
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
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.upscale = ConvTranspose3d(in_channels, out_channels, kernel_size=(2,2,2), stride=2)

        self.C1 = Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding='same')
        self.dropout = Dropout(p=0.1)
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
        self.dropout = Dropout(p=0.1)
        self.C2 = Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding='same')
    
    def forward(self, x):
        x = self.C1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = F.relu(x)
        return x

        
# initial dim = 128, 128, 3


class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.down1 = Down(3, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)

        self.bottleneck = Bottleneck(128, 256)

        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        

        self.out = Conv3d(16, 4, kernel_size=(1,1,1))
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


if __name__ == "__main__":
    train_dataset = SegDataset("./data/train/images", "./data/train/masks")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    model = Unet()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        
        all_preds = []
        all_masks = []

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)


            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            print(loss.item())
            
            if (epoch + 1) % 5 == 0:
                model.cpu()
                model_save_path = f'3d_UNet_Normal_{epoch+1}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved to {model_save_path}')
                model.to(device)
            # all_preds.append(outputs.detach().cpu().numpy())
            # all_masks.append(masks.detach().cpu().numpy())
            # all_preds = np.concatenate(all_preds, axis = 0)
            # all_masks = np.concatenate(all_masks, axis = 0)

            # auc = calculate_auc(all_masks, all_preds)
            # f1 = calculate_f1(all_masks, all_preds)
            
            # print(f"Epoch {epoch} - Loss: {loss.item()}, AUC: {auc}, F1: {f1}")
            # print("completed epoch:", epoch)

