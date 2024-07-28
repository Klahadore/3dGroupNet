import torch
from torch.nn import functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from einops.layers.torch import Reduce
import lightning as L

from GroupConv3d import GroupConv3d, LiftingGroupConv3d, GroupConvTranspose3d
from dataset import SegDataset
from metrics import calculate_metrics


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, lifting=False, dropout_prob=0.5):
        super().__init__()

        if lifting:
            self.C1 = LiftingGroupConv3d(in_channels, out_channels)
        else:
            self.C1 = GroupConv3d(in_channels, out_channels)
        
        self.C2 = GroupConv3d(out_channels, out_channels)

        self.pool = Reduce("b c g (h h2) (w w2) (d d2) -> b c g h w d", 'max', h2=2, w2=2, d2=2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.C1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = F.relu(x)

        skip_con = torch.clone(x)

        x = self.pool(x)

        return x, skip_con


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__() 
        self.C_transpose = GroupConvTranspose3d(in_channels, out_channels)
        self.C1 = GroupConv3d(in_channels, out_channels)
        self.C2 = GroupConv3d(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, skip_con):
        x = self.C_transpose(x)

        x = torch.cat((x, skip_con), dim=1)  
        x = self.C1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = F.relu(x)
        
        return x
        

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()

        self.C1 = GroupConv3d(in_channels, out_channels)
        self.C2 = GroupConv3d(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.C1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.C2(x)
        x = F.relu(x)
        
        return x

class GroupUnet3d(L.LightningModule):
    def __init__(self):
        super(GroupUnet3d, self).__init__()
        self.Down1 = Down(3, 8, lifting=True, dropout_prob=0)
        self.Down2 = Down(8,16)
        self.Down3 = Down(16,32)
        self.Down4 = Down(32,64)

        self.bottleneck = Bottleneck(64,128)

        self.Up1 = Up(128,64)
        self.Up2 = Up(64,32)
        self.Up3 = Up(32, 16)
        self.Up4 = Up(16,8)

        self.reduce = Reduce("b c g h w d -> b c h w d", "max")

        self.conv_out = nn.Conv3d(8,4, kernel_size=(1,1,1), stride=1)

    def forward(self, x):
        x, s1 = self.Down1(x)
        x, s2 = self.Down2(x)
        x, s3 = self.Down3(x)
        x, s4 = self.Down4(x)

        x = self.bottleneck(x)
        x = self.Up1(x, s4)
        x = self.Up2(x, s3)
        x = self.Up3(x, s2)
        x = self.Up4(x, s1)
        x = self.reduce(x)
        x = self.conv_out(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        print(x.shape)
        print(y.shape)
        y = torch.argmax(y, dim=1)
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.log('train_loss', loss)
        print(loss.item())

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

        print(metrics['f1_score'])
        return loss, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.000001)
        return optimizer
        
    

    
if __name__ == "__main__":
    train_dataset = SegDataset("./data/train/images", "./data/train/masks")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    model = GroupUnet3d()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    # for name, param in model.named_parameters():
    #     print(name, param.data)

    for epoch in range(num_epochs):
        model.train()
        for images, masks, in train_loader:
            optimizer.zero_grad()
            images = images.cuda()
            masks = masks.cuda()

            
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            
            print(loss.item())
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} has gradients:\n{param.grad}\n")
                else:
                    print(f"{name} has no gradients")
        if (epoch + 1) % 5 == 0:
                model.cpu()
                model_save_path = f'3d_Group_UNet_Normal_{epoch+1}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved to {model_save_path}')
                model.to(device)