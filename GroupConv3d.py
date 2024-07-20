import torch
from torch.nn import functional as F
from torch import nn


# Groups specifies is this is a first layer of a network which operates on images
# Or a later layer of a network which operates on the group orientations


class GroupConv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, groups=True):
        super(GroupConv3d, self).__init__()
        
        self.out_channels = out_channels
        self.stride = stride  
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        nn.init.xavier_uniform_(self.kernel)
        
        self.groups = groups
   
    def _rot_back_90(self, t):
        t.rot90(k=1, dims=(4,2))

    def _rot_right_90(self, t):
        t.rot90(k=1, dims=(3,2))
        
    def _conv_forward(self, x):
        return x
    
    def forward(self, x):
        if self.groups:
            assert len(x.shape) == 6
        else:
            assert len(x.shape) == 5

        
        # Adds dimension at dim=2
        if not self.groups:
            x.unsqueeze_(2)
        
        rotation = 1
        print(x.shape)

        new_x = torch.empty(x.shape[0], x.shape[1], self.out_channels, x.shape[3], x.shape[4], x.shape[5])
        for i in range(x.shape[2]):
            channel = x[:, i, :, :, :, :]
            print(channel.shape)
            # TODO: padding = 1 is only true if kernel size is 3x3x3
            channel = F.conv3d(channel, self.kernel, stride=self.stride, padding=1)
            print(channel.shape)
            new_x[:,i,:,:,:,:] = channel
            print(x.shape)

            self._rot_right_90(self.kernel)
            if rotation % 4 == 0:
                self._rot_back_90(self.kernel)
            print("made it through once")
        
        return new_x
                
            
if __name__ == '__main__':
    layer = GroupConv3d(5,2)
    
    data = torch.randn(1,16,5,128,128,128)
    
    data = layer(data)
    print(data.shape)
            
            
        
        
        
        
        
        
