import torch
from torch.nn import functional as F
from torch import nn

""" 
This will consider the following orientations

0,0
0,90
0,180
0,270

90,0
90,90
90,180
90,270

180,0
180,90
180,180
180,270

270,0
270,90
270,180
270,270

This will add a channel for each orientation considered.

Dimensions should be
(batches,channels, orientation channels , x , y , z)
"""


# Groups specifies is this is a first layer of a network which operates on images
# Or a later layer of a network which operates on the group orientations
def _rot_back_90(t):
    return torch.rot90(t, k=1, dims=(4,2))

def _rot_right_90(t):
    return torch.rot90(t, k=1, dims=(3,2))


class GroupConv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, groups=True):
        super(GroupConv3d, self).__init__()
        
        self.stride = stride  
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        nn.init.xavier_uniform_(self.kernel)
        
        self.groups = groups
   
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
        for i in range(x.shape[2]):
            channel = x[:, i, :, :, :, :]
            print(channel.shape)
            # TODO: padding = 1 is only true if kernel size is 3x3x3
            channel = F.conv3d(channel, self.kernel, stride=self.stride, padding=1)
            print(channel.shape)
            x[:,i,:,:,:,:] = channel
            print(x.shape)
            self.kernel = _rot_right_90(self.kernel)
            if rotation % 4 == 0:
                self.kernel = _rot_back_90(self.kernel)
                
            
if __name__ == '__main__':
    layer = GroupConv3d(5,2)
    
    data = torch.randn(1,16,5,128,128,128)
    
    data = layer(data)
    print(data.shape)
            
            
        
        
        
        
        
        
