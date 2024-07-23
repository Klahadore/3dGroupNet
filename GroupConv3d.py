import torch
from torch.nn import functional as F
from torch import nn


# Groups specifies is this is a first layer of a network which operates on images
# Or a later layer of a network which operates on the group orientations


"""
    Specify order:
    'first' means that the layer operates on images which contain no orientation channels
    'middle' indicates that the layer operates on group orientations, which contain orientation channels
    'last' indicates that the layer will add up all the separate orientation channels, and then output no orientation channels. 
"""

# IMPORTANT: the only kernel size supported is (3,3,3), and the only stride is one, and the only padding is "same".
class GroupConv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, order='middle'):
        super(GroupConv3d, self).__init__()

        if order not in {"first", "middle", "end"}:
            raise ValueError("Order needs to be 'first', 'middle', 'end'.")
        
        kernel_size=(3,3,3)

        self.out_channels = out_channels
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        nn.init.xavier_uniform_(self.kernel)
        print("ss", self.kernel.device)
        self.order = order
   
    def _rot_back_90(self, t):
        t.rot90(k=1, dims=(4,2))

    def _rot_right_90(self, t):
        t.rot90(k=1, dims=(3,2))

    
    
        
    
    def forward(self, x):
        if self.order == 'first':
            if len(x.shape) != 5:
                raise ValueError(f"Tensor shape is not correct. Expected shape for 'first' is 5 dims, but got {len(x.shape)}")    
            new_shape = list(x.shape)
            
            
            new_shape.insert(1, 16)

            # Add a dimension at position 1
            x = x.unsqueeze(1)

            # Expand to the new shape
            x = x.expand(*new_shape)

        elif self.order == 'middle':
            if len(x.shape) != 6:
                raise ValueError(f"Tensor shape is not correct. Expected shape for 'middle' is 5 dims, but got {len(x.shape)}")
        elif self.order == 'end':
            if len(x.shape) != 6:
                raise ValueError(f"Tensor shape is not correct. Expected shape for 'end' is 5 dims, but got {len(x.shape)}")
                      
        new_x = torch.empty(x.shape[0], x.shape[1], self.out_channels, x.shape[3], x.shape[4], x.shape[5])

        rotation = 0
        for i in range(16):
            channel = x[:, i, :, :, :, :]
            channel = F.conv3d(channel, self.kernel, stride=1, padding=1)
            new_x[:,i,:,:,:,:] = channel

            self._rot_right_90(self.kernel)
            if rotation % 4 == 0:
                self._rot_back_90(self.kernel)
        
        x = new_x
        if self.order == 'end':
            return x.sum(dim=1)
        
        return x
        
            
if __name__ == '__main__':
    # Testing code
    layer = GroupConv3d(5,2, order='end')
    
    data = torch.randn(2,16,5,128,128,128)
    
    data = layer(data)
    print(data.shape)
    
    layer1 = GroupConv3d(3,1,order="first")
    data = torch.randn(2,3,128,128,128)
    data = layer1(data)
    print(data.shape)
            
        
        
        
        
        
        
