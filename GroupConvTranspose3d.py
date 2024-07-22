import torch
from torch.nn import functional as F
from torch import nn

class GroupConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, order="middle"):
        super(GroupConvTranspose3d, self).__init__()

        if order not in {"first","middle", "end"}:
            raise ValueError("Order needs to be either 'middle', or 'end'")
        
        kernel_size=(2,2,2)

        self.out_channels = out_channels
        self.kernel = nn.Parameter(
            torch.empty(in_channels, out_channels, *kernel_size)
        )
        nn.init.xavier_uniform_(self.kernel)

        self.order = order

       # rotates kernal backwards in place
    def _rot_back_90(self, t):
        t.rot90(k=1, dims=(4,2))

    # rotates kernel towards the right in place
    def _rot_right_90(self, t):
        t.rot90(k=1, dims=(3,2))
        
    def forward(self, x):
        if self.order == "first":
            # TODO: implement first
            pass
        elif self.order == "middle":
            pass
       
        print(x.shape)
        new_x = torch.empty(x.shape[0], x.shape[1], self.out_channels, 2 * x.shape[3], 2 * x.shape[4], 2 * x.shape[5])

        rotation = 0
        for i in range(16):
            channel = x[:, i, :, :, :, :]
            channel = F.conv_transpose3d(channel, self.kernel, stride=2)
            
            new_x[:,i,:,:,:,:] = channel

            self._rot_right_90(self.kernel)
            if rotation % 4 == 0:
                self._rot_back_90(self.kernel)

        x = new_x
        if self.order == "end":
            return x.sum(dim=1)
        
        return new_x


if __name__ == "__main__":
    data = torch.randn(2,16,5,64,64,64)

    layer = GroupConvTranspose3d(5, 2, order="end")

    data = layer(data)

    print(data.shape)

