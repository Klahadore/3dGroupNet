import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange, Reduce

def create_rotated_kernels(kernel):
    rotations = []

    for rot_x in range(4):
        for rot_y in range(4):
            rotated_kernel = torch.rot90(kernel, rot_x, [3,4])
            rotated_kernel = torch.rot90(rotated_kernel, rot_y, [2,4])
            rotations.append(rotated_kernel)
    
    return torch.stack(rotations, dim=2)


class GroupConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GroupConv3d, self).__init__()
        self.groups = 16
        self.weight = nn.Parameter(
            torch.zeros(out_channels * self.groups, in_channels, 3, 3, 3)
        )
        nn.init.xavier_uniform_(self.weight)

        self.rearrange_weight_1 = Rearrange("o i g h w d -> o (i g) h w d")

        self.rearrange_x_1 = Rearrange("b i g h w d -> b (i g) h w d")
        self.rearrange_x_2 = Rearrange("b (o g) h w d -> b o g h w d", g=self.groups)

    def forward(self, x):
        transformed_weight = create_rotated_kernels(self.weight)
        transformed_weight = self.rearrange_weight_1(transformed_weight)

        x = self.rearrange_x_1(x)
        
        x = F.conv3d(x, transformed_weight, padding=1)

        x = self.rearrange_x_2(x)

        return x


class LiftingGroupConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LiftingGroupConv3d, self).__init__()
        self.groups = 16

        self.weight = nn.Parameter(
            torch.zeros(out_channels * self.groups, in_channels, 3,3,3)
        )
        nn.init.xavier_uniform_(self.weight)

        self.rearrange_weight_1 = Rearrange("o i g h w d -> o (i g) h w d")


        self.rep
        self.rearrange_x_1 = Rearrange("b i h w d -> b (i g) h w d", g=self.groups)
        
        self.rearrange_x_2 = Rearrange("b (o g) h w d -> b o g h w d", g=self.groups)


    def forward(self, x):
        transformed_weight = create_rotated_kernels(self.weight)
        transformed_weight = self.rearrange_weight_1(transformed_weight)     
        print(x.shape)

        x = self.rearrange_x_1(x)
        print(x.shape)
        x = F.conv3d(x, transformed_weight, padding=1)
        print(x.shape)
        x = self.rearrange_x_2(x)

        return x


if __name__ == "__main__":


    # print("got here")
    # x = torch.randn(1,32,16,128,128,128).cuda()
    # layer = GroupConv3d(32, 32).cuda()


    # print(layer(x).shape)

    x = torch.randn(1,3,128,128,128)
    layer = LiftingGroupConv3d(3, 8)

    print(layer(x).shape)

    
    