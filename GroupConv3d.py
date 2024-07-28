import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange, Reduce

from einops import repeat
from einops import rearrange



# These are all combinations of 90 degree rotations in the D4 group 
# That will result in all possible orientations of a 3d object

orientations = [
    (0,0),
    (1,0),
    (2,0),
    (3,0),
    (0,1),
    (0,2),
    (1,1),
    (1,2),
    (2,1),
    (2,2),
    (3,1),
    (3,2),

]


class RotatedKernels(nn.Module):
    def __init__(self):
        super(RotatedKernels, self).__init__()
    
    # def create_rotated_kernels(self, kernel):
    #     rotations = []
    #     for n, m in orientations:
    #         t_kernel = torch.rot90(kernel, n, (2, 3))
    #         t_kernel = self.rot_diagonal(t_kernel, m)
    #         rotations.append(t_kernel)
    #     return torch.stack(rotations, dim=2)
    
    # def rot_diagonal(self, k, n):
    #     for i in range(n):
    #         k = rearrange(k, "o i h w d -> o i d h w")
    #     return k
    def create_rotated_kernels(self, kernel):
        rotations = []

        for rot_x in range(4):
            for rot_y in range(4):
                # for rot_z in range(4):
                rotated_kernel = torch.rot90(kernel, rot_x, [3,2])
                rotated_kernel = torch.rot90(rotated_kernel, rot_y, [4,2])
                    # rotated_kernel = torch.rot90(rotated_kernel, rot_z, [4,3])
                rotations.append(rotated_kernel)
        
        return torch.stack(rotations, dim=2)
    
    def forward(self, kernel):
        return self.create_rotated_kernels(kernel)




# input out, in, 3,3,3
# output should be 
# out * groups, in*groups, 3,3,3
# x will be
# batches, in, groups, 128,128,128
#reshaped to 
# batches, in*groups, 128,128,128
# conv will produce
# batches, out*groups, 128,128,128
# final output will be
# bathces, out, groups, 128,128,128



class GroupConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GroupConv3d, self).__init__()
        self.groups = 16
        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, 3, 3, 3, requires_grad=True)
        )
        nn.init.xavier_uniform_(self.weight)

        self.rearrange_weight_1 = Rearrange("o g2 i g h w d -> (o g2) (i g) h w d")

        self.rearrange_x_1 = Rearrange("b i g h w d -> b (i g) h w d")
        self.rearrange_x_2 = Rearrange("b (o g) h w d -> b o g h w d", g=self.groups)

        self.create_rotated_kernels = RotatedKernels()



    def forward(self, x):
        transformed_weight = self.create_rotated_kernels(self.weight)
        transformed_weight = repeat(transformed_weight, "o i g h w d -> o g2 i g h w d", g2=self.groups)
        transformed_weight = self.rearrange_weight_1(transformed_weight)
        print(transformed_weight.shape)
        x = self.rearrange_x_1(x)
        print('x', x.shape)
        x = F.conv3d(x, transformed_weight, padding=1, stride=1)

        x = self.rearrange_x_2(x)

        return x


class LiftingGroupConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LiftingGroupConv3d, self).__init__()
        self.groups = 16

        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, 3,3,3)
        )
        nn.init.xavier_uniform_(self.weight)

        self.rearrange_weight_1 = Rearrange("(o g2) i g h w d -> (o g2) (i g) h w d", g2=self.groups)
        
        self.rearrange_x_1 = Rearrange("b i g h w d -> b (i g) h w d", g=self.groups)
        
        self.rearrange_x_2 = Rearrange("b (o g) h w d -> b o g h w d", g=self.groups)

        self.create_rotated_kernels = RotatedKernels()



    def forward(self, x):
        transformed_weight = self.create_rotated_kernels(self.weight)
        transformed_weight = repeat(transformed_weight, "o i g h w d -> (o g2) i g h w d", g2=self.groups)
        transformed_weight = self.rearrange_weight_1(transformed_weight)  
        
        x = repeat(x, 'b c h w d -> b c g h w d', g=self.groups)
        x = self.rearrange_x_1(x)
        print(x.shape)
        print(transformed_weight.shape)
        x = F.conv3d(x, transformed_weight, padding=1)
        
        x = self.rearrange_x_2(x)

        return x

class GroupConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groups = 16
        
        self.weight = nn.Parameter(
            torch.zeros(in_channels, out_channels, 2,2,2)
        ) 
        nn.init.xavier_uniform_(self.weight)
        
        self.rearrange_weight_1 = Rearrange("i o g h w d -> i (o g) h w d")
        
        self.rearrange_x_1 = Rearrange("b i g h w d -> b (i g) h w d")
        self.rearrange_x_2 = Rearrange("b (o g) h w d -> b o g h w d", g=self.groups)    
        self.create_rotated_kernels = RotatedKernels()



    def forward(self, x):
        transformed_weight = self.create_rotated_kernels(self.weight)
        transformed_weight = repeat(transformed_weight, "i o g h w d -> (i g2) o g h w d", g2=self.groups)
        transformed_weight = self.rearrange_weight_1(transformed_weight)

        x = self.rearrange_x_1(x)
        x = F.conv_transpose3d(x, transformed_weight, stride=2)
        x = self.rearrange_x_2(x)

        return x


# from torch.autograd import gradcheck


# def test_gradient_flow():
#     in_channels = 1
#     out_channels = 1
#     groups = 12
#     batch_size = 1
#     depth, height, width = 8, 8, 8  # Reduced dimensions for memory efficiency

#     # Initialize the model
#     model = GroupConv3d(in_channels, out_channels).cuda()

#     # Create a random input tensor
#     input_tensor = torch.randn(batch_size, in_channels, groups, depth, height, width, requires_grad=True, dtype=torch.float32).cuda()

#     # Perform gradient check
#     input_tensor = input_tensor.double()  # gradcheck requires double precision
#     model = model.double()  # Ensure the model is in double precision
#     test = gradcheck(model, (input_tensor,), eps=1e-6, atol=1e-4)
#     print(f"Gradient check passed: {test}")

if __name__ == "__main__":


    print("got here")
    x = torch.randn(1,32,12,128,128,128).cuda()
    layer = GroupConv3d(32, 32).cuda()
    x = layer(x)
    print(x.shape)

    x = torch.randn(1,3,128,128,128).cuda()
    layer = LiftingGroupConv3d(3, 16).cuda()
    x = layer(x)
    print(x.shape)

    x = torch.randn(1, 8, 12, 32, 32, 32).cuda()
    layer = GroupConvTranspose3d(8, 4).cuda()
    x = layer(x)
    print(x.shape)

    
    