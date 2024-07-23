import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    This wraps MaxPool3d to be compatible with the extra orientation 
    channels added by GroupConv3d

    This is not an efficient implementation.
"""
class MaxPool3DWrapper(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool3DWrapper, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        if len(x.shape) != 6:
            raise ValueError("Input tensor must have 6 dimensions (batch size, orientation channels, channels, h, w, d)")

        pooled_outputs = []
        for i in range(x.size(1)): 
            orientation_channel = x[:, i]
            pooled_channel = F.max_pool3d(orientation_channel, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
            pooled_outputs.append(pooled_channel.unsqueeze(1))

        output = torch.cat(pooled_outputs, dim=1)
        
        return output


if __name__ == '__main__':
    input_tensor = torch.randn(2, 3, 4, 8, 8, 8)  
    print(input_tensor.shape)
    kernel_size = 2
    stride = 2
    max_pool = MaxPool3DWrapper(kernel_size, stride)
    output_tensor = max_pool(input_tensor)

    print(output_tensor.shape) 