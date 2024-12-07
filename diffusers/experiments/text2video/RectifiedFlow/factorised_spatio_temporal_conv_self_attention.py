'''
Given a B, C, T, H, W dim PyTorch tensor do the following:
1. 2D spatial convolution
2. 1D temporal convolution and half the temporal resolution
'''

import torch
import torch.nn as nn
from einops import rearrange

# Example tensor of shape (B, C, T, H, W)
B, C, T, H, W = 2, 3, 4, 5, 6
x = torch.randn(B, C, T, H, W)

# 2D spatial convolution
conv2d = nn.Conv2d(in_channels=C, out_channels=8, kernel_size=3, padding=1)
x_2d_conv = conv2d(rearrange(x, 'b c t h w -> (b t) c h w'))
x_2d_conv = rearrange(x_2d_conv, '(b t) c h w -> b c t h w', b=B, t=T)
print(f"x_2d_conv.shap = {x_2d_conv.shape}")
# x_2d_conv.shap = torch.Size([2, 8, 4, 5, 6])

# 1D Temporal convolution
conv1d = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=2)
B, C, T, H, W = x_2d_conv.shape
x_1d_conv = conv1d(rearrange(x_2d_conv, 'b c t h w -> (b h w) c t'))
x_1d_conv = rearrange(x_1d_conv, '(b h w) c t -> b c t h w', b=B, h=H, w=W)
print(f"x_1d_conv.shap = {x_1d_conv.shape}")
# x_1d_conv.shap = torch.Size([2, 8, 2, 5, 6])
