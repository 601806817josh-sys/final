import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super().__init__()
        if is_batchnorm:
            self.block = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_size),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_size, out_size, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.block.apply(init_weights)

    def forward(self, x):
        return self.block(x)


class UnetUp3CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True)
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm)

    def forward(self, skip, x):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
