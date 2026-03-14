import torch.nn as nn
import torch.nn.functional as F

from .blocks import UnetConv3, UnetUp3CT, init_weights


class UNet3D(nn.Module):
    """3D U-Net used for CT volume segmentation."""

    def __init__(self, feature_scale=4, n_classes=2, in_channels=1, is_batchnorm=True):
        super().__init__()
        filters = [int(x / feature_scale) for x in [64, 128, 256, 512, 1024]]

        self.conv1 = UnetConv3(in_channels, filters[0], is_batchnorm)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = UnetConv3(filters[0], filters[1], is_batchnorm)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = UnetConv3(filters[1], filters[2], is_batchnorm)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = UnetConv3(filters[2], filters[3], is_batchnorm)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.center = UnetConv3(filters[3], filters[4], is_batchnorm)

        self.up4 = UnetUp3CT(filters[4], filters[3], is_batchnorm)
        self.up3 = UnetUp3CT(filters[3], filters[2], is_batchnorm)
        self.up2 = UnetUp3CT(filters[2], filters[1], is_batchnorm)
        self.up1 = UnetUp3CT(filters[1], filters[0], is_batchnorm)

        self.dropout = nn.Dropout(p=0.3)
        self.final = nn.Conv3d(filters[0], n_classes, kernel_size=1)
        self.final.apply(init_weights)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        center = self.dropout(self.center(self.pool4(c4)))

        u4 = self.up4(c4, center)
        u3 = self.up3(c3, u4)
        u2 = self.up2(c2, u3)
        u1 = self.dropout(self.up1(c1, u2))
        return self.final(u1)

    @staticmethod
    def apply_argmax_softmax(pred):
        return F.softmax(pred, dim=1)
