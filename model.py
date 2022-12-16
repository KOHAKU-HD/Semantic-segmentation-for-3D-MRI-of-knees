import random
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, tri_linear=True):
        super(UpSampling, self).__init__()
        if tri_linear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, out_channels // 2, kernel_size=2, stride=2)

        self.double_conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # [Batch_size, channels, depth, hight, width]

        diff_depth = x2.shape[2] - x1.shape[2]
        diff_height = x2.shape[3] - x1.shape[3]
        diff_width = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, [diff_width // 2, diff_width - diff_width // 2, 
                        diff_height // 2, diff_height - diff_height // 2,
                        diff_depth // 2, diff_depth - diff_depth // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels, n_classes, deep_supervision=True):
        super().__init__()

        num_filters = [64, 64, 128, 256]

        self.deep_supervision = deep_supervision

        # First column
        self.conv0_0 = DoubleConv3D(in_channels, num_filters[0])
        self.conv1_0 = DownSampling(num_filters[0], num_filters[1])
        self.conv2_0 = DownSampling(num_filters[1], num_filters[2])
        self.conv3_0 = DownSampling(num_filters[2], num_filters[3])

        # Second column
        self.conv0_1 = UpSampling(num_filters[0] * 1 + num_filters[1], num_filters[0])
        self.conv1_1 = UpSampling(num_filters[1] * 1 + num_filters[2], num_filters[1])
        self.conv2_1 = UpSampling(num_filters[2] * 1 + num_filters[3], num_filters[2])

        # Third column
        self.conv0_2 = UpSampling(num_filters[0] * 2 + num_filters[1], num_filters[0])
        self.conv1_2 = UpSampling(num_filters[1] * 2 + num_filters[2], num_filters[1])

        # Fourth column
        self.conv0_3 = UpSampling(num_filters[0] * 3 + num_filters[1], num_filters[0])

        # 1x1
        if self.deep_supervision:
            self.o1 = Out(num_filters[0], n_classes)
            self.o2 = Out(num_filters[0], n_classes)
            self.o3 = Out(num_filters[0], n_classes)
        else:
            self.o = Out(num_filters[0], n_classes)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(x1_0, x0_0)

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(x2_0, x1_0)
        x0_2 = self.conv0_2(x1_1, torch.cat([x0_0, x0_1], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(x3_0, x2_0)
        x1_2 = self.conv1_2(x2_1, torch.cat([x1_0, x1_1], 1))
        x0_3 = self.conv0_3(x1_2, torch.cat([x0_0, x0_1, x0_2], 1))

        if self.deep_supervision:
            out1 = self.o1(x0_1)
            out2 = self.o2(x0_2)
            out3 = self.o3(x0_3)
            return [out1, out2, out3]

        else:
            out = self.o(x0_3)
            return out


class Vnet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.main_net = UnetPlusPlus(in_channels, n_classes, True)
        self.conv1 = Out(n_classes, n_classes)
        self.conv2 = Out(n_classes, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feature_maps = self.main_net(x)
        output = self.softmax(self.conv2(self.conv1(sum(feature_maps))))

        return feature_maps, output