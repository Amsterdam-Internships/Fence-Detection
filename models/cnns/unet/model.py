import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import config
import torch.nn as nn

from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F


# class Block(Module):
#     def __init__(self, inChannels, outChannels):
#         super().__init__()
#         # store the convolution and RELU layers
#         self.conv1 = Conv2d(inChannels, outChannels, 3)
#         self.relu = ReLU()
#         self.conv2 = Conv2d(outChannels, outChannels, 3)
#     def forward(self, x):
#         # apply CONV => RELU => CONV block to the inputs and return it
#         return self.conv2(self.relu(self.conv1(x)))


# class Encoder(Module):
#     def __init__(self, channels=(3, 16, 32, 64)):
#         super().__init__()
#         # store the encoder blocks and maxpooling layer
#         self.encBlocks = ModuleList(
#             [Block(channels[i], channels[i + 1])
#                  for i in range(len(channels) - 1)])
#         self.pool = MaxPool2d(2)
#     def forward(self, x):
#         # initialize an empty list to store the intermediate outputs
#         blockOutputs = []
#         # loop through the encoder blocks
#         for block in self.encBlocks:
#             # pass the inputs through the current encoder block, store
#             # the outputs, and then apply maxpooling on the output
#             x = block(x)
#             blockOutputs.append(x)
#             x = self.pool(x)
#         # return the list containing the intermediate outputs
#         return blockOutputs


# class Decoder(Module):
#     def __init__(self, channels=(64, 32, 16)):
#         super().__init__()
#         # initialize the number of channels, upsampler blocks, and
#         # decoder blocks
#         self.channels = channels
#         self.upconvs = ModuleList(
#             [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
#                  for i in range(len(channels) - 1)])
#         self.dec_blocks = ModuleList(
#             [Block(channels[i], channels[i + 1])
#                  for i in range(len(channels) - 1)])
#     def forward(self, x, encFeatures):
#         # loop through the number of channels
#         for i in range(len(self.channels) - 1):
#             # pass the inputs through the upsampler blocks
#             x = self.upconvs[i](x)
#             # crop the current features from the encoder blocks,
#             # concatenate them with the current upsampled features,
#             # and pass the concatenated output through the current
#             # decoder block
#             encFeat = self.crop(encFeatures[i], x)
#             x = torch.cat([x, encFeat], dim=1)
#             x = self.dec_blocks[i](x)
#         # return the final decoder output
#         return x
#     def crop(self, encFeatures, x):
#         # grab the dimensions of the inputs, and crop the encoder
#         # features to match the dimensions
#         (_, _, H, W) = x.shape
#         encFeatures = CenterCrop([H, W])(encFeatures)
#         # return the cropped features
#         return encFeatures


# class UNet(Module):
#     def __init__(self, encChannels=(3, 16, 32, 64),
#          decChannels=(64, 32, 16),
#          nbClasses=1, retainDim=True,
#          outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
#         super().__init__()
#         # initialize the encoder and decoder
#         self.encoder = Encoder(encChannels)
#         self.decoder = Decoder(decChannels)
#         # initialize the regression head and store the class variables
#         self.head = Conv2d(decChannels[-1], nbClasses, 1)
#         self.retainDim = retainDim
#         self.outSize = outSize

#     def forward(self, x):
#         # grab the features from the encoder
#         encFeatures = self.encoder(x)
#         # pass the encoder features through decoder making sure that
#         # their dimensions are suited for concatenation
#         decFeatures = self.decoder(encFeatures[::-1][0],
#             encFeatures[::-1][1:])
#         # pass the decoder features through the regression head to
#         # obtain the segmentation mask
#         map = self.head(decFeatures)
#         # check to see if we are retaining the original output
#         # dimensions and if so, then resize the output to match them
#         if self.retainDim:
#             map = F.interpolate(map, self.outSize)
#         # return the segmentation map
#         return torch.sigmoid(map)

class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, False)
        self.up2 = up(512, 128, False)
        self.up3 = up(256, 64, False)
        self.up4 = up(128, 64, False)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)