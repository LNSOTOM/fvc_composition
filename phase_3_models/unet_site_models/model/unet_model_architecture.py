#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:58:56 2023

@author: lauransotomayor
"""
# 1. Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from PIL import Image
import tifffile as tiff
from pytorch_lightning.loggers import TensorBoardLogger  # Corrected import
import time

import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision

from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import rasterio

import config_param

# Clear cache
torch.cuda.empty_cache()

import gc
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

#%%
######### 1_MODEL UNET ARCHITECTURE #####################
'''Basic building block for the U-Net consisting of two consecutive Conv-BatchNorm-ReLU-Dropout layers'''
# 2. Model Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.conv(x)

#%%
'''' Main UNetModel class implementing the U-Net architecture'''
## option a with device selection: UNetModel implements the U-Net architecture
class UNetModel(nn.Module):
    def __init__(self, in_channels=config_param.IN_CHANNELS, out_channels=config_param.OUT_CHANNELS, features=[64, 128, 256, 512, 1024], dropout_prob=0.3):
        super(UNetModel, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET- Downsampling path (encoder)
        for feature in features:
            # Each DoubleConv block reduces the spatial dimensions and increases the number of channels
            self.downs.append(DoubleConv(in_channels, feature, dropout_prob))
            in_channels = feature

        # Up part of UNET - Upsampling path (decoder)
        for feature in reversed(features):
            # Upsampling followed by a DoubleConv block to increase spatial dimensions and decrease channels
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature, dropout_prob))

        # Bottom part of UNET - Bottleneck part (bridge between encoder and decoder)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_prob)
        # Last top conv - Final convolution layer to map to output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsample path - Forward pass through downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsample path with skip connections - Forward pass through upsampling path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                # x = F.interpolate(x, size=skip_connection.shape[2:])
                # x = F.interpolate(x, size=skip_connection.shape[2:], mode='nearest')
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x) # 1x1 convolution to reduce channels to the number of classes
        return F.softmax(x, dim=1)   # Apply softmax along the channel dimension to get probabilities

def unet_model_print():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # moves the model to the selected device.
    model = UNetModel(in_channels=config_param.IN_CHANNELS, out_channels=config_param.OUT_CHANNELS).to(config_param.DEVICE)  # Move model to the appropriate device
    print(model)

    # Print a summary with an example input
    summary(model, (config_param.IN_CHANNELS, 256, 256))  # Assuming an input image size of 256x256 with 5 channels

# To run and print the model summary
if __name__ == "__main__":
    # Create a U-Net model with specified hyperparameters
    # Hyperparameters:
    #   - Number of Layers: 5 down-sampling and 5 up-sampling layers (excluding the bottleneck)
    #   - Learning Rate: 0.001
    #   - Optimizer: Adam
    #   - Loss Function: CrossEntropyLoss
    #   - Number of Channels: Input: 5, Output: 4, Features: [64, 128, 256, 512, 1024]
    #   - Kernel Size: 3x3
    #   - Pooling Size: 2x2
    #   - Batch Size: Not specified here, typically chosen based on hardware and memory constraints
    unet_model_print()
