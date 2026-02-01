#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:58:56 2023

@author: lauransotomayor
"""
#%%
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

#### testing parameters
# import sys
# import os
# # Add the project root directory to Python path
# sys.path.append('/media/laura/Extreme SSD/code/fvc_composition')
# from phase_3_models.unet_site_models import config_param
####

# Clear cache
torch.cuda.empty_cache()

import gc
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

#%%
######### SPECTRAL SQUEEZE-AND-EXCITATION MODULE #####################
'''Squeeze-and-Excitation module for spectral channel attention.
   Adaptively reweights spectral channels based on global spatial statistics.
   Produces per-sample channel weights of shape (B, C).
'''
class SpectralSE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        """
        Args:
            in_channels: Number of input spectral channels
            reduction_ratio: Reduction factor for the bottleneck layer
        """
        super(SpectralSE, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Global average pooling (spatial dimensions)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Spectral encoder: FC layers for channel recalibration
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Recalibrated tensor of shape (B, C, H, W)
            Channel weights of shape (B, C) are computed internally
        """
        b, c, _, _ = x.size()
        
        # Squeeze: Global spatial pooling -> (B, C, 1, 1)
        y = self.gap(x)
        
        # Flatten to (B, C)
        y = y.view(b, c)
        
        # Excitation: Channel-wise weights -> (B, C)
        weights = self.fc(y)
        
        # Reshape to (B, C, 1, 1) for broadcasting
        weights = weights.view(b, c, 1, 1)
        
        # Scale input by learned weights
        return x * weights.expand_as(x)


#%%
######### SPECTRAL ADAPTER MODULE #####################
'''Spectral encoder/adapter for multispectral and hyperspectral inputs.
   Combines 1x1 conv for spectral feature extraction with SE attention.
'''
class SpectralAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=8, adapter_type="mlp"):
        super().__init__()
        self.se = SpectralSE(in_channels, reduction_ratio)
        self.last_band_weights = None

        if adapter_type == "1x1":
            self.spectral_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        elif adapter_type == "mlp":
            hidden = max(out_channels * 2, out_channels)
            self.spectral_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        else:
            raise ValueError(f"adapter_type must be '1x1' or 'mlp', got {adapter_type}")

    def forward(self, x):
        x, w = self.se(x)           # return weights
        self.last_band_weights = w  # [B, Cin]
        
        return self.spectral_conv(x)



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
'''' Main UNetModel class implementing the U-Net architecture with Spectral Attention'''
## UNetModel with Spectral SE attention for multispectral/hyperspectral data
class UNetModel(nn.Module):
    def __init__(self, in_channels=config_param.IN_CHANNELS, out_channels=config_param.OUT_CHANNELS,               
                 features=[64, 128, 256, 512, 1024], dropout_prob=0.3, 
                 use_spectral_adapter=True, 
                 se_reduction_ratio=8,
                 adapter_out_channels=32,
                 adapter_type="mlp"):
        """
        Args:
            in_channels: Number of input spectral channels (5 for multispectral, 100+ for hyperspectral)
            out_channels: Number of output classes
            features: List of feature dimensions for each encoder/decoder level
            dropout_prob: Dropout probability
            use_spectral_adapter: Whether to use spectral adapter with SE attention
            se_reduction_ratio: Reduction ratio for SE blocks
        """
        super(UNetModel, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_spectral_adapter = use_spectral_adapter
        
        # Spectral Adapter: Process multispectral/hyperspectral input
        if use_spectral_adapter:
            self.spectral_adapter = SpectralAdapter(
                in_channels=in_channels, 
                out_channels=adapter_out_channels,
                reduction_ratio=se_reduction_ratio,
                adapter_type=adapter_type
            )
            # First encoder block takes adapted features
            encoder_in_channels = adapter_out_channels
        else:
            encoder_in_channels = in_channels

        # Down part of UNET- Downsampling path (encoder)
        for idx, feature in enumerate(features):
            # Each DoubleConv block reduces the spatial dimensions and increases the number of channels
            if idx == 0 and use_spectral_adapter:
                # First block takes adapted features
                self.downs.append(DoubleConv(encoder_in_channels, feature, dropout_prob))
            else:
                self.downs.append(DoubleConv(encoder_in_channels, feature, dropout_prob))
            encoder_in_channels = feature

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
        """
        Args:
            x: Input tensor (B, C_spectral, H, W)
        Returns:
            Output probability maps (B, num_classes, H, W)
        """
        skip_connections = []
        
        # Apply spectral adapter if enabled
        if self.use_spectral_adapter:
            x = self.spectral_adapter(x)

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
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x) # 1x1 convolution to reduce channels to the number of classes
        return x #logits

def unet_model_print():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # moves the model to the selected device.
    model = UNetModel(
        in_channels=config_param.IN_CHANNELS, 
        out_channels=config_param.OUT_CHANNELS,
        use_spectral_adapter=True,
        se_reduction_ratio=4
    ).to(config_param.DEVICE)  # Move model to the appropriate device
    print(model)
    print(f"\n{'='*60}")
    print(f"Model Configuration:")
    print(f"  Input Channels (Spectral): {config_param.IN_CHANNELS}")
    print(f"  Output Channels (Classes): {config_param.OUT_CHANNELS}")
    print(f"  Spectral Adapter: Enabled with SE attention")
    print(f"  SE Reduction Ratio: 4")
    print(f"{'='*60}\n")

    # Print a summary with an example input
    summary(model, (config_param.IN_CHANNELS, 256, 256))  # Input: spectral channels x 256 x 256

# To run and print the model summary
if __name__ == "__main__":
    # Create a U-Net model with specified hyperparameters
    # Hyperparameters:
    #   - Number of Layers: 5 down-sampling and 5 up-sampling layers (excluding the bottleneck)
    #   - Learning Rate: 0.001
    #   - Optimizer: Adam
    #   - Loss Function: CrossEntropyLoss / FocalLoss
    #   - Number of Channels: Input: 5 (multispectral) or 100+ (hyperspectral), Output: 4, Features: [64, 128, 256, 512, 1024]
    #   - Spectral Adapter: SE attention for adaptive spectral channel weighting
    #   - SE Reduction Ratio: 4 (reduces channels by 4x in SE bottleneck)
    #   - Kernel Size: 3x3
    #   - Pooling Size: 2x2
    #   - Batch Size: Not specified here, typically chosen based on hardware and memory constraints
    unet_model_print()
