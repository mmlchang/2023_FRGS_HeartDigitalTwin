import os
import nibabel as nib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sam.sam import SAM
from torchsummary import summary

import timm

########################################################################################

class conv2D_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv2D = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.conv2D(x)

########################################################################################
        
class decoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = conv2D_block(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x

########################################################################################

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, pretrained=True):
        super().__init__()

        # Load Mamba encoder from timm
        self.encoder = timm.create_model('mambaout_small.in1k', pretrained=pretrained, features_only=True, in_chans=in_ch)
        encoder_channels = self.encoder.feature_info.channels()  # Channels at each stage

        # Decoder blocks (UNet-style)
        self.decoder4 = decoderBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.decoder3 = decoderBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.decoder2 = decoderBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])
        # self.decoder1 = decoderBlock(encoder_channels[0], encoder_channels[0], 32)

        self.UpSample2D_1 = nn.ConvTranspose2d(96, 96, 2, stride=2)
        self.Conv2D_1 = conv2D_block(96, 64)
        self.UpSample2D_2 = nn.ConvTranspose2d(64, 24, 2, stride=2)
        self.Conv2D_final = nn.Conv2d(24, out_ch, kernel_size=1)

    def forward(self, x):
        print("Input x:", x.shape)
        # print("x:", x.shape)

        # Encoder forward (returns list of feature maps)
        features = self.encoder(x)
        #print(self.encoder.feature_info.channels())
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features] # Convert NHWC -> NCHW

        #print(features)
        e1, e2, e3, e4 = features  # shallow to deep
        #print("e1:", e1.shape)
        #print("e2:", e2.shape)
        #print("e3:", e3.shape)
        #print("e4:", e4.shape)

        # Decoder
        d1 = self.decoder4(e4, e3)
        #print("d1:", d1.shape)

        d2 = self.decoder3(d1, e2)
        #print("d2:", d2.shape)

        d3 = self.decoder2(d2, e1)
        #print("d3:", d3.shape)

        d5 = self.UpSample2D_1(d3)
        #print("d5:", d5.shape)

        d5_1_1 = self.Conv2D_1(d5)
        #print("d5_1_1:", d5_1_1.shape)

        d5_1 = self.UpSample2D_2(d5_1_1)
        #print("d5_1:", d5_1.shape)

        d6 = self.Conv2D_final(d5_1)
        #print("d6:", d6.shape)

        return d6 # shape should be Batch, Class(4), 256, 256
        
########################################################################################

## Test Run

model = UNet2D(in_ch =1, out_ch = 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

weights_path = ".../unet_best.pth"
state_dict = torch.load(weights_path, map_location=device) 
model.load_state_dict(state_dict)
model.eval()  # evaluation mode

x = torch.randn(1, 1, 256, 256) # input.

with torch.no_grad():
    y = model(x) # logit output

probs = F.softmax(y, dim=1)       # shape: [1, num classes, 256, 256]
pred_mask = torch.argmax(probs, dim=1) # shape: [1, 256, 256]
pred_mask = pred_mask.squeeze(0).cpu() # shape: [256, 256]



