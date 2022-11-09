import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(mid_channels, affine=False),
            nn.LeakyReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2
        
        self.single_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.single_conv(x)

class Down(nn.Module):
    """Conv, Down with avgpool, BN, LeakyReLU, then single conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2
        
        self.maxpool_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.AvgPool1d(2), 
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU(),
            SingleConv(out_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_NoCat(nn.Module):
    """Upscaling then double conv, with no concatenation"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='linear')
        self.conv = DoubleConv(in_channels, out_channels, in_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        
        return x

class OutConv(nn.Module):
    """1x1 convolutions to get correct output channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class InputResidualConv(nn.Module):
    """
    Takes the input through 2 paths and sums the output.
    Path 1: Conv -> BN -> LeakyReLU -> Conv
    Path 2: 1x1 Conv  
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, use_skip=True):
        super().__init__()

        self.use_skip = use_skip

        pad = (kernel_size - 1) // 2

        self.input_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
        )

        if self.use_skip:
            self.input_skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            )
    
    def forward(self, x):
        if self.use_skip:
            return self.input_layer(x) + self.input_skip(x)
        else:
            return self.input_layer(x)

class ResidualConv(nn.Module):
    """
    Takes the input through 2 paths and sums the output.
    Path 1: BN -> LeakyReLU -> Conv -> AvgPool -> BN -> LReLU -> Conv
    Path 2: 1x1 Conv -> AvgPool.
    If argument "downsample" is false, then no avg pooling    

    AvgPool uses ceil_mode=True so for odd-sized inputs, output_len = (input_len + 1)/2.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False, use_skip=True):
        super().__init__()

        self.use_skip = use_skip

        pad = (kernel_size - 1) // 2

        if downsample:
            self.conv_block = nn.Sequential(
                nn.BatchNorm1d(in_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
                nn.AvgPool1d(2, ceil_mode=True), 
                nn.BatchNorm1d(out_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
            )

            if self.use_skip:
                self.conv_skip = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.AvgPool1d(2, ceil_mode=True) 
                )

        else:
            self.conv_block = nn.Sequential(
                nn.BatchNorm1d(in_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
                nn.BatchNorm1d(out_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
            )

            if self.use_skip:
                self.conv_skip = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) 
                )
    
    def forward(self, x):
        if self.use_skip:
            return self.conv_block(x) + self.conv_skip(x)
        else:
            return self.conv_block(x)

class UpConv(nn.Module):
    """
     Linear Upsampling -> 1x1 conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) 
        )

    def forward(self, x):
        return self.up_conv(x)

def crop_and_cat(x1, x2):
    """
    Crops x1 to match x2 in length, then cats the two and return.
    """
    diff = x1.size()[-1] - x2.size()[-1]

    if diff > 0:
        return torch.cat([x1[...,:-diff], x2], dim=1)
    else:
        return torch.cat([x1, x2], dim=1)
