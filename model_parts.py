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
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
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
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)

class Down(nn.Module):
    """Conv, Down with maxpool, BN, LeakyReLU, then single conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2
        
        self.maxpool_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.MaxPool1d(2), 
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
            SingleConv(out_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, linear=True, kernel_size=3):
        super().__init__()

        # if linear, use the normal convolutions to reduce the number of channels
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is [N, C, L]
        diffL = x2.size()[-1] - x1.size()[-1]
        x1 = F.pad(x1, [diffL // 2, diffL - diffL // 2])
        
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class Up_NoCat(nn.Module):
    """Upscaling then double conv, with no concatenation"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
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

class MultiScaleDown(nn.Module):
    """Version of Down that uses multiple conv kernel sizes"""

    def __init__(self, in_channels, out_channels, num_scales):
        super().__init__()
        
        out_channels = out_channels // num_scales
        
        layers = []
        for scale in range(num_scales):
            next_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=(2*scale + 3), padding=scale+1, padding_mode='reflect', bias=False),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            layers.append(next_layer)
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        
        for layer in self.layers:
            outputs.append(layer(x))
        
        return torch.cat(outputs, dim=1)

class MultiScaleUp_NoCat(nn.Module):
    """Version of Up_NoCat with multiple convolution kernel sizes"""

    def __init__(self, in_channels, out_channels, num_scales):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
        out_channels = out_channels // num_scales
        
        layers = []
        for scale in range(num_scales):
            next_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=(2*scale + 3), padding=scale+1, padding_mode='reflect', bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            layers.append(next_layer)
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.up(x)
        
        outputs = []
        
        for layer in self.layers:
            outputs.append(layer(x))
        
        return torch.cat(outputs, dim=1)

class Dilate_MultiScale_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales, downsample=False):
        super().__init__()

        self.downsample = downsample
        if not downsample:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        start_layers = []
        mid_layers = []
        for scale in range(num_scales):
            dilation = scale + 1
            padding = dilation 

            first_layer = [nn.Conv1d(in_channels, out_channels//num_scales, kernel_size=3, dilation=dilation, padding=padding, padding_mode='reflect', bias=False)]
            if downsample:
                first_layer.append(nn.MaxPool1d(2))
            first_layer.extend([nn.BatchNorm1d(out_channels//num_scales), nn.LeakyReLU(inplace=True)])
            start_layers.append(nn.Sequential(*first_layer))

            mid_layers.append(nn.Sequential(
                nn.Conv1d(out_channels, out_channels//num_scales, kernel_size=3, dilation=dilation, padding=padding, padding_mode='reflect', bias=False),
                nn.BatchNorm1d(out_channels//num_scales),
                nn.LeakyReLU(inplace=True)
            ))
        
        self.start_layers = nn.ModuleList(start_layers)
        self.mid_layers = nn.ModuleList(mid_layers)

        self.last_layers = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        if not self.downsample:
            x = self.up(x)
        
        outputs = []
        for layer in self.start_layers:
            outputs.append(layer(x))
        x = torch.cat(outputs, dim=1)

        outputs = []
        for layer in self.mid_layers:
            outputs.append(layer(x))
        x = torch.cat(outputs, dim=1)

        x = self.last_layers(x)

        return x

def get_primes(n):
    """
    Grab all prime numbers from 1 to n
    """
    numbers = set(range(n, 1, -1))
    primes = []
    while numbers:
        p = numbers.pop()
        primes.append(p)
        numbers.difference_update(set(range(p*2, n+1, p)))
    primes.remove(2)
    return primes

class OS_Block(nn.Module):
    """
    A block as seen in OMNI-SCALE CNNS (https://openreview.net/pdf?id=PDYs7Z2XFGv)
    """
    def __init__(self, in_channels, out_channels, num_scales, downsample=False):
        super().__init__()

        #Grab the list of prime numbers
        prime_list = get_primes(2**10)[:num_scales]

        self.downsample = downsample
        if not downsample:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        start_layers = []
        # mid_layers = []
        for scale in range(num_scales):
            kernel_size = prime_list[scale]
            padding = (kernel_size - 1) // 2

            first_layer = [nn.Conv1d(in_channels, out_channels//num_scales, kernel_size=kernel_size, padding=padding, padding_mode='reflect', bias=False)]
            if downsample:
                first_layer.append(nn.MaxPool1d(2))
            first_layer.extend([nn.BatchNorm1d(out_channels//num_scales), nn.LeakyReLU(inplace=True)])
            first_layer.extend([nn.Conv1d(out_channels//num_scales, out_channels//num_scales, kernel_size=1, bias=False),
                                nn.BatchNorm1d(out_channels//num_scales),
                                nn.LeakyReLU(inplace=True)])
            start_layers.append(nn.Sequential(*first_layer))

            # mid_layers.append(nn.Sequential(
            #     nn.Conv1d(out_channels, out_channels//num_scales, kernel_size=kernel_size, padding=padding, padding_mode='reflect', bias=False),
            #     nn.BatchNorm1d(out_channels//num_scales),
            #     nn.LeakyReLU(inplace=True)
            # ))
        
        self.start_layers = nn.ModuleList(start_layers)
        # self.mid_layers = nn.ModuleList(mid_layers)

        # self.last_layers = nn.Sequential(
        #     nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(out_channels),
        #     nn.LeakyReLU(inplace=True)
        # )
    
    def forward(self, x):
        if not self.downsample:
            x = self.up(x)
        
        outputs = []
        for layer in self.start_layers:
            outputs.append(layer(x))
        x = torch.cat(outputs, dim=1)

        # outputs = []
        # for layer in self.mid_layers:
        #     outputs.append(layer(x))
        # x = torch.cat(outputs, dim=1)

        # x = self.last_layers(x)

        return x