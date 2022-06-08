import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_parts import * 

class DCGAN(nn.Module):
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1, optimize_z=False):
        """
        Args:
            bs: the batch size
            nz: the channel depth of the initial random seed
            ngf: base number of filters per layer
            output_size: the desired output length
            nc: number of channels in the output
            optimize_z: whether to optimize over the random input to the network
        """
        super().__init__()
        
        self.bs = bs
        self.nz = nz
        self.ngf = ngf
        self.output_size = output_size
        self.nc = nc
        self.optimize_z = optimize_z
        
        num_layers = int(np.ceil(np.log2(output_size))) - 1 #number of upsampling layers
        
        layers = [Up_NoCat(nz, ngf*num_layers)] #added one manually, so now need -1 upsampling layers
        for l in range(num_layers - 1):
            layers.append(Up_NoCat(ngf*(num_layers-l), ngf*(num_layers-l-1)))
        
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, 2)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, 2), requires_grad=False))
        self.conv_net = nn.Sequential(*layers)
        self.output = OutConv(ngf, nc)
        
        unpad_num = (2**(num_layers+1)) - output_size
        l_unpad, r_unpad = unpad_num // 2, unpad_num - unpad_num // 2
        unpad_idxs = np.arange(2**(num_layers+1))[l_unpad:]
        unpad_idxs = unpad_idxs[:-r_unpad]
        
        self.unpad = lambda x: x[:, :, unpad_idxs]
        
    def forward(self, x):
        x = self.conv_net(x)
        x = self.output(x)
        x = nn.Tanh()(x)
        x = self.unpad(x)

        return x

    def forward_with_z(self):
        return self.forward(self.z)
    
class UNET(nn.Module):
    def __init__(self, bs, ngf=64, output_size=1024, nc=1, optimize_z=False):
        """
        Args:
            bs: the batch size
            ngf: base number of filters per layer
            output_size: the desired output length
            nc: number of channels in the output
            optimize_z: whether to optimize over the random input to the network
        """
        super().__init__()
        
        self.bs = bs
        self.ngf = ngf
        self.output_size = output_size
        self.nc = nc
        self.optimize_z = optimize_z
        
        self.input = DoubleConv(nc, ngf)
        
        self.down1 = Down(ngf, ngf*2)
        self.down2 = Down(ngf*2, ngf*4)
        self.down3 = Down(ngf*4, ngf*8)
        self.down4 = Down(ngf*8, ngf*8)
        
        self.up1 = Up(ngf*16, ngf*4)
        self.up2 = Up(ngf*8, ngf*2)
        self.up3 = Up(ngf*4, ngf)
        self.up4 = Up(ngf*2, ngf)
        
        self.output = OutConv(ngf, nc)
        
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nc, output_size)))
        else:
            self.register_buffer('z', torch.randn((bs, nc, output_size), requires_grad=False))
        
    def forward(self, x):
        x1 = self.input(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.output(x)
        
        x = nn.Tanh()(x)
        
        return x
    
    def forward_with_z(self):
        return self.forward(self.z)
        
class ENC_DEC(nn.Module):
    def __init__(self, bs, ngf=64, output_size=1024, nc=1, optimize_z=False):
        """
        Args:
            bs: the batch size
            ngf: base number of filters per layer
            output_size: the desired output length
            nc: number of channels in the output
            optimize_z: whether to optimize over the random input to the network
        """
        super().__init__()
        
        self.bs = bs
        self.ngf = ngf
        self.output_size = output_size
        self.nc = nc
        self.optimize_z = optimize_z
        
        self.input = DoubleConv(nc, ngf)
        
        num_layers = int(np.ceil(np.log2(output_size))) - 1 #number of downsampling layers
        
        
        
        self.output = OutConv(ngf, nc)
        
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nc, output_size)))
        else:
            self.register_buffer('z', torch.randn((bs, nc, output_size), requires_grad=False))
        
    def forward(self, x):
        x = self.input(x)
        
        x = self.output(x)
        
        x = nn.Tanh()(x)
        
        return x
    
    def forward_with_z(self):
        return self.forward(self.z)