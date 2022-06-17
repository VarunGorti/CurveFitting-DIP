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
        
        ###########
        #  PARAMS #
        ###########
        self.bs = bs
        self.nz = nz
        self.ngf = ngf
        self.output_size = output_size
        self.nc = nc
        self.optimize_z = optimize_z
        
        num_layers = int(np.ceil(np.log2(output_size))) - 1 #number of upsampling layers
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, 2)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, 2), requires_grad=False))
        
        ###########
        #NET STUFF#
        ###########
        self.input = OutConv(nz, ngf * (num_layers + 1))
        self.output = OutConv(ngf, nc)
        
        layers = []
        for l in range(num_layers):
            ch_1 = ngf * (num_layers - l + 1)
            ch_2 = ngf * (num_layers - l)
            layers.append(Up_NoCat(ch_1, ch_2))
        
        self.conv_net = nn.Sequential(*layers)
        
#         unpad_num = (2**(num_layers+1)) - output_size
#         l_unpad, r_unpad = unpad_num // 2, unpad_num - unpad_num // 2
#         unpad_idxs = np.arange(2**(num_layers+1))[l_unpad:]
#         unpad_idxs = unpad_idxs[:-r_unpad]
        
#         self.unpad = lambda x: x[:, :, unpad_idxs]

        if np.ceil(np.log2(output_size)) == np.floor(np.log2(output_size)):
            self.unpad = nn.Identity()
        else:
            self.unpad = ResizePool(ngf, ngf, 2**(num_layers+1), output_size)
        
    def forward(self, x):
        x = self.input(x)
        x = self.conv_net(x)
#         x = self.output(x)
#         x = nn.Tanh()(x)
#         x = self.unpad(x)
        x = self.unpad(x)
        x = self.output(x)
        x = nn.Tanh()(x)

        return x

    def forward_with_z(self):
        return self.forward(self.z)
    
    @torch.no_grad()
    def perturb_noise(self, std=0.1):
        self.z += torch.randn_like(self.z) * std
    
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
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1, optimize_z=False, init_z=None):
        """
        Args:
            bs: the batch size
            nz: the channel depth of the initial random seed
            ngf: base number of filters per layer
            output_size: the desired output length
            nc: number of channels in the output
            optimize_z: whether to optimize over the random input to the network
            init_z: the initial value for the input to the net, if desired
        """
        super().__init__()
        
        ###########
        #  PARAMS #
        ###########
        self.bs = bs
        self.nz = nz
        self.ngf = ngf
        self.output_size = output_size
        self.nc = nc
        self.optimize_z = optimize_z
        self.init_z = init_z
        
        num_layers = int(np.ceil(np.log2(output_size))) - 1 #number of upsampling layers
        
        ###########
        #  INPUT  #
        ###########
        if init_z is not None:
            if optimize_z:
                self.z = nn.Parameter(init_z.detach().clone().float())
            else:
                self.register_buffer('z', init_z.detach().clone().requires_grad_(False).float())
                
            nz = self.z.shape[1]
            self.nz = nz
        else:
            if optimize_z:
                self.z = nn.Parameter(torch.randn((bs, nz, 2**(num_layers+1))))
            else:
                self.register_buffer('z', torch.randn((bs, nz, 2**(num_layers+1)), requires_grad=False))
        
        ###########
        #NET STUFF#
        ###########
        self.input = OutConv(nz, ngf)
        self.output = OutConv(ngf, nc)
        
        encoder = []
        decoder = []
        for l in range(num_layers):
            ch_1 = ngf * (l + 1)
            ch_2 = ngf * (l + 2)
            encoder.append(Down(ch_1, ch_2))
            
            ch_3 = ngf * (num_layers - l + 1)
            ch_4 = ngf * (num_layers - l)
            decoder.append(Up_NoCat(ch_3, ch_4))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        ###########
        #UNPADDING#
        ###########
        if np.ceil(np.log2(output_size)) == np.floor(np.log2(output_size)):
            self.unpad = nn.Identity()
        else:
            self.unpad = ResizePool(ngf, ngf, 2**(num_layers+1), output_size)
        
    def forward(self, x):
        x = self.input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.unpad(x)
        x = self.output(x)
        x = nn.Tanh()(x)
        
        return x
    
    def forward_with_z(self):
        return self.forward(self.z)
    
    @torch.no_grad()
    def perturb_noise(self, std=0.1):
        self.z += torch.randn_like(self.z) * std