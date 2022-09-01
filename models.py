import numpy as np
import random
import math

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

        if np.ceil(np.log2(output_size)) == np.floor(np.log2(output_size)):
            self.unpad = nn.Identity()
        
    def forward(self, x):
        x = self.input(x)
        x = self.conv_net(x)
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
        
        #NOTE trying smaller num_layers now! Used to be - 1
        num_layers = int(np.ceil(np.log2(output_size))) - 5 #number of down/up sampling layers
        num_layers = max(num_layers, 5)
        padded_len = 2 ** int(np.ceil(np.log2(output_size)))
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, padded_len)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, padded_len), requires_grad=False))
        
        ###########
        #NET STUFF#
        ###########
        # self.input = OutConv(nz, ngf)
        # self.output = OutConv(ngf, nc)
        self.input = nn.Conv1d(nz, ngf, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.output = nn.Conv1d(ngf, nc, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        
        encoder = []
        decoder = []
        for l in range(num_layers):
            ch_1 = ngf * (l + 1)
            ch_2 = ngf * (l + 2)
            encoder.append(Down(ch_1, ch_2))
            
            ch_3 = 2 * ngf * (num_layers - l + 1) - ngf #account for concatenation
            ch_4 = ngf * (num_layers - l)
            decoder.append(Up(ch_3, ch_4))
        
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        
    def forward(self, x):
        x = self.input(x)
        
        enc_outs = [x]
        for layer in self.encoder:
            enc_outs.append(layer(enc_outs[-1]))
        
        for layer in self.decoder:
            x = layer(enc_outs[-1], enc_outs[-2])
            enc_outs = enc_outs[:-2]
            enc_outs.append(x)
        
        x = self.output(x)
        x = nn.Tanh()(x)
        
        return x
    
    def forward_with_z(self):
        return self.forward(self.z)
    
    @torch.no_grad()
    def perturb_noise(self, std=0.1):
        self.z += torch.randn_like(self.z) * std

class ENC_DEC(nn.Module):
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1, optimize_z=False, kernel_size=3):
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
        
        #NOTE trying smaller num_layers now! Used to be - 1
        num_layers = int(np.ceil(np.log2(output_size))) - 5 #number of down/up sampling layers
        num_layers = max(num_layers, 5)
        padded_len = 2 ** int(np.ceil(np.log2(output_size)))
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, padded_len)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, padded_len), requires_grad=False))
        
        ###########
        #NET STUFF#
        ###########
        # self.input = OutConv(nz, ngf)
        # self.output = OutConv(ngf, nc)
        self.input = nn.Conv1d(nz, ngf, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.output = nn.Conv1d(ngf, nc, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        
        encoder = []
        decoder = []
        for l in range(num_layers):
            ch_1 = ngf * (l + 1)
            ch_2 = ngf * (l + 2)
            encoder.append(Down(ch_1, ch_2, kernel_size=kernel_size))
            
            ch_3 = ngf * (num_layers - l + 1)
            ch_4 = ngf * (num_layers - l)
            decoder.append(Up_NoCat(ch_3, ch_4, kernel_size=kernel_size))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self, x):
        x = self.input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        x = nn.Tanh()(x)
        
        return x
    
    def forward_with_z(self):
        return self.forward(self.z)
    
    @torch.no_grad()
    def perturb_noise(self, std=0.1):
        self.z += torch.randn_like(self.z) * std

class MULTISCALE_ENC_DEC(nn.Module):
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
        
        #NOTE trying smaller num_layers now! Used to be - 1
        num_layers = int(np.ceil(np.log2(output_size))) - 3 #number of down/up sampling layers
        num_layers = max(num_layers, 5)
        padded_len = 2 ** int(np.ceil(np.log2(output_size)))
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, padded_len)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, padded_len), requires_grad=False))
        
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
            # num_scales_down = 2 ** int(np.ceil(np.log2(num_layers - l)))
            num_scales_down = 2 ** int((num_layers - l - 1) / 2)
            encoder.append(MultiScaleDown(ch_1, ch_2, num_scales_down))
            
            ch_3 = ngf * (num_layers - l + 1)
            ch_4 = ngf * (num_layers - l)
            # num_scales_up = 2 ** int(np.ceil(np.log2(l + 1)))
            num_scales_up = 2 ** int(l / 2)
            decoder.append(MultiScaleUp_NoCat(ch_3, ch_4, num_scales_up))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        ###########
        #UNPADDING#
        ###########
        if np.ceil(np.log2(output_size)) == np.floor(np.log2(output_size)):
            self.unpad = nn.Identity()
        
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

class DILATE_MULTISCALE_ENC_DEC(nn.Module):
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
        
        #NOTE trying smaller num_layers now! Used to be - 1
        num_layers = int(np.ceil(np.log2(output_size))) - 3 #number of down/up sampling layers
        num_layers = max(num_layers, 5)
        padded_len = 2 ** int(np.ceil(np.log2(output_size)))
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, padded_len)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, padded_len), requires_grad=False))
        
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
            # num_scales_down = 2 ** int(np.ceil(np.log2(num_layers - l)))
            num_scales_down = 2 ** int((num_layers - l - 1) / 2)
            encoder.append(Dilate_MultiScale_Block(ch_1, ch_2, num_scales_down, downsample=True))
            
            ch_3 = ngf * (num_layers - l + 1)
            ch_4 = ngf * (num_layers - l)
            # num_scales_up = 2 ** int(np.ceil(np.log2(l + 1)))
            num_scales_up = 2 ** int(l / 2)
            decoder.append(Dilate_MultiScale_Block(ch_3, ch_4, num_scales_up, downsample=False))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        ###########
        #UNPADDING#
        ###########
        if np.ceil(np.log2(output_size)) == np.floor(np.log2(output_size)):
            self.unpad = nn.Identity()
        
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

class OS_NET(nn.Module):
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
        
        #NOTE trying smaller num_layers now! Used to be - 1
        num_layers = int(np.ceil(np.log2(output_size))) - 3 #number of down/up sampling layers
        num_layers = max(num_layers, 5)
        padded_len = 2 ** int(np.ceil(np.log2(output_size)))
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, padded_len)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, padded_len), requires_grad=False))
        
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
            # num_scales_down = 2 ** int(np.ceil(np.log2(num_layers - l)))
            num_scales_down = 2 ** int((num_layers - l - 1) / 2)
            encoder.append(OS_Block(ch_1, ch_2, num_scales_down, downsample=True))
            
            ch_3 = ngf * (num_layers - l + 1)
            ch_4 = ngf * (num_layers - l)
            # num_scales_up = 2 ** int(np.ceil(np.log2(l + 1)))
            num_scales_up = 2 ** int(l / 2)
            decoder.append(OS_Block(ch_3, ch_4, num_scales_up, downsample=False))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        ###########
        #UNPADDING#
        ###########
        if np.ceil(np.log2(output_size)) == np.floor(np.log2(output_size)):
            self.unpad = nn.Identity()
        
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