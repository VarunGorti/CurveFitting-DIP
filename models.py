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
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1, optimize_z=False):
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
        
        num_layers = int(np.ceil(np.log2(output_size))) - 1 #number of upsampling layers
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, 2**(num_layers+1))))
        else:
#             self.register_buffer('z', torch.randn((bs, nz, 2**(num_layers+1)), requires_grad=False))
            a = torch.arange(0, 2**(num_layers+1)) 
            a = a.unsqueeze(0).unsqueeze(0).repeat(bs, nz, 1)
            
            b = torch.arange(0, nz) * math.pi / 2**(num_layers+1)
            b = b.view(1, -1, 1)
            
            z = torch.sin(a * b)
            self.register_buffer('z', z.clone().requires_grad_(False))
            
        
        ###########
        #NET STUFF#
        ###########
        self.input = OutConv(nz, ngf)
        self.output = OutConv(ngf, nc)
        
        self.encoder = []
        self.decoder = []
        for l in range(num_layers):
            ch_1 = ngf * (l + 1)
            ch_2 = ngf * (l + 2)
            self.encoder.append(Down(ch_1, ch_2))
            
            ch_3 = 2 * ngf * (num_layers - l + 1) - ngf #account for concatenation
            ch_4 = ngf * (num_layers - l)
            self.decoder.append(Up(ch_3, ch_4))
        
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        
        ###########
        #UNPADDING#
        ###########
        if np.ceil(np.log2(output_size)) == np.floor(np.log2(output_size)):
            self.unpad = nn.Identity()
        else:
            self.unpad = ResizePool(ngf, ngf, 2**(num_layers+1), output_size)
        
    def forward(self, x):
        x = self.input(x)
        
        enc_outs = [x]
        for layer in self.encoder:
            enc_outs.append(layer(enc_outs[-1]))
        
        for layer in self.decoder:
            x = layer(enc_outs[-1], enc_outs[-2])
            enc_outs = enc_outs[:-2]
            enc_outs.append(x)
        
        x = self.unpad(x)
        x = self.output(x)
        x = nn.Tanh()(x)
        
        return x
    
    def forward_with_z(self):
        return self.forward(self.z)
    
    @torch.no_grad()
    def perturb_noise(self, std=0.1):
        self.z += torch.randn_like(self.z) * std
        
class ENC_DEC(nn.Module):
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1, optimize_z=False):
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
        
        num_layers = int(np.ceil(np.log2(output_size))) - 1 #number of upsampling layers
        
        ###########
        #  INPUT  #
        ###########
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