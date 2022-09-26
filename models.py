import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_parts import * 

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
            kernel_size: can be a list - then must be length num_layers, symmetric, ordered from encoder.
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
        self.kernel_size = kernel_size
        
        #NOTE trying smaller num_layers now! Used to be - 1
        num_layers = int(np.ceil(np.log2(output_size))) - 5 #number of down/up sampling layers
        num_layers = max(num_layers, 5)
        
        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, output_size)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, output_size), requires_grad=False))
        
        ###########
        #NET STUFF#
        ###########
        self.input = OutConv(nz, ngf)
        self.output = OutConv(ngf, nc)

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers
        
        encoder = []
        decoder = []
        for l in range(num_layers):
            ch_1 = ngf * (l + 1)
            ch_2 = ngf * (l + 2)
            encoder.append(Down(ch_1, ch_2, kernel_size=kernel_size[l]))
            
            ch_3 = ngf * (num_layers - l + 1)
            ch_4 = ngf * (num_layers - l)
            decoder.append(Up_NoCat(ch_3, ch_4, kernel_size=kernel_size[num_layers - l - 1]))
        
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

class RES_UNET(nn.Module):
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1, optimize_z=False, kernel_size=3):
        """
        Args:
            bs: the batch size
            nz: the channel depth of the initial random seed
            ngf: base number of filters per layer. can be a list - then must be length num_layers, symmetric, ordered from encoder.
            output_size: the desired output length
            nc: number of channels in the output
            optimize_z: whether to optimize over the random input to the network
            kernel_size: can be a list - then must be length num_layers, symmetric, ordered from encoder.
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
        self.kernel_size = kernel_size
        
        #NOTE trying smaller num_layers now! Used to be - 1
        num_layers = int(np.ceil(np.log2(output_size))) - 5 #number of down/up sampling layers
        num_layers = max(num_layers, 5)

        ###########
        #  INPUT  #
        ###########
        if optimize_z:
            self.z = nn.Parameter(torch.randn((bs, nz, output_size)))
        else:
            self.register_buffer('z', torch.randn((bs, nz, output_size), requires_grad=False))
        
        ###########
        #NET STUFF#
        ###########
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers
        
        if not isinstance(ngf, list):
            ngf = [ngf * (l + 1) for l in range(num_layers)]

        self.output = nn.Sequential(
            OutConv(ngf[0], nc),
            nn.Tanh()
        )

        #the first encoder layer is not a downsampling layer
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for l in range(num_layers - 1):
            if l == 0:
                self.encoder.append(
                    InputResidualConv(in_channels=nc, out_channels=ngf[0], kernel_size=kernel_size[0])
                )
            else:
                self.encoder.append(
                    ResidualConv(in_channels=ngf[l-1], out_channels=ngf[l], kernel_size=kernel_size[l], downsample=True)
                )
            self.decoder.append(
                ResidualConv(in_channels=2*ngf[l], out_channels=ngf[l], kernel_size=kernel_size[l], downsample=False)
            )
            self.upsamples.append(
                UpConv(in_channels=ngf[l+1], out_channels=ngf[l])
            )
        
        self.middle = ResidualConv(in_channels=ngf[-2], out_channels=ngf[-1], kernel_size=kernel_size[-1], downsample=True)

    def forward(self, x):
        #encode
        out = x
        intermediate_outs = []
        for enc_layer in self.encoder:
            out = enc_layer(out)
            intermediate_outs.append(out)
        
        #bottleneck
        out = self.middle(out)

        #decode
        i = -1
        for up_layer, dec_layer in zip(self.upsamples[::-1], self.decoder[::-1]):
            out = up_layer(out)
            out = torch.cat([out, intermediate_outs[i]], dim=1)
            out = dec_layer(out)
            i -= 1

        #output
        out = self.output(out)

        return out

    def forward_with_z(self):
        return self.forward(self.z)
    
    @torch.no_grad()
    def perturb_noise(self, std=0.1):
        self.z += torch.randn_like(self.z) * std