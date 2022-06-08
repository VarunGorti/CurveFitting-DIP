import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_parts import * 

class DCGAN(nn.Module):
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1):
        """
        Args:
            bs: the batch size
            nz: the channel depth of the initial random seed
            ngf: base number of filters per layer
            output_size: the desired output length
            nc: number of channels in the output
        """
        super().__init__()
        
        self.bs = bs
        self.nz = nz
        self.ngf = ngf
        self.output_size = output_size
        self.nc = nc
        
        num_layers = int(np.ceil(np.log2(output_size))) - 1 #number of upsampling layers
        
        layers = [Up_NoCat(nz, ngf*num_layers)] #added one manually, so now need -1 upsampling layers
        for l in range(num_layers - 1):
            layers.append(Up_NoCat(ngf*(num_layers-l), ngf*(num_layers-l-1)))
        
        self.z = nn.Parameter(torch.randn((bs, nz, 2)))
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
        x = torch.clamp(x, min=-1., max=1.)
        x = self.unpad(x)

        return x

    def forward_with_z(self):
        return self.forward(self.z)