import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, bs, nz, ngf=64, output_size=1024, nc=1, optimize_z=False, kernel_size=3, num_layers=None, use_skip=True, causal_passive=False, p_dropout=0.0):
        """
        Args:
            bs: the batch size
            nz: the channel depth of the initial random seed
            ngf: base number of filters per layer. can be a list - then must be length num_layers, symmetric, ordered from encoder.
            output_size: the desired output length
            nc: number of channels in the output
            optimize_z: whether to optimize over the random input to the network
            kernel_size: can be a list - then must be length num_layers, symmetric, ordered from encoder.
            num_layers: the number of layers in the encoder/decoder. 
            use_skip: whether or not to use skip connections within blocks. 
            causal_passive: if True, adds a causality and passivity layer at the end of the network.
                                Also does one more resolution scale at the end of the decoder.
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
        self.num_layers = num_layers
        self.use_skip = use_skip
        self.causal_passive = causal_passive
        self.p_dropout = p_dropout
        assert self.p_dropout >= 0
        
        #NOTE (num_layers - 1) is the number of resolution scales (i.e. number of up/down samples)
        #e.g. for num_layers = 5, there are 4 scales - divides original resolution by 2^4 = 16  
        if self.num_layers is None:
            self.num_layers = int(np.ceil(np.log2(self.output_size))) - 4 #this will give (smallest resolution <= 32 pixels)
            self.num_layers = max(self.num_layers, 5) #ensure minimum size of net

        ###########
        #  INPUT  #
        ###########
        if self.optimize_z:
            self.z = nn.Parameter(torch.randn((self.bs, self.nz, self.output_size)))
        else:
            self.register_buffer('z', torch.randn((self.bs, self.nz, self.output_size), requires_grad=False))
        
        ###########
        #NET STUFF#
        ###########
        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size] * self.num_layers
        
        if not isinstance(self.ngf, list):
            self.ngf = [self.ngf * (l + 1) for l in range(self.num_layers)]

        #the first encoder layer is not a downsampling layer
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        self.upsample_dropouts = nn.ModuleList()

        for l in range(self.num_layers - 1):
            if l > (self.num_layers - 1) / 2:
                if l == 0:
                    self.encoder.append(
                        InputResidualConv(in_channels=self.nc, out_channels=self.ngf[0], kernel_size=self.kernel_size[0], use_skip = self.use_skip)
                    )
                else:
                    self.encoder.append(
                        ResidualConv(in_channels=self.ngf[l-1], out_channels=self.ngf[l], kernel_size=self.kernel_size[l], downsample=True, use_skip = self.use_skip, p_dropout=self.p_dropout)
                    )
                self.decoder.append(
                    ResidualConv(in_channels=2*self.ngf[l], out_channels=self.ngf[l], kernel_size=self.kernel_size[l], downsample=False, use_skip = self.use_skip, p_dropout=self.p_dropout)
                )
                self.upsamples.append(
                    UpConv(in_channels=self.ngf[l+1], out_channels=self.ngf[l], p_dropout=self.p_dropout)
                )
            else:
                if l == 0:
                    self.encoder.append(
                        InputResidualConv(in_channels=self.nc, out_channels=self.ngf[0], kernel_size=self.kernel_size[0], use_skip = self.use_skip)
                    )
                else:
                    self.encoder.append(
                        ResidualConv(in_channels=self.ngf[l-1], out_channels=self.ngf[l], kernel_size=self.kernel_size[l], downsample=True, use_skip = self.use_skip, p_dropout=0)
                    )
                self.decoder.append(
                    ResidualConv(in_channels=2*self.ngf[l], out_channels=self.ngf[l], kernel_size=self.kernel_size[l], downsample=False, use_skip = self.use_skip, p_dropout=0)
                )
                self.upsamples.append(
                    UpConv(in_channels=self.ngf[l+1], out_channels=self.ngf[l], p_dropout=0)
                )

        
        self.middle = ResidualConv(in_channels=self.ngf[-2], out_channels=self.ngf[-1], kernel_size=self.kernel_size[-1], downsample=True, use_skip = self.use_skip, p_dropout=self.p_dropout)

        if self.causal_passive:
            self.output = nn.Sequential(
                UpConv(in_channels=self.ngf[0], out_channels=self.ngf[0], p_dropout=self.p_dropout),
                ResidualConv(in_channels=self.ngf[0], out_channels=self.nc//2, mid_channels=self.nc, kernel_size=1, downsample=False, use_skip=self.use_skip, p_dropout=self.p_dropout),
                CausalityLayer(F=self.output_size),
                PassivityLayer()
            )
        else:
            self.output = nn.Sequential(
                OutConv(self.ngf[0], self.nc),
                nn.Tanh()
            )

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
            out = crop_and_cat(out, intermediate_outs[i])
            out = dec_layer(out)
            i -= 1

        #output
        out = self.output(out)

        return out

    def forward_with_z(self, perturb_noise_std=None):
        if perturb_noise_std is None:
            return self.forward(self.z)
        else:
            return self.forward(self.z + torch.randn_like(self.z) * perturb_noise_std)
    
    def make_causal_passive(self):
        if self.causal_passive:
            return

        self.output = nn.Sequential(
            UpConv(in_channels=self.ngf[0], out_channels=self.ngf[0]),
            ResidualConv(in_channels=self.ngf[0], out_channels=self.nc//2, mid_channels=self.nc, kernel_size=1, downsample=False, use_skip=self.use_skip),
            CausalityLayer(F=self.output_size),
            # PassivityLayer()
            nn.tanh()
        )
        
    @torch.no_grad()
    def set_z(self, latent_seed):
        if isinstance(latent_seed, np.ndarray):
            self.z = torch.from_numpy(latent_seed).type(self.z.dtype).to(self.z.device).requires_grad_(self.z.requires_grad)
            
        elif torch.is_tensor(latent_seed):
            self.z = latent_seed.detach().clone().type(self.z.dtype).to(self.z.device).requires_grad_(self.z.requires_grad)
            
        else:
            raise ValueError("argument must be a numpy array or a torch tensor")
        
        return