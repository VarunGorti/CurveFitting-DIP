import torch
import torch.nn as nn

import random
import numpy as np

import matplotlib.pyplot as plt

import time
from tqdm import tqdm

from skrf import Network, Frequency

import utils
from models import RESNET_BACKBONE, RESNET_HEAD, MODULAR_RESNET


def grab_sparams(root_pth, chip_num):
    """
    Returns a torch tensor of s-parameters with shape [1, 2*num_unique_sparams, num_freqs]
        given a path and a chip number.
    """
    #first grab the chip
    chip_dict = utils.get_network_from_file(root_pth, chip_num)
    
    out_network = chip_dict["network"]
    out_freqs = out_network.frequency
    
    #resample to minimum length if necessary
    MIN_LEN = 1000
    
    if out_freqs.npoints < MIN_LEN:
        scale_fac = int(np.ceil(MIN_LEN / out_freqs.npoints))
        new_len = scale_fac * (out_freqs.npoints - 1) + 1 #this is smarter scaling that just divides current spacing
        
        out_network.resample(new_len)
        out_freqs = out_network.frequency
    
    #convert to unique s-parameters tensor
    out_matrix_re = out_network.s.real
    out_matrix_im = out_network.s.imag
    out_matrix = np.stack((out_matrix_re, out_matrix_im), axis=-1)

    out_sparams = utils.matrix_to_sparams(out_matrix)

    out_sparams = out_sparams.reshape(1, -1, out_freqs.npoints)

    return torch.tensor(out_sparams)

def fit_DIP(model, y, z, 
            lr, num_iter, 
            train_loss, train_reg, reg_lambda=0, 
            start_noise=None, noise_decay=None):
    """
    Runs DIP for a single set of given measurements.

    Returns the fitted network and the final output.
    """
    
    #we can make the optmizer within the function since we don't need the stats between fits
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(num_iter):
        optim.zero_grad()
        
        #get the output with or without additive noise in the input
        if (start_noise is not None) and (noise_decay is not None):
            noisy_z = z + torch.randn_like(z) * start_noise
            out = model.forward(noisy_z)
            start_noise *= noise_decay
        else:
            out = model.forward(z)
        
        #loss and regularization
        error = train_loss(out, y) 
        if reg_lambda > 0:
            reg = reg_lambda * train_reg(out)
            loss = error + reg
        else:
            loss = error

        loss.backward()
        optim.step()
    
    return model, out

def grab_data_and_net(data_root, chip_num, measurement_spacing, num_measurements,
                      ngf, kernel_size, causal, passive, backbone):
    """
    Grabs the ground truth s-parameters for a chip along with measurements, adjoint
        solution as the network input, the indices of the measurements, and a network 
        head for the dimension of the data. 
    """
    
    x = grab_sparams(data_root, chip_num)

    #grab the appropriate measurements
    kept_inds, missing_inds = utils.get_inds(measurement_spacing, x.shape[-1], num_measurements)

    y = torch.clone(x)[:, :, kept_inds]

    z = torch.clone(x)
    z[:, :, missing_inds] = 0

    #set up the clone network and head and make modular net
    net_head = RESNET_HEAD(nz=x.shape[1],
                           ngf_in_out=ngf,
                           nc=x.shape[1],
                           output_size=x.shape[-1],
                           kernel_size=kernel_size,
                           causal=causal,
                           passive=passive)

    backbone_clone = backbone.make_clone() 

    net = MODULAR_RESNET(backbone=backbone_clone,
                         head=net_head)
    
    return x, y, z, kept_inds, net

def train_step(data_root, chip_num, measurement_spacing, num_measurements, ngf,
               kernel_size, causal, passive, backbone, device, lr_inner, 
               num_iters_inner, reg_lambda_inner, start_noise_inner, noise_decay_inner,
               plot_output=False):
    """
    Performs DIP on a single sample: grabs the measurements, fits the network, calculates the loss.

    Returns the fitted network and the test mse for the sample. 
    """
    #sample chip
    x, y, z, kept_inds, net = grab_data_and_net(data_root=data_root, chip_num=chip_num, 
                                measurement_spacing=measurement_spacing, 
                                num_measurements=num_measurements, ngf=ngf, kernel_size=kernel_size, 
                                causal=causal, passive=passive, backbone=backbone)
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    net = net.to(device)

    #set up losses and regularisations
    criterion = utils.Measurement_MSE_Loss(kept_inds=kept_inds, per_param=True, reduction="mean")
    criterion = criterion.to(device)

    regularizer = utils.Smoothing_Loss(per_param=True, reduction="mean")
    regularizer = regularizer.to(device)

    #Run DIP and get the metrics 
    updated_net, x_hat = fit_DIP(model=net, y=y, z=z, 
                                 lr=lr_inner, num_iter=num_iters_inner, 
                                 train_loss=criterion, train_reg=regularizer, reg_lambda=reg_lambda_inner, 
                                 start_noise=start_noise_inner, noise_decay=noise_decay_inner) 
    with torch.no_grad():
        test_mse = nn.MSELoss()(x_hat, x).item()
    
        if plot_output:
            x_mag = utils.sparams_to_mag(x)
            out_mag = utils.sparams_to_mag(x_hat)
            dip_errors_mag = x_mag - out_mag 

            _, axes = plt.subplots(3,1, figsize=(8, 6))
            axes = axes.flatten()

            for i in range(x_mag.shape[1]):
                axes[0].plot(x_mag[0,i].cpu(), label=str(i))
            axes[0].set_title("Ground Truth Magnitude Spectrum")
            axes[0].set_ylim(0,1)

            for i in range(x_mag.shape[1]):
                axes[1].plot(out_mag[0,i].detach().cpu(), label=str(i))
            axes[1].set_title("DIP Output Magnitude Spectrum")
            axes[1].set_ylim(0,1)
                
            for i in range(x_mag.shape[1]):
                axes[2].plot(dip_errors_mag[0,i].detach().cpu(), label=str(i))
            axes[2].set_title("DIP Errors Magnitude Spectrum")
            axes[2].set_ylim(-1,1)
            
            plt.show()
    
    return updated_net, test_mse

def reptile(backbone, data_root, device, measurement_spacing, num_measurements, 
            num_epochs, lr_outer, test_inds, train_inds, 
            lr_inner, num_iters_inner, reg_lambda_inner, start_noise_inner, noise_decay_inner,    
            ngf, kernel_size, causal, passive):
    """
    Performs REPTILE-style updates for a given backbone network over a training dataset. 

    Returns the updated network, test losses for the inner optimization, and test losses for the meta opt. 
    """
    
    optim = torch.optim.Adam(backbone.parameters(), lr=lr_outer)
    
    inner_test_losses = []
    outer_test_losses = []
    
    for epoch in range(num_epochs):

        print("STARTING EPOCH " + str(epoch) + "\n")

        #testing - don't update parameters, just track the metrics
        print("TESTING\n")

        for test_chip_ind in tqdm(test_inds):
            _, outer_test_mse = train_step(data_root=data_root, chip_num=test_chip_ind, 
                                           measurement_spacing=measurement_spacing, num_measurements=num_measurements, 
                                           ngf=ngf, kernel_size=kernel_size, causal=causal, passive=passive, 
                                           backbone=backbone, device=device, lr_inner=lr_inner, 
                                           num_iters_inner=num_iters_inner, reg_lambda_inner=reg_lambda_inner, 
                                           start_noise_inner=start_noise_inner, noise_decay_inner=noise_decay_inner)

            outer_test_losses.append(outer_test_mse)

            print("CHIP " + str(test_chip_ind) + " TEST MSE: " + str(outer_test_mse) + "\n")
        
        #training - update params and track metrics
        print("TRAINING\n")

        train_shuffle = np.random.permutation(train_inds)

        for train_chip_ind in tqdm(train_shuffle):
            updated_net, inner_test_mse = train_step(data_root=data_root, chip_num=train_chip_ind, 
                                           measurement_spacing=measurement_spacing, num_measurements=num_measurements, 
                                           ngf=ngf, kernel_size=kernel_size, causal=causal, passive=passive, 
                                           backbone=backbone, device=device, lr_inner=lr_inner, 
                                           num_iters_inner=num_iters_inner, reg_lambda_inner=reg_lambda_inner, 
                                           start_noise_inner=start_noise_inner, noise_decay_inner=noise_decay_inner)

            inner_test_losses.append(inner_test_mse)

            print("CHIP " + str(train_chip_ind) + " TEST MSE: " + str(inner_test_mse) + "\n")

            #update params
            new_backbone = updated_net.backbone.cpu()

            for p, new_p in zip(backbone.parameters(), new_backbone.parameters()):
                if type(p.grad) == type(None):
                    dummy_loss = torch.sum(p)
                    dummy_loss.backward()
                
                p.grad.copy_(p - new_p)

                #TODO convert the loss to a centered l2 loss using parameters_to_vector
                #also only print epoch mean loss
                #also print the norm of the gradient 
                #also try supervised learning
            
            optim.step()
            optim.zero_grad()
    
    return backbone, inner_test_losses, outer_test_losses
