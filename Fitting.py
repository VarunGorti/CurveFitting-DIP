import torch
import torch.nn as nn

import random
import numpy as np

import time
from tqdm import tqdm

from skrf import Network, Frequency

import utils
from models import RESNET_BACKBONE, RESNET_HEAD, MODULAR_RESNET


def grab_sparams(root_pth, chip_num):
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
               num_iters_inner, reg_lambda_inner, start_noise_inner, noise_decay_inner):
    #sample chip
    x, y, z, kept_inds, net = grab_data_and_net(data_root=data_root, chip_num=chip_num, measurement_spacing=measurement_spacing, 
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
    
    return updated_net, test_mse

def reptile(backbone, data_root, device, measurement_spacing, num_measurements, 
            num_epochs, lr_outer, test_inds, train_inds, 
            lr_inner, num_iters_inner, reg_lambda_inner, start_noise_inner, noise_decay_inner,    
            ngf, kernel_size, causal, passive):
    
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
                p.grad = p.data - new_p.data
            
            optim.step()
            optim.zero_grad()
    
    return backbone, inner_test_losses, outer_test_losses

#########################################################################################
def outer_loop(backbone, lr_outer, train_inds, data_root, device, 
               ngf, kernel_size, causal, passive, 
               measurement_spacing, num_measurements,
               lr_inner, num_iters_inner, reg_lambda_inner, start_noise_inner, noise_decay_inner):
    
    optim = torch.optim.Adam(backbone.parameters(), lr=lr_outer)
    
    inner_test_losses = []
    
    train_shuffle = np.random.permutation(train_inds)
    
    for t in tqdm(train_shuffle):
        #sample chip
        x, y, z, kept_inds, net = grab_data_and_net(data_root=data_root, chip_num=t, measurement_spacing=measurement_spacing, 
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
        new_backbone = updated_net.backbone.cpu() 
        
        with torch.no_grad():
            inner_test_mse = nn.MSELoss()(x_hat, x).item()
            inner_test_losses.append(inner_test_mse)
            
            print("Inner Test MSE: ", inner_test_mse)
            
        #Calculate the gradient and update
        for p, new_p in zip(backbone.parameters(), new_backbone.parameters()):
            if p.grad is None:
                p.grad = p.data - new_p.data
            else:
                p.grad += p.data - new_p.data
        
        optim.step()
        optim.zero_grad()
    
    return backbone, inner_test_losses