import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

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
    meta_losses = []

    epoch_pbar = tqdm(range(num_epochs))
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}")

        inner_test_losses_epoch = []
        outer_test_losses_epoch = []
        meta_losses_epoch = []

        #testing - don't update parameters, just track the metrics
        testing_pbar = tqdm(test_inds)
        for test_chip_ind in testing_pbar:
            testing_pbar.set_description(f"Testing - Sample {test_chip_ind}")

            _, outer_test_mse = train_step(data_root=data_root, chip_num=test_chip_ind, 
                                           measurement_spacing=measurement_spacing, num_measurements=num_measurements, 
                                           ngf=ngf, kernel_size=kernel_size, causal=causal, passive=passive, 
                                           backbone=backbone, device=device, lr_inner=lr_inner, 
                                           num_iters_inner=num_iters_inner, reg_lambda_inner=reg_lambda_inner, 
                                           start_noise_inner=start_noise_inner, noise_decay_inner=noise_decay_inner)
            #update progress
            outer_test_losses.append(outer_test_mse)
            outer_test_losses_epoch.append(outer_test_mse)

            testing_pbar.set_postfix({'sample mse': outer_test_mse,
                                      'epoch mse': np.mean(outer_test_losses_epoch)})
            epoch_pbar.set_postfix({'mean outer mse': np.mean(outer_test_losses), 
                                    'mean inner mse': 'N/A',
                                    'mean meta loss': 'N/A'})
        
        #training - update params and track metrics
        training_pbar = tqdm(np.random.permutation(train_inds))
        for train_chip_ind in training_pbar:
            training_pbar.set_description(f"Training - Sample {train_chip_ind}")

            updated_net, inner_test_mse = train_step(data_root=data_root, chip_num=train_chip_ind, 
                                           measurement_spacing=measurement_spacing, num_measurements=num_measurements, 
                                           ngf=ngf, kernel_size=kernel_size, causal=causal, passive=passive, 
                                           backbone=backbone, device=device, lr_inner=lr_inner, 
                                           num_iters_inner=num_iters_inner, reg_lambda_inner=reg_lambda_inner, 
                                           start_noise_inner=start_noise_inner, noise_decay_inner=noise_decay_inner)
            #update params
            new_backbone = updated_net.backbone.cpu()
            new_backbone.requires_grad_(False)

            params_cur = nn.utils.parameters_to_vector(backbone.parameters())
            params_new = nn.utils.parameters_to_vector(new_backbone.parameters())

            meta_loss = 0.5 * torch.sum((params_cur - params_new)**2)
            meta_loss.backward()

            optim.step()
            optim.zero_grad()

            #update progress bar
            inner_test_losses.append(inner_test_mse)
            inner_test_losses_epoch.append(inner_test_mse)

            meta_losses.append(meta_loss.item())
            meta_losses_epoch.append(meta_loss.item())

            training_pbar.set_postfix({'sample mse': inner_test_mse,
                                       'epoch mse': np.mean(inner_test_losses_epoch),
                                       'sample metaloss': meta_loss.item(),
                                       'epoch metaloss': np.mean(meta_losses_epoch)})
            epoch_pbar.set_postfix({'mean outer mse': np.mean(outer_test_losses), 
                                    'mean inner mse': np.mean(inner_test_losses),
                                    'mean meta loss': np.mean(meta_losses)})

    return backbone, inner_test_losses, outer_test_losses, meta_losses
