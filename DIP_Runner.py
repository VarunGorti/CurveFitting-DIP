import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import random
import numpy as np
import os

import utils
from utils import get_inds, plot_signal_and_measurements, get_paddings, Measurement_MSE_Loss

import models
from models import ENC_DEC

def run_experiment(data_params, model_params, experiment_params, device):
    """
    Fits DIP on all the time series data
    """
    exp_path = os.path.join(experiment_params.save_path, experiment_params.exp_name) #exp root
    problem_type = data_params.problem_type

    #will do multiple runs for the same num_meas depending on the device
    num_meas = data_params.num_meas[device]

    for chip_idx in data_params.chips:
        #grab the proper save path
        chip_path = os.path.join(exp_path, "chip" + str(chip_idx))
        output_path = os.path.join(chip_path, str(int(num_meas*100)) + ".pt")

        #Grab the correct chip
        X = torch.load(experiment_params.data_path)[chip_idx] #[1, 2, NF, N_Sparams]
        N_Sparams = X.shape[-1]

        #make the tensor to hold the results
        out_tensor = torch.zeros_like(X)

        for sparam_idx in range(N_Sparams):
            #grab the specific sparam we want
            x = X[:, :, :, sparam_idx].clone() #[1, 2, NF]
            x = x.to(device)
            LENGTH = x.shape[-1]

            #hold the best outputs
            best_train_mse = float('inf')
            best_output = None
            best_train_losses = None

            #grab the measurements just once
            kept_inds, _ = get_inds(problem_type=problem_type, length=LENGTH, num_kept_samples=num_meas)
            
            y = torch.clone(x)[:, :, kept_inds]
            y = y.to(device)

            # #save the image!
            # signal_path = os.path.join(chip_path, "XY_Sparam" + str(sparam_idx) + "_M" + str(int(num_meas*100)) + ".png")
            # plot_signal_and_measurements(x, y, kept_inds, signal_path)

            L_PAD, R_PAD = get_paddings(LENGTH)
            PADDED_LEN = 2**int(np.ceil(np.log2(LENGTH)))
            kept_inds = [k + L_PAD for k in kept_inds]

            if device == 0:
                tic = time.time()
                print("STARTING: ")
                print("CHIP " + str(chip_idx))
                print("S-PARAM " + str(sparam_idx))
                print(str(num_meas) + " " + problem_type.upper() + " MEASUREMENTS")
                print()

            for t in range(4):
                outputs, train_losses = run_dip(y=y, device=device, kept_inds=kept_inds, model_params=model_params, output_size=PADDED_LEN)

                if device == 0:
                    toc = time.time()
                    print("FINISHED RUN " + str (t))
                    print("TIME: ", str(toc - tic))
                    print()
                
                if np.min(train_losses) < best_train_mse:
                    best_train_mse = np.min(train_losses)
                    best_output = outputs[np.argmin(train_losses)].clone()[..., L_PAD:-R_PAD] #[1, 2, NF]
                    best_train_losses = train_losses
                
            if device == 0:
                toc = time.time()
                print("BEST MSE: ", str(best_train_mse))
                print("TIME: ", str(toc - tic))
                print()

            #store the tensor
            out_tensor[:, :, :, sparam_idx] = best_output.clone().cpu()

            loss_path = os.path.join(chip_path, "Loss_Sparam" + str(sparam_idx) + "_M" + str(int(num_meas*100)) + ".pt")
            torch.save(best_train_losses, loss_path)
        
        #save the output tensor
        if device == 0:
            print("FINISHED CHIP " + str(chip_idx))

        torch.save(out_tensor, output_path)
        
                    
def run_dip(y, device, kept_inds, model_params, output_size):
    """
    Runs a single Deep Image Prior fit.

    Args:
        y: Observations. Torch tensor [1, 2, m] on device.
        device: CUDA device number. int.
        kept_inds: Indices of kept observations. List with length m.
        model_params: Model and training specifications. Namespace.
        output_size: The desired length of the output from DIP. int.
    
    Returns:
        outputs: All the intermediate DIP outputs. List with length model_params.num_iter, each element is Torch tensor [1, 2, output_size] on device.
        train_losses: The intermediate training losses. List with length model_params.num_iter.
    """

    if model_params.net_type == "ENC_DEC":
        net = ENC_DEC(bs=1, nz=model_params.nz, ngf=model_params.ngf, output_size=output_size, nc=2)
    else:
        raise NotImplementedError("THIS NETWORK TYPE IS UNSUPPORTED")
    
    net = net.to(device)
    net = net.train()

    optim = torch.optim.Adam(net.parameters(), lr=model_params.lr)

    criterion = Measurement_MSE_Loss(kept_inds=kept_inds)
    criterion = criterion.to(device)

    outputs = []
    train_losses = []

    for i in range(model_params.num_iter):
        net.perturb_noise(0.05) 
        
        optim.zero_grad()
        
        out = net.forward_with_z()
        train_loss = criterion(out, y)
        
        train_loss.backward()
        optim.step()

        with torch.no_grad():
            outputs.append(out.detach().clone())
            train_losses.append(train_loss.item())
    
    return outputs, train_losses