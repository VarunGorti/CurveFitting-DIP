import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import os

from .utils import get_single_series, get_inds, plot_signal_and_measurements, get_paddings, Measurement_MSE_Loss
from .models import DCGAN, UNET, ENC_DEC, MULTISCALE_ENC_DEC, DILATE_MULTISCALE_ENC_DEC, OS_NET

data_params = {
    "chips": [0],
    "sparams": list(range(10)),
    "problem_type": ["equal", "random"],
    "num_meas": [0.01, 0.02, 0.05, 0.1, 0.2]
}

model_params = {
    "net_type": "ENC_DEC",
    "lr": 1e-3,
    "num_iter": 10000,
    "nz": 2,
    "ngf": 16
}

experiment_params = {
    "save_path": "/scratch/04703/sravula/dip_results",
    "exp_name": "run_1",
    "data_path": "/scratch/04703/sravula/UTAFSDataNew/X_RAW.pt"
}

def run_experiment(data_params, model_params, experiment_params, device):
    """
    Fits DIP on all the time series data
    """
    for chip_idx in data_params.chips:
        for sparam_idx in data_params.sparams:
            for problem_type in data_params.problem_type:
                for num_meas in data_params.num_meas:
                    x = get_single_series(data_path=experiment_params.data_path, sample_num=chip_idx, sparam_num=sparam_idx)

                    LENGTH = x.shape[-1]
                    kept_inds, missing_inds = get_inds(problem_type=problem_type, length=LENGTH, num_kept_samples=num_meas)
                    
                    y = torch.clone(x)[:, :, kept_inds]

                    x = x.to(device)
                    y = y.to(device)

                    L_PAD, R_PAD = get_paddings(LENGTH)
                    kept_inds = [k + L_PAD for k in kept_inds]

                    if device == 0:
                        tic = time.time()
                        print("STARTING: ")
                        print("CHIP " + str(chip_idx))
                        print("S-PARAM " + str(sparam_idx))
                        print(str(num_meas) + " " + problem_type.upper() + " MEASUREMENTS")

                    outputs, train_losses = run_dip(y=y, device=device, kept_inds=kept_inds, model_params=model_params, output_size=LENGTH)

                    if device == 0:
                        toc = time.time()
                        print("FINISHED!")
                        print("TIME: ", str(toc - tic))
                        print()
                    
                    outputs = torch.cat(outputs, dim=0)

                    

def run_dip(y, device, kept_inds, model_params, output_size):
    if model_params.net_type == "ENC_DEC":
        net = ENC_DEC(bs=1, nz=model_params.nz, ngf=model_params.ngf, output_size=output_size, nc=2)
    else
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
            outputs.append(out.detach().clone().cpu())
            train_losses.append(train_loss.item())
    
    return outputs, train_losses