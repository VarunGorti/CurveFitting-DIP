import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import os

from .utils import get_single_series, get_inds, plot_signal_and_measurements, get_paddings, Measurement_MSE_Loss

data_params = {
    "chips": [0],
    "sparams": list(range(10)),
    "problem_type": ["equal", "random"],
    "num_meas": [0.01, 0.02, 0.05, 0.1, 0.2]
}

model_params = {
    "net_type": "ENC_DEC",
    "lr": 1e-3,
    "num_iter": 3000,
    "nz": 512,
    "ngf": 64
}

experiment_params = {
    "save_path": "/scratch/04703/sravula/dip_results",
    "exp_name": "run_1",
    "data_path": "/scratch/04703/sravula/UTAFSDataNew/X_RAW.pt"
}

def run_experiment(data_params, model_params, experiment_params, device):
    for chip_idx in data_params.chips:
        for sparam_idx in data_params.sparams:
            for problem_type in data_params.problem_type:
                for num_meas in data_params.num_meas:
                    outputs = run_dip(data_params, model_params, experiment_params, device)

                    

def run_dip(data_params, model_params, experiment_params, device):
    #GRAB DATA
    x = get_single_series(data_path=data_params.data_path, sample_num=chip_idx, sparam_num=sparam_idx)
