import argparse
import yaml
import sys
import os
import torch
import numpy as np
import torch.distributed as dist

import utils
from utils import set_all_seeds

import DIP_Runner
from DIP_Runner import run_experiment

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    data_params = {
        "chips": [5, 15, 25, 35, 45, 55],
        "problem_type": "equal",
        "num_meas": [0.01, 0.02, 0.05, 0.1]
    }

    model_params = {
        "net_type": "ENC_DEC",
        "lr": 1e-3,
        "num_iter": 10000,
        "nz": 2,
        "ngf": 128
    }

    experiment_params = {
        "save_path": "/scratch/04703/sravula/dip_results",
        "exp_name": "run_2",
        "data_path": "/scratch/04703/sravula/UTAFSDataNew/NEW_DATA.pt"
    }

    #set up device
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        device = rank % 4 #torch.cuda.device_count()
    else:
        print("ERROR - NO RANK FOUND")
        sys.exit(0)
    
    #set up param namespaces
    data_params = dict2namespace(data_params)
    model_params = dict2namespace(model_params)
    experiment_params = dict2namespace(experiment_params)

    #set up saving folders
    if not os.path.exists(experiment_params.save_path) and device == 0:
        os.makedirs(experiment_params.save_path)
    
    exp_path = os.path.join(experiment_params.save_path, experiment_params.exp_name)
    if not os.path.exists(exp_path) and device == 0:
        os.makedirs(exp_path)

    for chip_idx in data_params.chips:
        chip_path = os.path.join(exp_path, "chip"+str(chip_idx))
        if not os.path.exists(chip_path) and device == 0:
            os.makedirs(chip_path)
    
    #save the configs
    if device == 0:
        with open(os.path.join(exp_path, 'data_params.yml'), 'w') as f:
            yaml.dump(data_params, f, default_flow_style=False)

        with open(os.path.join(exp_path, 'model_params.yml'), 'w') as f:
            yaml.dump(model_params, f, default_flow_style=False)

        with open(os.path.join(exp_path, 'experiment_params.yml'), 'w') as f:
            yaml.dump(experiment_params, f, default_flow_style=False)

    #set seeds
    set_all_seeds(2022)

    #run the experiments
    if device == 0:
        print("STARTING EXPERIMENTS!")
    
    run_experiment(data_params, model_params, experiment_params, device)

if __name__ == '__main__':
    main()