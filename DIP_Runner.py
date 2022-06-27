import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

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

