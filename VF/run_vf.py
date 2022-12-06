import skrf as rf
import vectorfit
import numpy as np
import os
import glob
import random

save_root = "/scratch/04703/sravula/dip_results/vf"
data_root = "/scratch/04703/sravula/UTAFSDataNew"

file_list = []

for i in range(62):
    if i == 22:
        file_list.append(None)
        continue
    
    num_str = str(i) if i > 9 else "0"+str(i)
    case_name = "case"+num_str
    
    cur_path = os.path.join(data_root, case_name)
    
    os.chdir(cur_path)
    for file in glob.glob("*" + case_name + ".s*p"):
        if ".sampled" not in file:
            file_list.append(os.path.join(cur_path, file))

for i in range(62):
    filename = file_list[i]
    if filename is None:
        continue
    
    fitter = vectorfit.VectorFitter(filename)
    
    full_sweep = fitter.ground_truth.f
    fmin = min(full_sweep)
    fmax = max(full_sweep)
    
    LEN = len(full_sweep)
    f1 = np.linspace(fmin, fmax, int(LEN*0.01))
    f2 = np.linspace(fmin, fmax, int(LEN*0.02))
    f5 = np.linspace(fmin, fmax, int(LEN*0.05))
    f10 = np.linspace(fmin, fmax, int(LEN*0.1))
    
    try:
        vf1 = fitter.vector_fit("Uniform 1", f1)
        fit1 = vf1.fitted_network.s
    except:
        fit1 = None
        
    try:
        vf2 = fitter.vector_fit("Uniform 2", f2)
        fit2 = vf2.fitted_network.s
    except:
         fit2 = None
        
    try:
        vf5 = fitter.vector_fit("Uniform 5", f5)
        fit5 = vf5.fitted_network.s
    except:
        fit5 = None
    
    try:
        vf10 = fitter.vector_fit("Uniform 10", f10)
        fit10 = vf10.fitted_network.s
    except:
        fit10 = None
    
    pth1 = os.path.join(save_root, "case"+str(i)+"_1.npy")
    pth2 = os.path.join(save_root, "case"+str(i)+"_2.npy")
    pth5 = os.path.join(save_root, "case"+str(i)+"_5.npy")
    pth10 = os.path.join(save_root, "case"+str(i)+"_10.npy")

    np.save(pth1, fit1)
    np.save(pth2, fit2)
    np.save(pth5, fit5)
    np.save(pth10, fit10)