import torch
import torch.nn as nn
import random
import numpy as np
import os

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    return

def get_single_series(data_path, sample_num, sparam_num, num_chan=2):
    """
    Grabs the Real or (Real, Im) series for a single sample and a single s-parameter.

    Args:
        data_path: Array filled with individual chip tensors.
                   Array with tensors of shape [L, 10, 2] (samples, S-Params, real/im).
        sample_num: Chip number to get.  
                    Int <= len(data).
        sparam_num: S-parameter number to grab for the given chip.
                    Int in [0, 9].
        num_chan: Set to 1 to grab just the real or 2 to grab real and imaginary responses.
                  Int in {1, 2}.
    Returns:
        x: Tensor with shape [1, num_chan, L].
    """
    x = torch.load(data_path)[sample_num][:, sparam_num, :] #(LEN, 2)
    
    if num_chan == 1:
        x = x[:, 0].unsqueeze(1) #(LEN, 1)
    
    x = x.unsqueeze(0) #(1, LEN, 1/2)
    x = x.permute(0, 2, 1) #(1, 1/2, LEN)
    
    return x

def get_inds(problem_type, length, num_kept_samples):
    """
    Given a number of samples to keep and a problem type, returns indices to keep from a list.

    Args:
        problem_type: What type of inpainting problem we are dealing with.
                      String in {"random", "equal", "forecast", "full"}.
        length: The original length of the signal.
                Int.
        num_kept_samples: The number (or proportion) of samples to keep.
                          Int <= length OR float <= 1.0.
    
    Returns:
        kept_inds: Array with (sorted, increasing) indices of samples to keep from the original signal.
                   Numpy array of Ints.
        missing_inds: Array with  (sorted, increasing) indices of discarded sampled from the original signal.
                      Numpy array of Ints.
    """
    if isinstance(num_kept_samples, float) and num_kept_samples <= 1.0:
        num_kept_samples = int(length * num_kept_samples)
        num_kept_samples = max(num_kept_samples, 1)

    if problem_type=="random":
        kept_inds = np.random.choice(length, num_kept_samples, replace=False)
    elif problem_type=="equal":
        kept_inds = np.arange(0, length, (length // num_kept_samples))
    elif problem_type=="forecast":
        kept_inds = np.arange(0, num_kept_samples)
    elif problem_type=="full":
        kept_inds = np.arange(0, length)
    else:
        raise NotImplementedError("THIS PROBLEM TYPE IS UNSUPPORTED")
    
    missing_inds = np.array([t for t in range(length) if t not in kept_inds])
    
    return np.sort(kept_inds), np.sort(missing_inds)

def plot_signal_and_measurements(x, y, kept_inds, fname=None):
    """
    Given a ground truth signal, plot the signal, interpolated observations, and raw observations.

    Args:
        x: Signal to plot. 
           Tensor with shape [1, NC, L].
        y: Observations to plot.
           Tensor with shape [1, NC, M], with M <= L. 
        kept_inds: Indices kept from the signal, used as horizontal axis to plot observations.
                   Array with len M.
        fname: Filename to save the plot as. Default=None, in which case will do plt.show().
               String, ending in desired file extension (e.g. .png).
    """
    NC = x.shape[1]

    fig, axes = plt.subplots(3,1, figsize=(16, 12))
    axes = axes.flatten()

    axes[0].plot(x[0,0,:].cpu().flatten(), label="real")
    if NC == 2:
        axes[0].plot(x[0,1,:].cpu().flatten(), label="imaginary")
    axes[0].legend()
    axes[0].set_title("ORIGINAL SIGNAL")

    axes[1].plot(kept_inds, y[0,0,:].cpu().flatten(), label="real")
    if NC == 2:
        axes[1].plot(kept_inds, y[0,1,:].cpu().flatten(), label="imaginary")
    axes[1].legend()
    axes[1].set_title("OBSERVED MEASUREMENTS - LINEAR INTERPOLATION")

    axes[2].scatter(kept_inds, y[0,0,:].cpu().flatten(), label="real")
    if NC == 2:
        axes[2].scatter(kept_inds, y[0,1,:].cpu().flatten(), label="imaginary")
    axes[2].legend()
    axes[2].set_title("OBSERVED MEASUREMENTS")

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname=fname, bbox_inches='tight')

    return

def get_paddings(in_len):
    """
    Given an input length, gives left and right padding factors to get to the 
        next closest power of 2 length (with centering).
    
    Args:
        in_len: Original length.
                Int.
    
    Returns:
        [L_PAD, R_PAD]: Left and right padding factors to center original length
                            in next highest power of 2 length.
                        Int, Int.
    """
    PADDED_LEN = 2**int(np.ceil(np.log2(in_len)))

    DIFF = (PADDED_LEN - in_len)

    L_PAD, R_PAD = DIFF // 2, DIFF - DIFF // 2

    return L_PAD, R_PAD

def grab_chip_data(root_pth, chip_num):
    """
    Given a root path and a chip number, grab all the relevant info for a chip.
    
    Args:
        root_pth: Root of the folder containing chip info.
        chip_num: Chip number.
    
    Returns:
        out_dict: Dictionary with all relevant chip data.
                  Entries:
                      "gt_freqs": Ground truth frequencies.
                                  (1D numpy array).
                      "gt_matrix": Ground truth S-param matrix. 
                                  (4D numpy array) - Axes are [Freq, In_port, Out_port, Re/Im].
                      "vf_matrix": VF interpolation given samples.
                                   (4D numpy array).
                      "y_freqs": Observed sample frequencies chosen by VF.
                                 (1D numpy array).
                      "y_matrix": Ground truth matrix sampled at the observed frequencies.
                                  (4D numpy array).
    """
    from skrf import Network, Frequency
    
    #Grab the correct folder
    chip_num = str(chip_num) if chip_num > 9 else "0" + str(chip_num)
    fname = os.path.join(root_pth, "case"+chip_num)
    
    def grab_network_info(folder_pth, net_str):
        """
        Helper function that takes a string, searches a given folder for touchstone 
            files matching the string, grabs the S-param and frequency data, and 
            formats and returns the S-param matrix and the corresponding frequencies.
        """
        
        #grab the correct file we want
        children = os.listdir(folder_pth)
        children = [f for f in children if net_str in f]
        
        #there should only be a single file with the given string
        if len(children) > 1:
             return None, None
        else:
            children = children[0]
        
        #grab the actual network stuff now
        data_path = os.path.join(folder_pth, children)
        
        out_network = Network(data_path)
        
        out_matrix_re = out_network.s.real.astype(np.float32)
        out_matrix_im = out_network.s.imag.astype(np.float32)
        out_matrix = np.stack((out_matrix_re, out_matrix_im), axis=-1)

        out_freqs = out_network.f.astype(np.float32).squeeze()
        
        return out_matrix, out_freqs
    
    #now make the proper filename strings and grab the gt, VF, and y data
    gt_str = str(chip_num) + ".s"
    vf_str = str(chip_num) + ".HLAS.s"
    y_str = "SIEMENS_AFS_SAMPLE_POINT_SIMULATIONS.s"

    gt_matrix, gt_freqs = grab_network_info(fname, gt_str)
    vf_matrix, _ = grab_network_info(fname, vf_str)
    y_matrix, y_freqs = grab_network_info(fname, y_str)

    out_dict = {"gt_matrix": gt_matrix,
                "gt_freqs": gt_freqs,
                "vf_matrix": vf_matrix,
                "y_matrix": y_matrix,
                "y_freqs": y_freqs}
            
    return out_dict

def frequencies_to_samples(gt_freqs, y_freqs):
    """
    Takes ground truth and observed frequency values and returns the corresponding list
        of sampled indices.

    Args:
        gt_freqs: Array of the ground truth frequency values that data is sampled at. 
        y_freqs: Array of frequencies where we have observed samples from the true signal.

    Returns:
        kept_inds: Array of the sample indices that are kept based on elements of gt_freqs
                            that are observed in y_freqs.
        A: Forward operator for the inpainting problem based on kept_inds. 
           2D array with a subset of the rows from the identity matrix. 
    """

    kept_inds = []

    #go through each kept frequency and add the index of the corresponding true frequency
    for obs_freq in y_freqs:
        
        for og_ind, gt_freq in enumerate(gt_freqs):
            
            if obs_freq == gt_freq:
                kept_inds.append(og_ind)
                break
    
    n = len(gt_freqs)

    kept_inds = np.array(kept_inds)

    A = np.eye(n)[kept_inds]

    return kept_inds, A

def matrix_to_sparams(data_matrix):
    """
    Takes a raw 4D sparam matrix and returns a 3D array of sparam series.

    Args:
        data_matrix: Raw 4D sparam matrix. 
                     (4D numpy array) - Axes are [Freq, In_port, Out_port, Re/Im].

    Returns:
        output: 3D array of time series.
                (3D numpy array) - [Unique Port Pair, Re/Im, Freq].
    """
    num_freqs = data_matrix.shape[0]
    num_ports = data_matrix.shape[1]

    num_unique = int(num_ports * (num_ports + 1) / 2)
    
    output = np.zeros((num_unique, 2, num_freqs))

    t = 0
    for i in range(num_ports):
        for j in range(i+1):
            output[t] = np.copy(data_matrix[:, i, j, :]).transpose()
            t += 1

    return output

class Measurement_MSE_Loss(nn.Module):
    """
    Given a signal x, observed measurements y, and observed indices kept_inds, 
        return the mse over the measurements of x vs true measurements y 
    """

    def __init__(self, kept_inds):
        super().__init__()

        self.kept_inds = kept_inds
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, y):
        return self.mse_loss(x[:, :, self.kept_inds], y)
