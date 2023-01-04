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

#NOTE DEPRECATED
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
                      String in {"random", "equal", "forecast", "full", "log", "sqrt"}.
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
        kept_inds = np.linspace(0, length-1, num=num_kept_samples, dtype=int)
    elif problem_type=="forecast":
        kept_inds = np.arange(0, num_kept_samples)
    elif problem_type=="full":
        kept_inds = np.arange(0, length)
    elif problem_type=="log": #NOTE fix collisions here and for sqrt
        kept_inds = np.geomspace(1, length, num=num_kept_samples, dtype=int) #geomspace can't take 0 as start index
        kept_inds = kept_inds - 1 
        kept_inds = np.sort(kept_inds) #making the list unique
        for i in range(1, len(kept_inds)):
            if kept_inds[i-1] == kept_inds[i]:
                kept_inds[i] = kept_inds[i] + 1 
    elif problem_type=="sqrt": 
        r = (length - 1) / ((num_kept_samples - 1)**2) #base quadratic factor
        kept_inds = [round(r * (s**2)) for s in range(num_kept_samples - 1)]
        kept_inds.append(length-1)
        kept_inds = np.array(kept_inds)
        kept_inds = np.sort(kept_inds) #making the list unique
        for i in range(1, len(kept_inds)):
            if kept_inds[i-1] == kept_inds[i]:
                kept_inds[i] = kept_inds[i] + 1 
    else:
        raise NotImplementedError("THIS PROBLEM TYPE IS UNSUPPORTED")
    
    missing_inds = np.array([t for t in range(length) if t not in kept_inds])
    
    return np.sort(kept_inds), np.sort(missing_inds)

#NOTE DEPRECATED
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

#NOTE DEPRECATED
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

def grab_chip_data(root_pth, chip_num, resample=False):
    """
    Given a root path and a chip number, grab all the relevant info for a chip.
    Converts the data type of the output to float32.
    
    Args:
        root_pth: Root of the folder containing chip info.
        chip_num: Chip number.
        resmple: If set to True, will also return a version of the ground truth data
                    that has been resampled to have (2^N >= #OG_Freqs) frequency values.
    
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
    import copy
    
    #Grab the correct folder
    chip_num = str(chip_num) if chip_num > 9 else "0" + str(chip_num)
    fname = os.path.join(root_pth, "case"+chip_num)
    
    def grab_network_info(folder_pth, net_str, resample=False):
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

        if resample:
            og_matrix_re = out_network.s.real
            og_matrix_im = out_network.s.imag
            og_matrix = copy.deepcopy(np.stack((og_matrix_re, og_matrix_im), axis=-1))

            og_freqs = copy.deepcopy(out_network.f.squeeze())
            
            og_len = len(og_freqs)
            new_len = 2 ** int(np.ceil(np.log2(og_len)))
            
            out_network.resample(new_len)
        
        out_matrix_re = out_network.s.real
        out_matrix_im = out_network.s.imag
        out_matrix = np.stack((out_matrix_re, out_matrix_im), axis=-1)

        out_freqs = out_network.f.squeeze()
        
        if resample:
            return og_matrix, og_freqs, out_matrix, out_freqs
        else: 
            return out_matrix, out_freqs
    
    #now make the proper filename strings and grab the gt, VF, and y data
    gt_str = str(chip_num) + ".s"
    vf_str = str(chip_num) + ".HLAS.s"
    y_str = "SIEMENS_AFS_SAMPLE_POINT_SIMULATIONS.s"

    if resample:
        og_matrix, og_freqs, gt_matrix, gt_freqs = grab_network_info(fname, gt_str, resample=resample)
    else:
        gt_matrix, gt_freqs = grab_network_info(fname, gt_str)    
    vf_matrix, _ = grab_network_info(fname, vf_str)
    y_matrix, y_freqs = grab_network_info(fname, y_str)

    out_dict = {"gt_matrix": gt_matrix.astype('float32'),
                "gt_freqs": gt_freqs,
                "vf_matrix": vf_matrix.astype('float32'),
                "y_matrix": y_matrix.astype('float32'),
                "y_freqs": y_freqs}

    if resample:
        out_dict["og_matrix"] = og_matrix.astype('float32')
        out_dict["og_freqs"] = og_freqs
            
    return out_dict

def matrix_to_network(data_matrix, data_freqs, name, resample_freqs=None):
    """
    Takes a raw 4D sparam matrix and returns a Network.

    Args:
        data_matrix: Raw 4D sparam matrix. 
                     (4D torch tensor) - Axes are [Freq, In_port, Out_port, Re/Im].
        data_freqs: Sampled frequencies. Must be same length as first axis of data_matrix.
                    (1D numpy array).   
        name: The name of the network.
              (String)
        resample_freqs: Frequencies to resample the data to. Default is None.
                        (1D numpy array). 
    
    Returns:
        net: A scikit-RF Network object.
    """
    from skrf import Network, Frequency
    import scipy
    
    #Convert the data properly to complex
    temp_data = data_matrix.detach().cpu().numpy().astype('float64')
    
    net_data = np.empty(temp_data.shape[:-1], dtype=np.complex128)
    net_data.real = temp_data[..., 0]
    net_data.imag = temp_data[..., 1]

    #Make the network
    net_freqs = Frequency.from_f(data_freqs, unit="hz")

    net = Network(frequency=net_freqs, s=net_data, name=name)

    #Check if we need to re-sample
    if resample_freqs is not None:
        new_freqs = Frequency.from_f(resample_freqs)
        net.resample(new_freqs)

    return net

def matrix_to_sparams(data_matrix):
    """
    Takes a raw 4D sparam matrix and returns a 3D array of sparam series.
    Output is float32.

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
    
    output = np.zeros((num_unique, 2, num_freqs), dtype='float32')

    t = 0
    for i in range(num_ports):
        for j in range(i+1):
            output[t] = np.copy(data_matrix[:, i, j, :]).transpose()
            t += 1

    return output

def sparams_to_matrix(sparams_data):
    """
    Function for converting a 3-D frequency series sparam data back to 
        a 4-D frequency matrix series.
    Output is the same datatype as the input.
    
    Args:
        sparams_data: 3D frequnecy series.
                      Torch tensor [1, 2 * unique_sparams, Freq].
    
    Returns:
        A: 4D matrix sparam frequency series. 
           Torch tensor [Freq, In_port, Out_port, Re/Im].
    """
    _, num_unique, num_freqs = sparams_data.shape
    num_unique = num_unique // 2

    num_ports = (-1 + np.sqrt(8*num_unique + 1)) // 2
    num_ports = int(num_ports)

    A = torch.zeros(num_freqs, num_ports, num_ports, 2).to(sparams_data.device).type(sparams_data.dtype)

    for i in range(num_ports):
        for j in range(i+1):
            A[:, i, j, 0] = sparams_data[0, i*(i+1) + 2*j, :].clone()
            A[:, i, j, 1] = sparams_data[0, i*(i+1) + 2*j + 1, :].clone()
    
    out_mat = A.clone()

    torch.diagonal(A, 0, 1, 2).zero_()

    out_mat += A.transpose(1, 2)

    return out_mat

def to_mag(data):
    """
    Converts a given signal in re/im to magnitude in Db.

    Args:
        data: A time/frequency series signal with real and imaginary channels.
              Can have shape [1, 2*num_sparams, length] where the second channel
                holds the real and imaginary parts of each s-param.
              Can also have shape [num_sparams, 2, length].
              Torch Tensor.
    
    Returns:
        magnitude_data: magnitude of each unique s-parameter in decibels.
    """
    if data.shape[0] == 1:
        n_sparams = data.shape[1] // 2
        data = data.squeeze().view(n_sparams, 2, -1)
    
    n_sparams = data.shape[0]
    length = data.shape[-1]
    
    mag_data = torch.zeros(n_sparams, length)
    
    for i in range(n_sparams):
        mag_data[i] = torch.square(data[i, 0, :]) + torch.square(data[i, 1, :])
    
    return 10 * torch.log10(mag_data)

def sparams_to_sing_vals(sparams_data):
    """
    Function for taking 3-D frequency series sparam data and
        calculating the singular values for every frequency.

    Args:
        sparams_data: 3D frequnecy series.
                      Torch tensor [1, 2 * unique_sparams, Freq].
    
    Returns:
        svd: The SVDs of each (num_ports x num_ports) matrix in the frequency series.
             Torch tensor [num_freqs, num_ports]
    """
    _, num_unique, num_freqs = sparams_data.shape
    num_unique = num_unique // 2

    num_ports = (-1 + np.sqrt(8*num_unique + 1)) // 2
    num_ports = int(num_ports)

    A = torch.zeros(num_freqs, num_ports, num_ports, 2, device=sparams_data.device, dtype=sparams_data.dtype)

    for i in range(num_ports):
        for j in range(i+1):
            #grab the real and imaginary parts
            #if they're diagonal, divide by two to account for double addition when we sum transpose
            if i != j:
                A[:, i, j, 0] = sparams_data[0, i*(i+1) + 2*j, :]
                A[:, i, j, 1] = sparams_data[0, i*(i+1) + 2*j + 1, :]
            else:
                A[:, i, j, 0] = sparams_data[0, i*(i+1) + 2*j, :] / 2
                A[:, i, j, 1] = sparams_data[0, i*(i+1) + 2*j + 1, :] / 2
    
    A = A + A.transpose(1, 2)

    #torch.complex.... makes a [num_freqs, num_ports, num_ports] tensor
    #must have compute_UV = True to use gradients
    return torch.linalg.svd(torch.complex(A[..., 0], A[..., 1]))[1]
    

class Measurement_MSE_Loss(nn.Module):
    """
    Given a signal x, observed measurements y, and observed indices kept_inds, 
        return the mse over the measurements of x vs true measurements y.
    
    Args:
        kept_inds: Array with the indices of the kept measurements.
        per_param: Whether to reduce the MSE for each S-param individually
                    before reducing the loss.
                   This can help with robustly fitting each S-param well.
        reduction: ["mean", "sum"]  
    """

    def __init__(self, kept_inds, per_param=False, reduction="mean"):
        super().__init__()

        self.kept_inds = kept_inds
        self.per_param = per_param
        self.reduction = reduction

        if not self.per_param:
            self.mse_loss = nn.MSELoss(reduction=self.reduction)
    
    def forward(self, x, y):
        if not self.per_param:
            return self.mse_loss(x[:, :, self.kept_inds], y)
        
        else:
            square_error = torch.square(x[:, :, self.kept_inds] - y) #[1, 2 * N_sparams, m]

            if self.reduction == "mean":
                mse_per_chan = torch.mean(square_error, dim=2) #[1, 2 * N_sparams]
                rmse_per_chan = torch.sqrt(mse_per_chan) #[1, 2 * N_sparams]
                
                return torch.mean(rmse_per_chan) 
            
            elif self.reduction == "sum":
                sse_per_chan = torch.sum(square_error, dim=2) #[1, 2 * N_sparams]
                rsse_per_chan = torch.sqrt(sse_per_chan) #[1, 2 * N_sparams]
                
                return torch.sum(rsse_per_chan) 

class Smoothing_Loss(nn.Module):
    """
    Loss function that penalizes second-order differences in a time series.

    Args:
        per_param: Whether to reduce the loss for each S-param individually
                    before reducing the loss.
                   This can help with robustly fitting each S-param well.
        reduction: ["mean", "sum"]
    """
    def __init__(self, per_param=False, reduction="mean"):
        super().__init__()

        self.per_param = per_param
        self.reduction = reduction
    
    def forward(self, x):
        second_order_diffs = torch.diff(torch.diff(x, dim=2), dim=2) #[1, 2 * N_sparams, N_freqs - 2]
        squared_second_diffs = torch.square(second_order_diffs) #[1, 2 * N_sparams, N_freqs - 2]

        if not self.per_param:
            if self.reduction == "mean":
                return torch.mean(squared_second_diffs)
            elif self.reduction == "sum":
                return torch.sum(squared_second_diffs)
        
        else:
            if self.reduction == "mean":
                loss_per_chan = torch.mean(squared_second_diffs, dim=2) #[1, 2 * N_sparams]
                rloss_per_chan = torch.sqrt(loss_per_chan) #[1, 2 * N_sparams]
                
                return torch.mean(rloss_per_chan) 
            
            elif self.reduction == "sum":
                loss_per_chan = torch.sum(squared_second_diffs, dim=2) #[1, 2 * N_sparams]
                rloss_per_chan = torch.sqrt(loss_per_chan) #[1, 2 * N_sparams]
                
                return torch.sum(rloss_per_chan) 
                
class Passive_Loss(nn.Module):
    """
    Loss function that penalizes singular values larger than 1
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        sing_vals = sparams_to_sing_vals(x) #[num_freqs, num_ports]

        bad_vals = torch.nn.functional.relu(sing_vals - 1)

        return torch.sum(torch.square(bad_vals))
