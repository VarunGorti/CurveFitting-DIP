import torch
import torch.nn as nn
import random
import numpy as np

def get_single_series(data, sample_num, sparam_num, num_chan=2):
    """
    Grabs the Real or (Real, Im) series for a single sample and a single s-parameter.

    Args:
        data: Array filled with individual chip tensors.
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
    x = data[sample_num][:, sparam_num, :] #(LEN, 2)
    
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

class Measurement_MSE_Loss(nn.Module):
    """
    Given a signal x, observed measurements y, and observed indices kept_inds, 
        return the mse over the measurements of x vs true measurements y 
    """

    def __init__(self, kept_inds, y):
        super().__init__()

        self.kept_inds = kept_inds
        self.y = y

        self.mse_loss = nn.MSELoss()
    
    def forward(self, x):
        return self.mse_loss(x[:, :, self.kept_inds], self.y)

