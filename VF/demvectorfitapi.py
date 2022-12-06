import skrf as rf
import vectorfit
import numpy as np
import random
import matplotlib.pyplot as plt

###############################################
# Select the source Touchstone parameters
filename = "case01/case01.s4p"

###############################################
# Create a fitter instance to fill get vector fit
# models of this Touchstone file
fitter = vectorfit.VectorFitter(filename)

###############################################
# Define a few frequency samples

# Start by getting the full sweep range
full_sweep = fitter.ground_truth.f
fmin = min(full_sweep)
fmax = max(full_sweep)

# Uniform samples
f10 = np.linspace(fmin, fmax, 10)
f20 = np.linspace(fmin, fmax, 20)
f100 = np.linspace(fmin, fmax, 100)

# A random sample, just for fun
random.seed(5)
fr50 = [random.uniform(fmin, fmax) for _ in range(50)]

###############################################
# Create vector fits using the specified frequency points
vf10 = fitter.vector_fit("Uniform 10", f10)
vf20 = fitter.vector_fit("Uniform 20", f20)
vf100 = fitter.vector_fit("Uniform 100", f100)
vfr50 = fitter.vector_fit("Random 50", fr50)

###############################################
# Plot these fits
def plot_parameter(vector_fits, ground_truth, row, col):
    plt.figure()
    names = [vf.name for vf in vector_fits] + ["Ground Truth"]
    networks = [vf.fitted_network for vf in vector_fits] +  [ground_truth]
    
    for network in networks:
        frequencies = network.f
        s = [network.s[ff, row, col] for ff in range(len(frequencies))]
        plt.plot(frequencies, [abs(val) for val in s])
    plt.title(f"$|S_{{{row+1}{col+1}}}|$")
    plt.xlabel("Frequency")
    plt.ylabel("$|S|$")
    plt.legend(names)
plot_parameter([vf10, vf20, vf100, vfr50], fitter.ground_truth, 0, 0)
plot_parameter([vf10, vf20, vf100, vfr50], fitter.ground_truth, 0, 1)
plt.show()