'''Test script for Homework 4, Computational Photonics, SS 2020:  Fourier modal method.
'''

import numpy as np
from fmm import fmm1d_te
from matplotlib import pyplot as plt
import time as time 

plt.rcParams.update({
       'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.145,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.9,
        'figure.subplot.top': 0.9,
        'figure.subplot.wspace': 0.35,
        'figure.subplot.hspace': 0.3,
        'axes.grid': False,
        'image.cmap': 'viridis',
})

plt.close('all')


def print_results(N, theta, eta_t, eta_r, thicknesses):
    # filter for non-zero entries
    eta_r = eta_r[eta_r > 0]
    eta_t = eta_t[eta_t > 0]
    # printing
    print("Number of positive Fourier orders:\t{}".format(N))
    print("Layer thicknesses:\t\t\t{}".format(thicknesses))
    print("Angle:\t\t\t\t\t{:.4f}".format(theta*180/np.pi))
    print("Sum of eta_t and eta_r:\t\t\t{:.4f}".format(np.sum(eta_t) +
                                                     np.sum(eta_r)))
    print("Number of reflection orders:\t\t{}".format(len(eta_r)))
    print("Number of transmission orders:\t\t{}".format(len(eta_t)))
    print("eta_r:\t\t\t\t\t{}".format(eta_r))
    print("eta_t:\t\t\t\t\t{}".format(eta_t))
    print("\n")
    return 0

# %% parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lam         = 1.064  # vacuum wavelength [µm]
period      = 3.0    # period of the grating [µm]
widths      = np.array([1, 2, 3])/4.0*period  # widths of the high index
                                              # regions [µm]
thicknesses = np.array([1, 1, 1])*0.25  # thicknesses of the
                                        # grating layers [µm]
perm_l      = 1.0   # permittivity of the low index regions of the grating
perm_h      = 4.0   # permittivity of the high index regions of the grating
perm_in     = 1.0   # permittivity on the incidence side
perm_out    = 4.0   # permittivity on the exit side
Nx          = 1001  # number of points to discretize the permittivity
                    # ditribution
N           = 20    # number of positive Fourier orders

theta = 50 / 180 * np.pi

x = np.arange(Nx) * period / Nx
layer_perm = perm_l * np.ones((len(widths), Nx))

for i in range(len(widths)):
    layer_perm[i, x <= widths[i]] = perm_h


for N in np.arange(10, 40, 5):
    eta_r, eta_t, r, t = fmm1d_te(lam, theta, period, perm_in, perm_out,
                                  layer_perm, thicknesses, N)
    _ = print_results(N, theta, eta_t, eta_r, thicknesses)

N = 20
thicknesses = np.array([1, 1, 1]) * 0.5  # thicknesses of the
eta_r, eta_t, r, t = fmm1d_te(lam, theta, period, perm_in, perm_out,
                              layer_perm, thicknesses, N)
_ = print_results(N, theta, eta_t, eta_r, thicknesses)


for dtype in [np.complex64, np.complex128]:
    times = []
    for _ in range(30):
        a = time.time()
        eta_r, eta_t, r, t = fmm1d_te(lam, theta, period, perm_in, perm_out,
                                  layer_perm, thicknesses, N, dtype=dtype)
        b = time.time()
        times.append(b - a)

    print("Time elapsed for {}:\t({:.4f} ± {:.4f})s".format(dtype, np.mean(times),
                                                        np.std(times)))
