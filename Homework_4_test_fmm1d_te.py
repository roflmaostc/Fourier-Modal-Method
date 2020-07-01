'''Test script for Homework 4, Computational Photonics, SS 2020:  Fourier modal method.
'''

import numpy as np
from fmm import fmm1d_te
from matplotlib import pyplot as plt

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

theta = 0#np.pi / 6 


x = np.arange(Nx) * period / Nx
layer_perm = perm_l * np.ones((len(widths), Nx))

for i in range(len(widths)):
    layer_perm[i, x <= widths[i]] = perm_h


eta_r, eta_t, r, t = fmm1d_te(lam, theta, period, perm_in, perm_out,
                              layer_perm, thicknesses, N)

print(eta_t)
print(np.sum(np.abs(t) ** 2 + np.abs(r) ** 2))
print(r)
# print(np.sum(eta_r) + np.sum(eta_t))
