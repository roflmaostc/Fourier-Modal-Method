'''Test script for Homework 4, Computational Photonics, SS 2020:  Fourier modal method.
'''

import numpy as np
from fmm import fmm1d_te_layer_modes
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fftpack import ifft

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
lam       = 1.064  # vacuum wavelength [µm]
period    = 3.0    # period of the grating [µm]
width     = 1.5    # width of the high index region [µm]
perm_l    = 1.0    # permittivity of the low index regions of the grating
perm_h    = 4.0    # permittivity of the high index regions of the grating
kx        = 0.0    # Bloch vector
Nx        = 1001   # number of points to discretize the permittivity
                   # ditribution
N         = 25     # number of Fourier orders

k0 = 2*np.pi/lam   # vacuum wavenumber

# create permittivity array
perm = np.ones((Nx))
perm[0:Nx // 2] = perm_h
perm[Nx // 2:] = perm_l

k0 = 2 * np.pi / lam
kx = 0


beta, eig_vectors = fmm1d_te_layer_modes(perm, period, k0, kx, N)
print(beta, eig_vectors)
