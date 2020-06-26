'''Homework 4, Computational Photonics, SS 2020:  Fourier modal method.
'''

import numpy as np
from numpy.linalg import eig, solve
from scipy.linalg import toeplitz
from scipy.fftpack import fft
from scipy.sparse import diags


def fmm1d_te_layer_modes(perm, period, k0, kx, N):
    '''Calculates the TE eigenmodes of a one-dimensional grating layer.

    Arguments
    ---------
        perm: 1d-array
            permittivity distribution
        period: float
            grating period
        k0: float
            vacuum wavenumber
        kx: float
            transverse wave vector
        N: int
            number of positive Fourier orders

    Returns
    -------
        beta: 1d-array
            propagation constants of the eigenmodes
        phie: 2d-array
            Fourier coefficients of the eigenmodes (each column
            corresponds to one mode)
    '''
    Nx = perm.size
    # Fourier coefficients of the permittivity
    perm_fc = fft(perm) / (Nx - 1)

    # take the first 2 * N positive and 0 frequency
    perm_fc_pos = perm_fc[0:2 * N + 1]
    # take the first 2 * N negative and 0 frequency
    perm_fc_neg = np.concatenate((np.array(perm_fc[0:1]),
                                  perm_fc[-(2 * N):][::-1]), axis=0)

    # calculate grating
    Gm = np.arange(-N, N + 1, 1) * 2 * np.pi / period

    # create the toeplitz matrix containing the Fourier coefficients of perm
    eps_hat = toeplitz(perm_fc_pos, perm_fc_neg)
    # create \hat K Matrix
    K_hat = diags(Gm + kx, offsets=0).todense()

    # create final matrix
    M_hat = (k0 ** 2 * eps_hat - np.matmul(K_hat, K_hat))

    # calculate the eigenvalues and eigenvectors of M_hat
    eig_values, eig_vectors = eig(M_hat)

    # take sqrt to get the propagation constant
    beta = np.sqrt(eig_values)
    # invert eigenvalue if it corresponds to a backward propagating direction
    beta[np.real(beta) + np.imag(beta) < 0] *= -1

    return beta, eig_vectors



def fmm1d_te(lam, theta, period, perm_in, perm_out,
             layer_perm, layer_ticknesses, N):
    '''Calculates the TE diffraction efficiencies for a one-dimensional
    layered grating structure using the T-matrix method.

    Arguments
    ---------
        lam: float
            vacuum wavelength
        theta: float
            angle of incidence in rad
        period: float
            grating period
        perm_in: float
            permittivity on the incidence side
        perm_out: float
            permittivity on the exit side
        layer_perm: 2d-array
            permittivity distribution within the grating
            layers (matrix, each row corresponds to one layer)
        layer_thicknesses: 1d-array
            thicknesses of the grating layers
        N: int
            number of positive Fourier orders

    Returns
    -------
        eta_r: 1d-array
            diffraction efficiencies of the reflected diffraction orders
        eta_t: 1d-array
            diffraction efficiencies of the transmitted diffraction orders
        r: 1d-array
            amplitude reflection coefficients of the reflected
            diffraction orders
        t: 1d-array
            amplitude transmission coefficients of the transmitted
            diffraction orders
    '''
    pass
