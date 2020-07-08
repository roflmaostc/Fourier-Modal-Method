'''Homework 4, Computational Photonics, SS 2020:  Fourier modal method.
'''

import numpy as np
from numpy.linalg import eig, solve
from scipy.linalg import toeplitz
from scipy.fftpack import fft
from scipy.sparse import diags


def fmm1d_te_layer_modes(perm, period, k_0, k_x, N, dtype=np.complex128):
    '''Calculates the TE eigenmodes of a one-dimensional grating layer.

    Arguments
    ---------
        perm: 1d-array
            permittivity distribution
        period: float
            grating period
        k_0: float
            vacuum wavenumber
        k_x: float
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
    # number of points in x direction
    N_x = perm.size
    perm = perm.astype(dtype)
    # Fourier coefficients of the permittivity
    perm_fc = (fft(perm) / (N_x - 1)).astype(dtype)

    # take the first 2 * N positive and 0 frequency
    perm_fc_pos = perm_fc[:2 * N + 1]
    # take the first 2 * N negative and 0 frequency
    perm_fc_neg = np.concatenate((np.array(perm_fc[:1]),
                                  perm_fc[-(2 * N):][::-1]), axis=0)

    # calculate grating
    Gm = np.arange(-N, N + 1, 1, dtype=dtype) * 2 * np.pi / period

    # create the Toeplitz matrix containing the Fourier coefficients of perm
    eps_hat = toeplitz(perm_fc_pos, perm_fc_neg).astype(dtype)
    # create \hat K Matrix
    K_hat_square = diags((Gm + k_x) ** 2, offsets=0).todense().astype(dtype)

    # create final matrix
    M_hat = (k_0 ** 2 * eps_hat - K_hat_square).astype(dtype)

    # calculate the eigenvalues and eigenvectors of M_hat
    eig_values, eig_vectors = eig(M_hat)

    # take sqrt to get the propagation constant
    beta = np.sqrt(eig_values).astype(dtype)
    # invert eigenvalue if it corresponds to a backward propagating direction
    beta[np.real(beta) + np.imag(beta) < 0] *= -1

    return beta, eig_vectors


def fmm1d_te(lam, theta, period, perm_in, perm_out,
             layer_perm, layer_thicknesses, N, dtype=np.complex128):
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
    # vacuum wave vector
    k_0 = 2 * np.pi / lam + 0j
    # x component of k
    k_x = k_0 * np.sqrt(perm_in) * np.sin(theta)
    # create grating
    G = (2 * np.pi / period) * np.arange(-N, N + 1, dtype=dtype)
    # create K_hat matrix
    K_hat_square = np.diag((k_x + G) ** 2).astype(dtype)
    # initial phi electrical
    ident = np.identity(2 * N + 1, dtype=dtype)

    # initial beta
    beta_0_hat = np.sqrt(k_0 ** 2 * perm_in * ident - K_hat_square, dtype=dtype)
    beta_0_hat[np.real(beta_0_hat) + np.imag(beta_0_hat) < 0.0] *= -1
    # initial block matrix
    T_matrix = np.block([[ident, ident],
                        [beta_0_hat, -beta_0_hat]])
    # iterate over all z layers
    for lt, perm in zip(layer_thicknesses, layer_perm):
        # get the betas and phi_e in this layer
        beta, phi_e = fmm1d_te_layer_modes(perm, period, k_0, k_x, N,
                                           dtype=dtype)
        # convert beta to beta_hat containing the entries on the diagonal
        beta_hat = np.diag(beta).astype(dtype)

        # matrices for forward and backward propagation
        p_pos = np.diag(np.exp(1j * beta * lt)).astype(dtype)
        p_neg = np.diag(np.exp(-1j * beta * lt)).astype(dtype)

        # create A matrix which is needed to get the new transfer matrix
        A = np.block([[phi_e, phi_e],
                      [np.dot(phi_e, beta_hat),
                       np.dot(-phi_e, beta_hat)]])
        # put the propagation matrices in a block matrix
        p_mat = np.block([[p_pos, np.zeros(p_pos.shape)],
                          [np.zeros(p_pos.shape), p_neg]])
        T = A @ solve(A.T, p_mat.T).T
        T_matrix = T @ T_matrix

    # beta_out_hat matrix
    beta_out_hat = np.sqrt(k_0 ** 2 * perm_out * ident
                           - K_hat_square, dtype=dtype)
    beta_out_hat[np.real(beta_out_hat) + np.imag(beta_out_hat) < 0.0] *= -1
    # last missing matrix which inverse is left multiplied
    T_final = np.block([[ident, ident],
                        [beta_out_hat, - beta_out_hat]]).astype(dtype)

    # create the final transfer matrix
    T_matrix = solve(T_final, T_matrix)

    # initial amplitudes
    a_in = np.zeros(2 * N + 1, dtype=dtype)
    # set only this input coefficient to 1
    a_in[N] = 1 + 0j

    # extract the four block matrices from the T_matrix
    index_1 = slice(None, 2 * N + 1)
    index_2 = slice(2 * N + 1, None)

    t11 = T_matrix[index_1, index_1]
    t12 = T_matrix[index_1, index_2]
    t21 = T_matrix[index_2, index_1]
    t22 = T_matrix[index_2, index_2]

    # calculate R and T matrices
    r = np.dot(-solve(t22, t21), a_in[:, np.newaxis])
    t = np.dot((t11 - t12 @ solve(t22, t21)), a_in[:, np.newaxis])

    # extract the diagonal elements
    beta_in = np.diag(beta_0_hat).astype(dtype)

    # calculate transmission and reflection efficiencies
    eta_r = np.real(1 / np.real(beta_in[N]) *
                             np.dot(np.real(beta_0_hat),
                                    np.multiply(r, np.conj(r))))
    eta_t = np.real(1 / np.real(beta_in[N]) *
                             np.dot(np.real(beta_out_hat),
                                    np.multiply(t, np.conj(t))))
    
    #return 1D arrays
    eta_r = eta_r.A1
    eta_t = eta_t.A1
    r = r.A1
    t = t.A1

    return eta_r, eta_t, r, t
