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
    K_hat_square = diags((Gm + kx) ** 2, offsets=0).todense()

    # create final matrix
    M_hat = (k0 ** 2 * eps_hat - K_hat_square)

    # calculate the eigenvalues and eigenvectors of M_hat
    eig_values, eig_vectors = eig(M_hat)

    # take sqrt to get the propagation constant
    beta = np.sqrt(eig_values)
    # invert eigenvalue if it corresponds to a backward propagating direction
    beta[np.real(beta) + np.imag(beta) < 0] *= -1

    return beta, eig_vectors



def fmm1d_te(lam, theta, period, perm_in, perm_out,
             layer_perm, layer_thicknesses, N):
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
    k0 = 2 * np.pi / lam + 0j
    d0 = 0
    # x component of k
    kx = 0j + (2 * np.pi / lam) * np.sqrt(perm_in) * np.sin(theta)
    G = (2 * np.pi / period) * np.arange(-N, N+1)
    # create K_hat matrix
    K_hat_square = np.diag((kx + G) ** 2)
    # initial phi electrical
    phi_e = np.identity(2 * N + 1)

    # initial beta
    beta_0 = np.sqrt(k0**2 * perm_in * np.identity(2 * N + 1) - K_hat_square)
   
    # TODO
    # what is this...
    beta_in = 1
    

    print("matmul", np.dot(phi_e, beta_0).shape)
    print("phi_e", phi_e.shape) 
    # initial transfer matrix
    T_matrix = np.identity(2 * (2*N + 1))
    print("T_matrix", T_matrix.shape)
    # initial block matrix
    B = np.block([[phi_e, phi_e],
                  [np.dot(phi_e, beta_0), np.dot(-phi_e, beta_0)]])
    print("B_matrix", B.shape)
    # iterate over all z layers
    for lt, perm in zip(layer_thicknesses,  layer_perm):
        beta, phi_e = fmm1d_te_layer_modes(perm, period, k0, kx, N)
        # convert beta to beta_hat containing the entries on the diagonal
        beta_hat = np.diag(beta)

        # matrices for forward and backward propagation
        p_pos = np.diag(np.exp(1j * beta * lt))
        p_neg = np.diag(np.exp(-1j * beta * lt))
        A = np.block([[phi_e, phi_e],
                      [np.dot(phi_e, beta_hat),
                       np.dot(-phi_e, beta_hat)]])

        print("A_matrix", A.shape)
        t_mat = np.linalg.solve(A, B)
        B = A
        T_mat = t_mat @ np.block([[p_pos, np.zeros((2 * N + 1, 2 * N + 1))],
                                  [np.zeros((2 * N + 1, 2 * N + 1)), p_neg]])
        T_matrix = T_mat @ T_matrix

    beta_out = np.sqrt(k0 ** 2 * perm_out * phi_e - K_hat_square)
    t_mat = np.linalg.solve(np.block([[phi_e, phi_e],
                                      [np.dot(phi_e, beta_out),
                                       np.dot(- phi_e, beta_out)]]),
                            B)


    T_mat = t_mat @ np.block([[np.identity(2 * N + 1),
                               np.zeros((2 * N + 1, 2 * N + 1))],
                              [np.zeros((2 * N + 1, 2 * N + 1)),
                               np.identity(2 * N + 1)]])

    T_matrix = T_mat @ T_matrix

    # initial amplitudes
    a_in = np.zeros(2 * N + 1)
    # set only this input coefficient to 1
    a_in[N + 1] = 1

    # extract the four block matrices from the T_matrix
    index_1 = slice(None, 2 * N + 1)
    index_2 = slice(2 * N + 1, None)
    t11 = T_matrix[index_1, index_1]
    t12 = T_matrix[index_1, index_2]
    t21 = T_matrix[index_2, index_1]
    t22 = T_matrix[index_2, index_2]

    # calculate R and T matrices
    R = np.dot(-np.linalg.solve(t22, t21),
                    a_in[:, np.newaxis])

    T = np.dot((t11 - t12 @ np.linalg.solve(t22, t21)),
                    a_in[:, np.newaxis])
    # print(np.linalg.solve(t22, t21).shape)
    # print(a_in[:, np.newaxis].shape)
    
    # extract efficiencies
    eta_r = 1 / np.real(beta_in) * np.real(beta_0) *\
            R * np.conj(np.transpose(R))

    eta_t = 1 / np.real(beta_in) * np.real(beta_out) *\
            T * np.conj(np.transpose(T))

    return eta_r, eta_t, R, T
