"""
Somes utilities function
"""
import numpy as np
import scipy.linalg


def loadTestDatas_est(paths, dim_x, dim_h):
    """Loads some test trajectories with known hidden variables

    Parameters
    ----------
    paths : list of str
        List of paths to trajectory files, one trajectory per file
    dim_x : int
        Visible dimension
    dim_h : int
        Hidden dimension
    """

    X = None
    idx_trajs = []
    X_h = None

    for chemin in paths:
        trj = np.loadtxt(chemin)
        txv = np.asarray(trj[:, : 1 + 2 * dim_x])
        h = np.asarray(trj[:, 1 + 2 * dim_x : 1 + 2 * dim_x + dim_h])
        if X is None:
            X = txv
        else:
            idx_trajs.append(len(X))
            X = np.vstack((X, txv))

        if X_h is None:
            X_h = h
        else:
            X_h = np.vstack((X_h, h))

    return X, idx_trajs, X_h


def loadDatas_est(paths, dim_x):
    """Loads some test trajectories with known hidden variables

    Parameters
    ----------
    paths : list of str
        List of paths to trajectory files, one trajectory per file
    dim_x : int
        Visible dimension
    """

    X = None
    idx_trajs = []

    for chemin in paths:
        trj = np.loadtxt(chemin)
        txv = np.asarray(trj[:, : 1 + 2 * dim_x])
        if X is None:
            X = txv
        else:
            idx_trajs.append(len(X))
            X = np.vstack((X, txv))

    return X, idx_trajs


def filter_kalman(mutm, Sigtm, Xt, mutilde_tm, expAh, SST, dim_x, dim_h):
    """
    Compute the foward step using Kalman filter, predict and update step

    Parameters
    ----------
    mutm, Sigtm: Values of the foward distribution at t-1
    Xt, mutilde_tm: Values of the trajectories at T and t-1
    expAh, SST: Coefficients parameters["expA"][:, dim_x:] (dim_x+dim_h, dim_h) and SS^T (dim_x+dim_h, dim_x+dim_h)
    dim_x,dim_h: Dimension of visibles and hidden variables
    """
    # Predict step marginalization Normal Gaussian
    mutemp = mutilde_tm + np.matmul(expAh, mutm)
    Sigtemp = SST + np.matmul(expAh, np.matmul(Sigtm, expAh.T))
    # Update step conditionnal Normal Gaussian
    invSYY = np.linalg.inv(Sigtemp[:dim_x, :dim_x])
    marg_mu = mutemp[dim_x:] + np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, Xt - mutemp[:dim_x]))
    marg_sig = Sigtemp[dim_x:, dim_x:] - np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, Sigtemp[dim_x:, :dim_x].T))

    R = expAh[dim_x:, :] - np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, expAh[:dim_x, :]))
    # Pair probability distibution Z_t,Z_{t-1}
    mu_pair = np.hstack((marg_mu, mutm))
    Sig_pair = np.zeros((2 * dim_h, 2 * dim_h))
    Sig_pair[:dim_h, :dim_h] = marg_sig
    Sig_pair[dim_h:, :dim_h] = np.matmul(R, Sigtm)
    Sig_pair[:dim_h, dim_h:] = Sig_pair[dim_h:, :dim_h].T
    Sig_pair[dim_h:, dim_h:] = Sigtm

    return marg_mu, marg_sig, mu_pair, Sig_pair


def smoothing_rauch(muft, Sigft, muStp, SigStp, Xtplus, mutilde_t, expAh, SST, dim_x, dim_h):
    """
    Compute the backward step using Kalman smoother
    """

    invTemp = np.linalg.inv(SST + np.matmul(expAh, np.matmul(Sigft, expAh.T)))
    R = np.matmul(np.matmul(Sigft, expAh.T), invTemp)

    mu_dym_rev = muft + np.matmul(R[:, :dim_x], Xtplus) - np.matmul(R, np.matmul(expAh, muft) + mutilde_t)
    Sig_dym_rev = Sigft - np.matmul(np.matmul(R, expAh), Sigft)

    marg_mu = mu_dym_rev + np.matmul(R[:, dim_x:], muStp)
    marg_sig = np.matmul(R[:, dim_x:], np.matmul(SigStp, R[:, dim_x:].T)) + Sig_dym_rev

    # Pair probability distibution Z_{t+1},Z_{t}
    mu_pair = np.hstack((muStp, marg_mu))
    Sig_pair = np.zeros((2 * dim_h, 2 * dim_h))
    Sig_pair[:dim_h, :dim_h] = SigStp
    Sig_pair[dim_h:, :dim_h] = np.matmul(R[:, dim_x:], SigStp)
    Sig_pair[:dim_h, dim_h:] = Sig_pair[dim_h:, :dim_h].T
    Sig_pair[dim_h:, dim_h:] = marg_sig

    return marg_mu, marg_sig, mu_pair, Sig_pair


def memory_kernel(ntimes, dt, coeffs, dim_x, noDirac=False):
    """
    Return the value of the estimated memory kernel

    Parameters
    ----------
    ntimes,dt: Number of timestep and timestep
    coeffs : Coefficients for diffusion and friction
    dim_x: Dimension of visible variables
    noDirac: Remove the dirac at time zero

    Returns
    -------
    timespan : array-like, shape (n_samples, )
        Array of time to evaluate memory kernel

    kernel_evaluated : array-like, shape (n_samples, dim_x,dim_x)
        Array of values of the kernel at time provided
    """
    Avv = coeffs["A"][:dim_x, :dim_x]
    Ahv = coeffs["A"][dim_x:, :dim_x]
    Avh = coeffs["A"][:dim_x, dim_x:]
    Ahh = coeffs["A"][dim_x:, dim_x:]
    Kernel = np.zeros((ntimes, dim_x, dim_x))
    for n in range(ntimes):
        Kernel[n, :, :] = -np.matmul(Avh, np.matmul(scipy.linalg.expm(-1 * n * dt * Ahh), Ahv))
    if not noDirac:
        Kernel[0, :, :] += 2 * Avv
    return dt * np.arange(ntimes), Kernel


def memory_timescales(coeffs, dim_x):
    """
    Compute the eigenvalues of A_hh to get the timescale of the memory
    """
    return np.linalg.eigvals(coeffs["A"][dim_x:, dim_x:])


def generateRandomDefPosMat(dim_x=1, dim_h=1, rng=np.random.default_rng()):
    """Generate a random value of the A matrix
    """
    A = 4 * rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h))
    # A[dim_x:, :dim_x] = 1
    if not np.all(np.linalg.eigvals(A + A.T) > 0):
        A += np.abs(0.75 * np.min(np.linalg.eigvals(A + A.T))) * np.identity(dim_x + dim_h)
    return A


def xcorr(x):
    """FFT based autocorrelation function, which is faster than numpy.correlate"""
    from numpy.fft import fft, ifft

    # x is supposed to be an array of sequences, of shape (totalelements, length)
    fftx = fft(x, axis=0)
    ret = ifft(fftx * np.conjugate(fftx), axis=0)
    return ret
