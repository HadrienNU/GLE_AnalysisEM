"""
Somes utilities function
"""
import numpy as np
import scipy.linalg
import xarray as xr
from sklearn.utils import check_array


def generateRandomDefPosMat(dim_tot=2, rng=np.random.default_rng()):
    """
    Generate a random value of the A matrix
    """
    A = rng.standard_normal(size=(dim_tot, dim_tot))
    if not np.all(np.linalg.eigvals(A + A.T) > 0):
        A += np.abs(0.75 * np.min(np.linalg.eigvals(A + A.T))) * np.identity(dim_tot)
    return A


def loadTestDatas_est(paths, global_param):
    """
    Just one trajectory for test
    """
    dim_x = global_param["dim_x"]
    dim_h = global_param["dim_h"]

    traj_list_x = None
    traj_list_v = None
    traj_list_h = None
    for chemin in paths:
        trj = np.loadtxt(chemin)
        time = np.asarray(trj[:, 0])
        dt = time[1] - time[0]
        global_param["dt"] = dt

        x = np.asarray(trj[:, 1 : 1 + dim_x])
        v = np.asarray(trj[:, 1 + dim_x : 1 + 2 * dim_x])
        h = np.asarray(trj[:, 1 + 2 * dim_x : 1 + 2 * dim_x + dim_h])
        if traj_list_x is None:
            traj_list_x = np.array([np.concatenate((x, v), axis=1)])
        else:
            traj_list_x = np.concatenate((traj_list_x, np.array([np.concatenate((x, v), axis=1)])))
        if traj_list_v is None:
            traj_list_v = np.array([v])
        else:
            traj_list_v = np.concatenate((traj_list_v, np.array([v])))
        if traj_list_h is None:
            traj_list_h = np.array([h])
        else:
            traj_list_h = np.concatenate((traj_list_h, np.array([h])))

    return time, traj_list_x, traj_list_v, traj_list_h


def preprocessingTraj(X, dt, dim_x, force):
    """
    From a flat array compute everythong that is needed for the follwoing computation
    """
    X = check_array(X, ensure_min_features=4, allow_nd=True)
    traj_list = []
    for xv in X:
        x, v = np.hsplit(xv, 2)
        tps = dt * np.arange(x.shape[0])
        # v = (np.roll(x, -1, axis=0) - x) / dt
        # print(v)
        # xv_np = np.hstack((x, v))
        xhalf = xr.DataArray(x + 0.5 * dt * v, coords={"t": tps}, dims=["t", "space"])
        bk = xr.apply_ufunc(lambda x, fb: fb(x), xhalf, kwargs={"fb": force}, input_core_dims=[["space"]], output_core_dims=[["space"]], vectorize=True)

        projmat = np.zeros((dim_x, 2 * dim_x))
        projmat[:dim_x, :dim_x] = 0.5 * dt / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)
        projmat[:dim_x, dim_x : 2 * dim_x] = 1.0 / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)

        P = projmat.copy()
        P[:dim_x, dim_x : 2 * dim_x] = (1 + ((0.5 * dt) ** 2 / (1 + (0.5 * dt) ** 2))) * np.identity(dim_x)
        xv_plus_proj = (np.matmul(projmat, np.roll(xv, -1, axis=0).T)).T
        xv_proj = np.matmul(P, xv.T).T

        xv = xr.Dataset({"xv_plus_proj": (["t", "dim_x"], xv_plus_proj), "xv_proj": (["t", "dim_x"], xv_proj), "v": (["t", "dim_x"], v), "bk": (["t", "dim_x"], bk)}, coords={"t": tps})
        xv.attrs["lenTraj"] = x.shape[0]
        traj_list.append(xv)
    return traj_list


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


def memory_kernel(ntimes, dt, coeffs, dim_x):
    """
    Return the value of the estimated memory kernel
    Parameters
    ----------
    ntimes,dt: Number of timestep and timestep
    coeffs : Coefficients for diffusion and friction
    dim_x: Dimension of visible variables
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
    Kernel[0, :, :] += 2 * Avv
    return dt * np.arange(ntimes), Kernel


def xcorr(x):
    """FFT based autocorrelation function, which is faster than numpy.correlate"""
    from numpy.fft import fft, ifft

    # x is supposed to be an array of sequences, of shape (totalelements, length)
    fftx = fft(x, axis=0)
    ret = ifft(fftx * np.conjugate(fftx), axis=0)
    return ret
