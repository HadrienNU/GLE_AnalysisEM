"""
Somes utilities function
"""
import numpy as np
import scipy.linalg
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample


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
    """Loads some test trajectories

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

def loadDatas_pos(paths, dim_x):
    """Loads some test trajectories

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
        tps = np.asarray(trj[:, : 1])
        pos=np.asarray(trj[:, 1: 1 + dim_x])
        velocity = np.gradient(pos,tps[:,0],axis=0)
        txv=np.hstack((tps,pos,velocity))
        if X is None:
            X = txv
        else:
            idx_trajs.append(len(X))
            X = np.vstack((X, txv))

    return X, idx_trajs

def cutTrajs(trj, n_cut):
    """
    Cut trajectory into smaller piece
    """
    sub_trajs=np.array_split(trj,n_cut)

    X = None
    idx_trajs = []
    for txv in sub_trajs:
        if X is None:
            X = txv
        else:
            idx_trajs.append(len(X))
            X = np.vstack((X, txv))

    return X, idx_trajs

def split_loadDatas(paths, dim_x, n_splits=5, test_size=None, train_size=0.9, random_state=None):
    """
    Give a generator that give only a subset of the paths for cross validation
    See sklearn.model_selection.ShuffleSplit for documentation
    """
    nppaths = np.asarray(paths)
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
    for train_index, test_index in ss.split(paths):
        yield loadDatas_est(nppaths[train_index], dim_x)


def bootstrap_Datas(paths, dim_x, n_splits=5, test_size=None, train_size=0.9, random_state=None):
    """
    Give a generator that give only a subset of the paths with replacement for bootstrapping
    See sklearn.utils.resample for documentation
    """
    nppaths = np.asarray(paths)
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
    for train_index, test_index in ss.split(paths):
        yield loadDatas_est(nppaths[train_index], dim_x)


def generateRandomDefPosMat(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """Generate a random value of the A matrix
    """
    # A = rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / dim_x + dim_h  # Eigenvalues of A mainly into the unit circle
    # mat = max_ev * scipy.linalg.expm(0.25 * scipy.linalg.logm(A)) + min_re_ev * np.identity(dim_x + dim_h)  # map the unit circle into the quarter disk
    return random_gen_bruteforce(dim_x=dim_x, dim_h=dim_h, rng=rng, max_ev=max_ev, min_re_ev=min_re_ev)


def random_gen_bruteforce(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """
    Brute force generation of correct matrix
    """
    notFound = True
    n = 0
    while notFound:
        A = max_ev * rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / dim_x + dim_h
        A[:dim_x,dim_x:]=1
        n += 1
        if np.all(np.real(np.linalg.eigvals(A)) > min_re_ev) and np.all(np.linalg.eigvals(A + A.T) > min_re_ev):
            notFound = False
    # print("Brute force", n, np.linalg.eigvals(A + A.T))
    return A


def oldrandom_gen(dim_x=1, dim_h=1, rng=np.random.default_rng()):
    """
    Old code for generating random matrix
    """
    A = 4 * rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h))
    # A[dim_x:, :dim_x] = 1
    if not np.all(np.linalg.eigvals(A + A.T) > 0):
        A += np.abs(0.505 * np.min(np.linalg.eigvals(A + A.T))) * np.identity(dim_x + dim_h)
    return A


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


def filtersmoother(Xtplus, mutilde, R, diffusion_coeffs, mu0, sig0):
    """
    Apply Kalman filter and Rauch smoother. Fallback for the fortran implementation
    """
    # Initialize, we are going to use a numpy array for storing intermediate values and put the resulting µh and \Sigma_h into the xarray only at the end
    lenTraj = Xtplus.shape[0]
    dim_x = Xtplus.shape[1]
    dim_h = mu0.shape[0]

    muf = np.zeros((lenTraj, dim_h))
    Sigf = np.zeros((lenTraj, dim_h, dim_h))
    mus = np.zeros((lenTraj, dim_h))
    Sigs = np.zeros((lenTraj, dim_h, dim_h))
    # To store the pair probability distibution
    muh = np.zeros((lenTraj, 2 * dim_h))
    Sigh = np.zeros((lenTraj, 2 * dim_h, 2 * dim_h))

    # Forward Proba
    muf[0, :] = mu0
    Sigf[0, :, :] = sig0
    # Iterate and compute possible value for h at the same point
    for i in range(1, lenTraj):
        # try:
        muf[i, :], Sigf[i, :, :], muh[i - 1, :], Sigh[i - 1, :, :] = filter_kalman(muf[i - 1, :], Sigf[i - 1, :, :], Xtplus[i - 1], mutilde[i - 1], R, diffusion_coeffs, dim_x, dim_h)
        # except np.linalg.LinAlgError:
        #     print(i, muf[i - 1, :], Sigf[i - 1, :, :], Xtplus[i - 1], mutilde[i - 1], self.friction_coeffs[:, self.dim_x :], self.diffusion_coeffs)
    # The last step comes only from the forward recursion
    Sigs[-1, :, :] = Sigf[-1, :, :]
    mus[-1, :] = muf[-1, :]
    # Backward proba
    for i in range(lenTraj - 2, -1, -1):  # From T-1 to 0
        # try:
        mus[i, :], Sigs[i, :, :], muh[i, :], Sigh[i, :, :] = smoothing_rauch(muf[i, :], Sigf[i, :, :], mus[i + 1, :], Sigs[i + 1, :, :], Xtplus[i], mutilde[i], R, diffusion_coeffs, dim_x, dim_h)
        # except np.linalg.LinAlgError as e:
        #     print(i, muf[i, :], Sigf[i, :, :], mus[i + 1, :], Sigs[i + 1, :, :], Xtplus[i], mutilde[i], self.friction_coeffs[:, self.dim_x :], self.diffusion_coeffs)
        #     print(repr(e))
        #     raise ValueError
    return muh, Sigh


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
        Kernel[0, :, :] = Kernel[0, :, :] + 2 * Avv
    return dt * np.arange(ntimes), Kernel


def memory_timescales(coeffs, dim_x):
    """
    Compute the eigenvalues of A_hh to get the timescale of the memory
    """
    return np.linalg.eigvals(coeffs["A"][dim_x:, dim_x:])


def fit_memory_fct_helper(dt, ntimes, noDirac):
    """
    The function to fit
    """
    dim_x = Avh.shape[0]
    Kernel = np.zeros((ntimes, dim_x, dim_x))
    for n in range(ntimes):
        Kernel[n, :, :] = -np.matmul(Avh, np.matmul(scipy.linalg.expm(-1 * n * dt * Ahh), Ahv))
    if not noDirac:
        Kernel[0, :, :] += 2 * Avv
    return Kernel


def fit_memory_function():
    """
    Provide a fit of the memory function to sum of exponential
    """


def correlation(a, b=None, subtract_mean=False):
    """
    Correlation between a and b
    """
    meana = int(subtract_mean) * np.mean(a)
    a2 = np.append(a - meana, np.zeros(2 ** int(np.ceil((np.log(len(a)) / np.log(2)))) - len(a)))
    data_a = np.append(a2, np.zeros(len(a2)))
    fra = np.fft.fft(data_a)
    if b is None:
        sf = np.conj(fra) * fra
    else:
        meanb = int(subtract_mean) * np.mean(b)
        b2 = np.append(b - meanb, np.zeros(2 ** int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)))
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra) * frb
    res = np.fft.ifft(sf)
    cor = np.real(res[: len(a)]) / np.array(range(len(a), 0, -1))
    return cor[: len(cor) // 2]


def forcefield(x_lims, basis, force_coeffs):
    """Compute and save the force field that have been fitted

    Parameters
    ----------

    x_lims: array, shape (dim_x,3)
        Bounds of the plot

    basis: GLE_BasisTransform instance
        The instance of the basis projection

    force_coeffs: array-like
        Value of the force coefficients in the basis

    """
    x_lims = np.asarray(x_lims)
    if x_lims.ndim == 1:  # In case of flat array
        x_lims.reshape(1, -1)
    if x_lims.shape[0] == 1:  # 1D:
        X = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2]).reshape(-1, 1)
    elif x_lims.shape[0] == 2:  # 2D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2])
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], x_lims[1][2])
        x, y = np.meshgrid(x_coords, y_coords)
        X = np.vstack((x.flatten(), y.flatten())).T
    elif x_lims.shape[0] == 3:  # 3D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2])
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], x_lims[1][2])
        z_coords = np.linspace(x_lims[2][0], x_lims[2][1], x_lims[2][2])
        x, y, z = np.meshgrid(x_coords, y_coords, z_coords)
        X = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    elif x_lims.shape[0] == 4:  # 4D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2])
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], x_lims[1][2])
        z_coords = np.linspace(x_lims[2][0], x_lims[2][1], x_lims[2][2])
        c_coords = np.linspace(x_lims[3][0], x_lims[3][1], x_lims[3][2])
        x, y, z, c = np.meshgrid(x_coords, y_coords, z_coords, c_coords)
        X = np.vstack((x.flatten(), y.flatten(), z.flatten(), c.flatten())).T
    elif x_lims.shape[0] > 4:
        raise NotImplementedError("Dimension higher than 3 are not implemented")
    force_field = np.matmul(force_coeffs, basis.predict(X).T).T
    return np.hstack((X, force_field))


def forcefield_plot2D(x_lims, basis, force_coeffs):
    """Compute and save the force field that have been fitted

    Parameters
    ----------

    x_lims: array, shape (dim_x,3)
        Bounds of the plot

    basis: GLE_BasisTransform instance
        The instance of the basis projection

    force_coeffs: array-like
        Value of the force coefficients in the basis

    """
    x_coords = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2])
    y_coords = np.linspace(x_lims[1][0], x_lims[1][1], x_lims[1][2])
    x, y = np.meshgrid(x_coords, y_coords)
    X = np.vstack((x.flatten(), y.flatten())).T
    force_field = np.matmul(force_coeffs, basis.predict(X).T).T
    f = force_field.reshape(x_lims[0][2], x_lims[1][2], 2)
    return x, y, f[:, :, 0], f[:, :, 1]


def potential_reconstruction():
    """ From the force field reconstruct the potential
    """
