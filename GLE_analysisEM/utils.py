"""
Somes utilities function
"""
import numpy as np
import scipy.linalg
from sklearn.model_selection import ShuffleSplit


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


def loadDatas_est(paths, dim_x, maxlenght=None):
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
        if maxlenght is None:
            txv = np.asarray(trj[:, : 1 + 2 * dim_x])
        else:
            txv = np.asarray(trj[:maxlenght, : 1 + 2 * dim_x])
        if X is None:
            X = txv
        else:
            idx_trajs.append(len(X))
            X = np.vstack((X, txv))

    return X, idx_trajs


def loadDatas_pos(paths, dim_x, maxlenght=None):
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
        if maxlenght is None:
            tps = np.asarray(trj[:, :1] - trj[0, 0])  # Set origin of time to zero
            pos = np.asarray(trj[:, 1 : 1 + dim_x])
        else:
            tps = np.asarray(trj[:maxlenght, :1] - trj[0, 0])  # Set origin of time to zero
            pos = np.asarray(trj[:maxlenght, 1 : 1 + dim_x])
        velocity = np.gradient(pos, tps[:, 0], axis=0)
        txv = np.hstack((tps, pos, velocity))
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
    sub_trajs = np.array_split(trj, n_cut)

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


def bootstrap_Datas(paths, dim_x, n_splits=5, test_size=None, train_size=0.9, random_state=np.random.default_rng()):
    """
    Give a generator that give only a subset of the paths with replacement for bootstrapping
    See sklearn.utils.resample for documentation
    """
    nppaths = np.asarray(paths)
    number_paths = np.floor(train_size * len(nppaths))
    for n in range(n_splits):
        paths_n = random_state.choice(nppaths, size=number_paths, replace=True)
        yield loadDatas_est(paths_n, dim_x)


def generateRandomDefPosMat(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """Generate a random value of the A matrix"""
    # A = rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / dim_x + dim_h  # Eigenvalues of A mainly into the unit circle
    # mat = max_ev * scipy.linalg.expm(0.25 * scipy.linalg.logm(A)) + min_re_ev * np.identity(dim_x + dim_h)  # map the unit circle into the quarter disk
    # print(min_re_ev, max_ev)
    return sub_matrix_gen(dim_x=dim_x, dim_h=dim_h, rng=rng, max_ev=max_ev, min_re_ev=min_re_ev)


def sub_matrix_gen(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """
    Build specific matrix
    """
    A = np.zeros((dim_x + dim_h, dim_x + dim_h))
    Ahh = random_clever_gen(dim_x=0, dim_h=dim_h, rng=rng, max_ev=max_ev, min_re_ev=min_re_ev)
    A[dim_x:, dim_x:] = Ahh
    A[:dim_x, :dim_x] = random_clever_gen(dim_x=dim_x, dim_h=0, rng=rng, max_ev=max_ev * 1e-2, min_re_ev=min_re_ev * 1e-2)
    min_dim = min(dim_x, dim_h)
    A[dim_x : dim_x + min_dim, :min_dim] = -np.eye(min_dim)
    A[:min_dim, dim_x : dim_x + min_dim] = np.eye(min_dim)
    return A


def random_clever_gen(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """
    Generate random matrix with positive real part eigenvalues
    """
    notFound = True
    n = 0
    # We still had to avoid too small eigenvalues
    while notFound:
        A = max_ev * rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / (dim_x + dim_h)
        # try:  # In case we are unlucky and have non diagonalisable matrix
        lamb, v = np.linalg.eig(A)
        lamb_p = 0.5 * np.abs(np.real(lamb)) + 1.0j * np.imag(lamb)
        Ap = np.real(np.matmul(v, np.matmul(np.diag(lamb_p), np.linalg.inv(v))))
        n += 1
        if np.all(np.real(np.linalg.eigvals(Ap)) > min_re_ev) and np.all(np.linalg.eigvals(Ap + Ap.T) > min_re_ev):
            notFound = False
        # except:
        #     continue
    # print(n)
    return Ap


def random_gen_bruteforce(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """
    Brute force generation of correct matrix
    """
    notFound = True
    n = 0
    while notFound:
        A = max_ev * rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / (dim_x + dim_h)
        A[:dim_x, dim_x:] = 1
        n += 1
        if np.all(np.real(np.linalg.eigvals(A)) > min_re_ev) and np.all(np.linalg.eigvals(A + A.T) > min_re_ev):
            notFound = False
    # print("Brute force", n, np.linalg.eigvals(A))
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
    for n in np.arange(ntimes):
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
        X = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2])).reshape(-1, 1)
    elif x_lims.shape[0] == 2:  # 2D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2]))
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], int(x_lims[1][2]))
        x, y = np.meshgrid(x_coords, y_coords)
        X = np.vstack((x.flatten(), y.flatten())).T
    elif x_lims.shape[0] == 3:  # 3D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2]))
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], int(x_lims[1][2]))
        z_coords = np.linspace(x_lims[2][0], x_lims[2][1], int(x_lims[2][2]))
        x, y, z = np.meshgrid(x_coords, y_coords, z_coords)
        X = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    elif x_lims.shape[0] == 4:  # 4D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2]))
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], int(x_lims[1][2]))
        z_coords = np.linspace(x_lims[2][0], x_lims[2][1], int(x_lims[2][2]))
        c_coords = np.linspace(x_lims[3][0], x_lims[3][1], int(x_lims[3][2]))
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
    """From the force field reconstruct the potential"""
