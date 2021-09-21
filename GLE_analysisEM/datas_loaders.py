"""
Somes utilities function
"""
import numpy as np
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
    """Loads trajectories from a list of file

    Parameters
    ----------
    paths : list of str
        List of paths to trajectory files, one trajectory per file
        The file are loaded with ``numpy.loadtxt`` and should have one column by dimension and one data point per line.
    dim_x : int
        Number of column to take from each file
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


def cutTrajs(X, idx_trajs=[], n_cut=1):
    """
    Cut trajectory into smaller piece
    """

    X_cut = None
    idx_cut = []
    traj_list = np.split(X, idx_trajs)
    for trj in traj_list:
        sub_trajs = np.array_split(trj, n_cut)
        for txv in sub_trajs:
            if X is None:
                X_cut = txv
            else:
                idx_cut.append(len(X))
                X_cut = np.vstack((X_cut, txv))

    return X_cut, idx_cut


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
    number_paths = int(np.floor(train_size * len(nppaths)))
    for n in range(n_splits):
        paths_n = random_state.choice(nppaths, size=number_paths, replace=True)
        yield loadDatas_est(paths_n, dim_x)
