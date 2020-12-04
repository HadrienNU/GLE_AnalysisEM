"""
This the main estimator module
"""
import numpy as np
import scipy.linalg
import xarray as xr

import warnings
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, check_array


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


class GLE_LinearTransformer(TransformerMixin, BaseEstimator):
    """ A simple transformer that project the trajectory onto a linear basis.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from what was seen" "in `fit`")
        return np.sqrt(X)
