"""
This the main estimator module
"""
import numpy as np
import xarray as xr

import warnings
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array


class GLE_LinearTransformer(TransformerMixin, BaseEstimator):
    """ A simple transformer that give values of the linear basis along the trajectories.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    Attributes
    ----------
    basis_type_  : str
        The type of the basis.
    """

    def __init__(self, dim_x=1):
        self.dim_x = dim_x

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_time_step, dim_x)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.basis_type_ = "linear"
        X = check_array(X, ensure_min_features=4, allow_nd=True)
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Take the x_{1/2} as input and output basis expansion

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)

        """
        # Check is fit had been called
        check_is_fitted(self, "basis_type_")

        # Input validation
        X = check_array(X, ensure_min_features=4, allow_nd=True)
        traj_list = []
        for xhalf in X:
            # xhalf = xr.DataArray(x + 0.5 * self.dt_ * v, coords={"t": tps}, dims=["t", "space"])  # Déjà fourni
            bk = xr.apply_ufunc(lambda x: -1.0 * x, xhalf, input_core_dims=[["space"]], output_core_dims=[["space"]], vectorize=True)  # Linear features
            xv = xr.Dataset({"bk": (["t", "dim_x"], bk)}, coords={"t": tps})
            traj_list.append(xv)
        return traj_list


class GLE_PolynomialTransformer(TransformerMixin, BaseEstimator):
    """ A transformer that give values of for a polynomial basis along the trajectories.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    Attributes
    ----------
    basis_type_  : str
        The type of the basis.
    """

    def __init__(self, dim_x=1, poly_deg=1):
        self.dim_x = dim_x
        self.poly_deg = poly_deg

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_time_step, dim_x)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.basis_type_ = "polynomial"
        X = check_array(X, ensure_min_features=4, allow_nd=True)
        return self


class GLE_BSplinesTransformer(TransformerMixin, BaseEstimator):
    """ A transformer that give values of for a B-splines basis along the trajectories.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    Attributes
    ----------
    basis_type_  : str
        The type of the basis.
    """

    def __init__(self, dim_x=1, poly_deg=1):
        self.dim_x = dim_x
        self.poly_deg = poly_deg

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_time_step, dim_x)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.basis_type_ = "B-splines"
        X = check_array(X, ensure_min_features=4, allow_nd=True)
        return self
