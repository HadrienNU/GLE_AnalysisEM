"""
This the main estimator module
"""
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array


class GLE_BasisTransform(TransformerMixin, BaseEstimator):
    """ A simple transformer that give values of the linear basis along the trajectories.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    basis_type : dict default={"name": "linear"}
        Give the wanted basis projection
        Must be one of::
            {"name": "linear"} : coefficients are initialized using markovian approximation.
            {"name": "polynomial", "poly_deg": poly_deg} : coefficients are initialized at values provided by the user
            'random' : coefficients are initialized randomly.

    Attributes
    ----------
    basis_type_  : str
        The type of the basis.
    """

    def __init__(self, dim_x=1, basis_type={"name": "linear"}, model="ABOBA"):
        self.dim_x = dim_x
        self.basis_type = basis_type
        self.model = model

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, dim_x)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.fitted_ = True
        return self

    def _check_basis_type(self):
        """Check the inputed parameter for the basis type
        """

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
        check_is_fitted(self, "fitted_")

        # Input validation
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        dt = X[1, 0] - X[0, 0]
        xhalf = X[:, 1 : 1 + self.dim_x] + 0.5 * dt * X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        if self.basis_type["name"] == "linear":
            bk = xhalf
        elif self.basis_type["name"] == "polynomial":
            bk = xhalf
        return np.hstack((X, bk))


class GLE_LinearBasis(TransformerMixin, BaseEstimator):
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
        X : {array-like, sparse matrix}, shape (n_samples, dim_x)
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
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        dt = X[1, 0] - X[0, 0]
        xhalf = X[:, 1 : 1 + self.dim_x] + 0.5 * dt * X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        return xhalf


class GLE_PolynomialBasis(TransformerMixin, BaseEstimator):
    """ A transformer that give values of for a polynomial basis along the trajectories.
    .. todo:: To implement the Polynomial Features, include a choice of Hermite basis?

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
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        dt = X[1, 0] - X[0, 0]
        xhalf = X[:, 1 : 1 + self.dim_x] + 0.5 * dt * X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        return xhalf


class GLE_BSplinesBasis(TransformerMixin, BaseEstimator):
    """ A transformer that give values of for a B-splines basis along the trajectories.
    .. todo:: To implement the Bsplines Features

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
        self.basis_type_ = "BSplines"
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
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        dt = X[1, 0] - X[0, 0]
        xhalf = X[:, 1 : 1 + self.dim_x] + 0.5 * dt * X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        return xhalf
