"""
This the main estimator module
"""
import numpy as np
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

import scipy.interpolate
import scipy.stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.neighbors import KDTree


def freedman_diaconis(data):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin number.

    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR = scipy.stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    datmin, datmax = data.min(), data.max()
    datrng = datmax - datmin
    return int((datrng / bw) + 1)


def _combinations(n_features, degree, interaction_only, include_bias):
    comb = combinations if interaction_only else combinations_w_r
    start = int(not include_bias)
    return chain.from_iterable(comb(range(n_features), i) for i in range(start, degree + 1))


def _get_bspline_basis(knots, degree=3, periodic=False):
    """Get spline coefficients for each basis spline."""
    nknots = len(knots)
    y_dummy = np.zeros(nknots)

    knots, coeffs, degree = scipy.interpolate.splrep(knots, y_dummy, k=degree, per=periodic)
    ncoeffs = len(coeffs)
    bsplines = []
    for ispline in range(nknots):
        coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
        bsplines.append((knots, coeffs, degree))
    return bsplines


class BSplineFeatures(TransformerMixin):
    def __init__(self, n_knots=5, degree=3, periodic=False):
        self.periodic = periodic
        self.degree = degree
        self.n_knots = n_knots  # knots are position along the axis of the knots

    def fit(self, X, y=None):
        # TODO determine position of knots given the datas
        knots = np.linspace(np.min(X), np.max(X), self.n_knots)
        self.bsplines_ = _get_bspline_basis(knots, self.degree, periodic=self.periodic)
        self.nsplines_ = len(self.bsplines_)
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def _get_fitted(self):
        """
        Get fitted parameters
        """
        return {"bsplines": self.bsplines_}

    def _set_fitted(self, fitted_dict):
        """
        Set fitted parameters
        """
        self.bsplines_ = fitted_dict["bsplines"]
        self.nsplines_ = len(self.bsplines_)


class BinsFeatures(KBinsDiscretizer):
    def __init__(self, n_bins_arg, strategy):
        """
        Init class
        """
        super().__init__(encode="onehot-dense", strategy=strategy)
        self.n_bins_arg = n_bins_arg

    def fit(self, X, y=None):
        """
        Determine bin number
        """
        if self.n_bins_arg == "auto":  # Automatique determination of the number of bins via maximum of sturges and freedman diaconis rules
            # Sturges rules
            self.n_bins = 1 + np.log2(X.shape[0])
            for d in range(self.dim_x):
                # Freedman–Diaconis rule
                n_bins = max(self.n_bins, freedman_diaconis(X[:, d]))
            self.n_bins = int(n_bins)
        elif isinstance(self.n_bins_arg, int):
            self.n_bins = self.n_bins_arg
        else:
            raise ValueError("The number of bins must be an integer")
        return super().fit(X, y)

    def _get_fitted(self):
        """
        Get fitted parameters
        """
        return {"n_bins": self.n_bins_, "bin_edges": self.bin_edges_}

    def _set_fitted(self, fitted_dict):
        """
        Set fitted parameters
        """
        self.n_bins_ = fitted_dict["n_bins"]
        self.bin_edges_ = fitted_dict["bin_edges_"]


class LinearElement(object):
    """1D element with linear basis functions.

    Attributes:
        index (int): Index of the element.
        x_l (float): x-coordinate of the left boundary of the element.
        x_r (float): x-coordinate of the right boundary of the element.
    """

    def __init__(self, index, x_left, x_center, x_right):
        self.num_nodes = 2
        self.index = index
        self.x_left = x_left
        self.x_center = x_center
        self.x_right = x_right
        self.center = np.asarray([0.5 * (self.x_right + self.x_left)])
        self.size = 0.5 * (self.x_right - self.x_left)

    def basis_function(self, x):
        x = np.asarray(x)
        return ((x >= self.x_left) & (x < self.x_center)) * (x - self.x_left) / (self.x_center - self.x_left) + ((x >= self.x_center) & (x < self.x_right)) * (self.x_right - x) / (self.x_right - self.x_center)


class FEM1DFeatures(TransformerMixin):
    def __init__(self, mesh, periodic=False):
        self.periodic = periodic
        # Add two point for start and end point
        extra_point_start = 2 * mesh.x[0] - mesh.x[1]
        extra_point_end = 2 * mesh.x[-1] - mesh.x[-2]
        x_dat = np.concatenate((np.array([extra_point_start]), mesh.x, np.array([extra_point_end])))
        # Create list of instances of Element
        self.elements = [LinearElement(i, x_dat[i], x_dat[i + 1], x_dat[i + 2]) for i in range(len(x_dat) - 2)]
        self.num_elements = len(self.elements)

    def fit(self, X, y=None):
        self.tree = KDTree(X)
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, self.num_elements))
        for k, element in enumerate(self.elements):
            istart = k  # * nfeatures
            iend = k + 1  # * nfeatures
            features[:, istart:iend] = element.basis_function(X)
        return features


# Mainly useful for pickling
def linear_fct(x):
    return x


class GLE_BasisTransform(TransformerMixin, BaseEstimator):
    """A transformer that give values of the basis along the trajectories.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    model : str, default= "aboba"
        The statistical model to use

    basis_type : str, default= "linear"
        Give the wanted basis projection
        Must be one of::
            "linear" : Linear basis.
            "polynomial" : Polynomial basis.
            "bins" : Bins basis.
            "bsplines" : BSplines basis.
            "custom": custom basis, you should pass a Transformer class

    degree : int, default=1
        The number of basis element to consider

    Attributes
    ----------
    basis_type_  : str
        The type of the basis.
    """

    def __init__(self, basis_type="linear", transformer=None, **kwargs):

        self.basis_type = basis_type
        self.featuresTransformer = transformer
        self.kwargs = kwargs

    def _initialize(self):
        """
        Initialize th class
        """
        self.basis_type = self.basis_type.casefold()

        self.to_combine_ = False
        if self.featuresTransformer is None:
            if self.basis_type == "linear":
                self.featuresTransformer = FunctionTransformer(linear_fct, validate=False)

            elif self.basis_type == "polynomial":
                degree = self.kwargs.get("degree", 3)
                if degree < 0:
                    raise ValueError("The number of basis element must be positive")
                self.featuresTransformer = PolynomialFeatures(degree=degree)

            elif self.basis_type == "bins":
                strategy = self.kwargs.get("strategy", "uniform")
                n_bins_arg = self.kwargs.get("n_bins", "auto")  # n_bins should be a number, we cannot have different number of bins for different direction

                self.featuresTransformer = BinsFeatures(n_bins_arg=n_bins_arg, strategy=strategy)  # TODO use sparse array
                self.to_combine_ = True and (not self.dim_x == 1)  # No need for combinaison if only one dimensionnal datas

            elif self.basis_type == "bsplines":
                n_knots = self.kwargs.get("n_knots", 5)
                degree = self.kwargs.get("degree", 3)
                periodic = self.kwargs.get("periodic", False)
                self.featuresTransformer = BSplineFeatures(n_knots, degree=degree, periodic=periodic)
                self.to_combine_ = True and (not self.dim_x == 1)

            elif self.basis_type == "custom":
                raise ValueError("No transformer have been passed as argument for custom transformer")
            else:
                raise ValueError("The basis type {} is not implemented.".format(self.basis_type))
        else:
            self.featuresTransformer = self.featuresTransformer

        if self.to_combine_:  # If it is needed to combine the features
            self.combinations = _combinations(self.nb_basis_elt_per_dim, self.dim_x, False, False)
            self.ncomb_ = sum(1 for _ in combinations)
            self.nb_basis_elt_ = self.nb_basis_elt_per_dim ** self.dim_x

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, dim_x)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        basis_params :  dict
            Additional basis parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = check_array(X)
        self.dim_x = X.shape[1]

        self._initialize()

        self.featuresTransformer = self.featuresTransformer.fit(X)

        if self.basis_type == "linear":
            self.nb_basis_elt_ = self.dim_x
        elif self.basis_type == "polynomial":
            self.nb_basis_elt_ = self.featuresTransformer.n_output_features_

        elif self.basis_type == "bins":
            self.nb_basis_elt_ = self.featuresTransformer.n_bins_
        elif self.basis_type == "bsplines":
            self.nb_basis_elt_ = self.featuresTransformer.nsplines_
        else:
            self.nb_basis_elt_ = self.dim_x

        self.fitted_ = True
        return self

    def combine(self, X):
        """
        If basis are one dimensionnal, combine them into product
        """
        n_samples, _ = X.shape
        # What follows is a faster implementation of:
        # for i, comb in enumerate(combinations):
        #     bk[:, i] = X[:, comb].prod(1)
        # This implementation uses two optimisations.
        # First one is broadcasting,
        # multiply ([X1, ..., Xn], X1) -> [X1 X1, ..., Xn X1]
        # multiply ([X2, ..., Xn], X2) -> [X2 X2, ..., Xn X2]
        # ...
        # multiply ([X[:, start:end], X[:, start]) -> ...
        # Second optimisation happens for degrees >= 3.
        # Xi^3 is computed reusing previous computation:
        # Xi^3 = Xi^2 * Xi.
        combX = np.empty((n_samples, self.ncomb_), dtype=X.dtype)
        current_col = 0

        # d = 1
        combX[:, current_col : current_col + self.dim_x] = X
        index = list(range(current_col, current_col + self.dim_x))
        current_col += self.dim_x
        index.append(current_col)

        # d >= 2
        for _ in range(1, self.dim_x):
            new_index = []
            end = index[-1]
            for feature_idx in range(self.dim_x):
                start = index[feature_idx]
                new_index.append(current_col)
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                # bk[:, start:end] are terms of degree d - 1
                # that exclude feature #feature_idx.
                np.multiply(combX[:, start:end], X[:, feature_idx : feature_idx + 1], out=combX[:, current_col:next_col], casting="no")
                current_col = next_col

            new_index.append(current_col)
            index = new_index
        return combX[:, :]

    def transform(self, X):
        """A reference implementation of a transform function.
        Take the position as input and output basis expansion

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
        X = check_array(X)
        n_samples, n_features = X.shape

        if n_features != self.dim_x:
            raise ValueError("X shape does not match training shape")
        bk = self.featuresTransformer.transform(X)

        if self.to_combine_:
            bk = self.combine(bk)
        return bk

    def get_coefficients(self):
        """
        Save fitted coefficients
        """
        params = self.get_params()
        params.update({"dim_x": self.dim_x, "nb_basis_elt": self.nb_basis_elt_})
        params.update(self.kwargs)
        if hasattr(self.featuresTransformer, "_get_fitted"):
            params.update(self.featuresTransformer._get_fitted())
        return params

    def set_coefficients(self, params):
        """
        Set fitted coefficients
        """
        self.set_params(**{k: params[k] for k in ("basis_type", "transformer", "kwargs") if k in params})
        self.dim_x = params["dim_x"]
        self._initialize()
        if hasattr(self.featuresTransformer, "_set_fitted"):
            self.featuresTransformer._set_fitted(params)
        self.nb_basis_elt_ = params["nb_basis_elt"]
        self.fitted_ = True
