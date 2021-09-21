"""
This the main estimator module
"""
import numpy as np

import scipy.interpolate
import scipy.stats

from sklearn.base import TransformerMixin

from sklearn.preprocessing import KBinsDiscretizer
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
    IQR = scipy.stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    datmin, datmax = data.min(), data.max()
    datrng = datmax - datmin
    return int((datrng / bw) + 1)


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
        self.n_output_features_ = len(self.bsplines_)
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.n_output_features_))
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
        self.n_output_features_ = len(self.bsplines_)


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
        nsamples, dim_x = X.shape
        if self.n_bins_arg == "auto":  # Automatique determination of the number of bins via maximum of sturges and freedman diaconis rules
            # Sturges rules
            self.n_bins = 1 + np.log2(nsamples)
            for d in range(dim_x):
                # Freedman–Diaconis rule
                n_bins = max(self.n_bins, freedman_diaconis(X[:, d]))
            self.n_bins = int(n_bins)
        elif isinstance(self.n_bins_arg, int):
            self.n_bins = self.n_bins_arg
        else:
            raise ValueError("The number of bins must be an integer")
        super().fit(X, y)
        self.n_output_features_ = self.n_bins
        return self

    def _get_fitted(self):
        """
        Get fitted parameters
        """
        return {"n_bins": self.n_bins, "bin_edges": self.bin_edges_}

    def _set_fitted(self, fitted_dict):
        """
        Set fitted parameters
        """
        self.n_bins = fitted_dict["n_bins"]
        self.bin_edges_ = fitted_dict["bin_edges"]


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


class LinearFeatures(TransformerMixin):
    def __init__(self, to_center=False):
        """
        """
        self.centered = to_center

    def fit(self, X, y=None):
        self.n_output_features_ = X.shape[1]
        if self.centered:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros((self.n_output_features_,))
        return self

    def transform(self, X):
        return X - self.mean_

    def _get_fitted(self):
        """
        Get fitted parameters
        """
        return {"mean": self.mean_, "n_output_features": self.n_output_features_}

    def _set_fitted(self, fitted_dict):
        """
        Set fitted parameters
        """
        self.n_output_features_ = fitted_dict["dim_x"]
        if "mean" in fitted_dict:
            self.mean_ = fitted_dict["mean"]
        else:
            self.mean_ = np.zeros((self.n_output_features_,))
