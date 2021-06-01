"""
This the main estimator module
"""
import numpy as np
from scipy import interpolate

from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array


class GLE_PotentialTransform(TransformerMixin, BaseEstimator):
    """A transformer that use potential estimation to compute value of the force along the trajectories.
    For histogram input of dimension 1 or 2 are fitted to B-splines and evaluated from there.
    Higher dimension simply use the value of the histogram closest to the point of evaluation.
    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    estimator : str, default= "histogram"
        Give the wanted basis projection
        Must be one of::
            "histogram" : The potential is computed through an histogram.
            "kde" : Kernel density estimation of the potential.

    bins : str, or int, default="auto"
         The number of bins. It is passed to the numpy.histogram routine,
            see its documentation for details.

    bandwidth: float, default=1.0
        The bandwidth for the KDE

    kernel: str, default="gaussian"
        The kernel of the KDE
    """

    def __init__(self, dim_x=1, estimator="histogram", bins="auto", kernel="gaussian", bandwidth=1.0, per=False):

        self.dim_x = dim_x
        self.estimator = estimator

        self.per = per

        self.bins = bins

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_output_features_ = self.dim_x

    def _check_parameters(self):
        """Check the inputed parameter for the basis type"""
        self.estimator = self.estimator.casefold()
        if self.estimator not in ["histogram", "kde"]:
            raise ValueError("The estimator {} is not implemented.".format(self.estimator))

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
        X = check_array(X, ensure_min_samples=2)
        self._check_parameters()  # Check that input parameters are coherent
        # self.dim_x = (X.shape[1] - 1) // 2
        # print("DIM x", self.dim_x)
        self.n_output_features_ = self.dim_x

        self.min_vals = np.amin(X, axis=0)
        self.max_vals = np.amax(X, axis=0)
        if self.estimator == "histogram":
            if self.dim_x == 1:
                fehist = np.histogram(X, bins=self.bins)
                xfa = (fehist[1][1:] + fehist[1][:-1]) / 2.0
                xf = xfa[np.nonzero(fehist[0])]
            else:
                fehist = np.histogramdd(X)
            self.edges_hist_ = fehist[1]
            pf = fehist[0]
            self.fe_ = np.where(pf > 0, -np.log(pf), np.zeros_like(pf))
        elif self.estimator == "kde":
            self.kde_ = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(X)
            self.fe_ = self.kde_.score_samples(X)

        # Spline interpolation
        if self.dim_x == 1:
            self.fe_spline_ = interpolate.splrep(xf, self.fe_, s=0, per=self.per)
        elif self.dim_x == 2:
            xfa = [(edge[1:] + edge[:-1]) / 2.0 for edge in self.edges_hist_]
            # x, y = np.meshgrid(xfa[0], xfa[1])
            # fe_flat = pf.flatten()
            # x_coords = x.flatten()[np.nonzero(fe_flat)]
            # y_coords = y.flatten()[np.nonzero(fe_flat)]
            self.fe_spline_ = interpolate.RectBivariateSpline(xfa[0], xfa[1], self.fe_)
            # self.fe_spline_ = interpolate.bisplrep(x_coords, y_coords, fe_flat[np.nonzero(fe_flat)])
        self.fitted_ = True
        return self

    def transform(self, X):
        """Compute the force from derivative of fitted potential at given points

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

        if self.estimator == "histogram":
            if self.dim_x == 1:
                bk = interpolate.splev(X, self.fe_spline_, der=1).reshape(-1, 1)
            elif self.dim_x == 2:
                bkx = self.fe_spline_.ev(X[:, 0], X[:, 1], dx=1).reshape(-1, 1)
                bky = self.fe_spline_.ev(X[:, 0], X[:, 1], dy=1).reshape(-1, 1)

                # bkx = interpolate.bisplev(X[:, 0], X[:, 1], self.fe_spline_, dx=1).reshape(-1, 1)
                # bky = interpolate.bisplev(X[:, 0], X[:, 1], self.fe_spline_, dy=1).reshape(-1, 1)
                bk = np.hstack((bkx, bky))
            else:
                raise NotImplementedError
        elif self.estimator == "kde":
            bk = self.differentiateKernel(X)
        return -bk

    def predict(self, X):
        """Predict the values of the potential for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # Check is fit had been called
        check_is_fitted(self, "fitted_")
        # Input validation
        X = check_array(X)
        n_samples, n_features = X.shape
        if n_features != self.dim_x:
            raise ValueError("X shape does not match training shape")

        if self.estimator == "histogram":
            if self.dim_x == 1:
                return interpolate.splev(X, self.fe_spline_).reshape(-1, 1)
            elif self.dim_x == 2:
                return self.fe_spline_.ev(X[:, 0], X[:, 1]).reshape(-1, 1)
                # return -interpolate.bisplev(X[:, 0], X[:, 1], self.fe_spline_)
            else:
                # interpolate.interpn(points, self.fe_, x_pos, method="linear")
                raise NotImplementedError
                self.digitize(X)
        elif self.estimator == "kde":
            return self.kde_.score_samples(X).reshape(-1, 1)

    def digitize(self, X):
        """Find location of a given points inside the bins of the histogram"""
        out_indx = np.empty((X.shape[0]))
        for i in range(self.dim_x):
            out_indx[i] = np.digitize(X[:, i], self.edges_hist_[i])
        return out_indx

    def differentiateKernel(self, X):
        """Numerical differentiation in N-D"""
        grad = np.empty_like(X)
        for n in range(self.dim_x):
            eps = np.zeros_like(X)
            eps[:, n] = 0.1 * self.bandwidth
            grad[:, n] = (self.kde_.score_samples(X + eps) - self.kde_.score_samples(X - eps)) / (self.bandwidth)
        return grad

    def _get_fitted(self):
        """
        Get fitted parameters
        """
        return {"estimator": self.estimator, "min_vals": self.min_vals, "max_vals": self.max_vals, "n_output_features": self.n_output_features_, "fe_spline": self.fe_spline_}

    def _set_fitted(self, fitted_dict):
        """
        Set fitted parameters
        """
        self.estimator = "histogram"  # As we use the spline interpolation
        self.min_vals = fitted_dict["min_vals"]
        self.max_vals = fitted_dict["max_vals"]
        self.n_output_features_ = fitted_dict["n_output_features"]
        if "fe_spline" in fitted_dict:
            self.fe_spline_ = fitted_dict["fe_spline"]
        self.fitted_ = True
