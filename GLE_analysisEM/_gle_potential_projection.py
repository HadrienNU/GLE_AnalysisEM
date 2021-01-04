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
    """ A transformer that use potential estimation to compute value of the force along the trajectories.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    model : str, default= "aboba"
        The statistical model to use

    estimator : str, default= "histogram"
        Give the wanted basis projection
        Must be one of::
            "histogram" : The potential is computed through an histogram.
            "KDE" : Kernel density estimation of the potential.

    bins : str, or int, default="auto"
         The number of bins. It is passed to the numpy.histogram routine,
            see its documentation for details.

    bandwidth: float, default=1.0
        The bandwidth for the KDE

    kernel: str, default="gaussian"
        The kernel of the KDE
    """

    def __init__(self, model="aboba", estimator="histogram", bins="auto", kernel="gaussian", bandwidth=1.0):

        self.model = model
        self.estimator = estimator

        self.bins = bins

        self.kernel = kernel
        self.bandwidth = bandwidth

    def _check_parameters(self):
        """Check the inputed parameter for the basis type
        """
        if self.estimator not in ["histogram", "KDE"]:
            raise ValueError("The estimator {} is not implemented.".format(self.basis_type))

        if self.degree < 0:
            raise ValueError("The number of basis element must be positive")
        self.model = self.model.casefold()

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
        dt = X[1, 0] - X[0, 0]
        self.dim_x = (X.shape[1] - 1) // 2
        self.n_output_features_ = self.dim_x
        if self.model in ["aboba"]:
            x_pos = X[:, 1 : 1 + self.dim_x] + 0.5 * dt * X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        elif self.model in ["euler", "euler_noiseless"]:
            x_pos = X[:, 1 : 1 + self.dim_x]
        if self.estimator == "histogram":
            fehist = np.histogramdd(x_pos, bins=self.bins)  # TODO reshape x_pos
            xfa = (fehist[1][1:] + fehist[1][:-1]) / 2.0
            pf = fehist[0]
            xf = xfa[np.nonzero(pf)]
            fe = -np.log(pf[np.nonzero(pf)])
            self.fe_spline = interpolate.splrep(xf, fe, s=0, per=self.per)
        elif self.estimator == "KDE":
            self.kde_ = KernelDensity(kernel=self.kernel, bandwidth=self.bandwith).fit(x_pos)
        self.fitted_ = True
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Compute the force from derivative of fitted potential

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)

        """
        # Check is fit had been called
        check_is_fitted(self, "n_output_features_")

        # Input validation
        X = check_array(X, ensure_min_samples=3)
        dt = X[1, 0] - X[0, 0]
        if self.model in ["aboba"]:
            x_pos = X[:, 1 : 1 + self.dim_x] + 0.5 * dt * X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        elif self.model in ["euler", "euler_noiseless"]:
            x_pos = X[:, 1 : 1 + self.dim_x]
        n_samples, n_features = x_pos.shape

        if n_features != self.dim_x:
            raise ValueError("X shape does not match training shape")

        if self.estimator == "histogram":
            bk = interpolate.splev(x_pos, self.fe_spline, der=1)
        elif self.estimator == "KDE":
            raise NotImplementedError
        return np.hstack((X, bk))

    def predict(self, X):
        """Predict the values of the force basis for the data samples in X using trained model.

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
        check_is_fitted(self, "n_output_features_")
        n_samples, n_features = X.shape
        if self.basis_type == "linear":
            bk = X
        elif self.basis_type == "polynomial":
            bk = np.empty((self.n_output_features_,), dtype=X.dtype)

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

            # Constant term
            bk[:, 0] = 1
            current_col = 1

            # d = 1
            bk[:, current_col : current_col + n_features] = X
            index = list(range(current_col, current_col + n_features))
            current_col += n_features
            index.append(current_col)

            # d >= 2
            for _ in range(1, self.degree):
                new_index = []
                end = index[-1]
                for feature_idx in range(n_features):
                    start = index[feature_idx]
                    new_index.append(current_col)
                    if self.interaction_only:
                        start += index[feature_idx + 1] - index[feature_idx]
                    next_col = current_col + end - start
                    if next_col <= current_col:
                        break
                    # bk[:, start:end] are terms of degree d - 1
                    # that exclude feature #feature_idx.
                    np.multiply(bk[:, start:end], X[:, feature_idx : feature_idx + 1], out=bk[:, current_col:next_col], casting="no")
                    current_col = next_col

                new_index.append(current_col)
                index = new_index
        return bk
