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
        self.bsplines = _get_bspline_basis(knots, self.degree, periodic=self.periodic)
        self.nsplines = len(self.bsplines)
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.bsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features


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

    def __init__(self, model="euler", basis_type="linear", transformer=None, **kwargs):

        self.model = model
        self.basis_type = basis_type
        self.featuresTransformer = transformer
        self.kwargs = kwargs

    def project_data(self, X):
        """
        Project the datas according to the model to return position at evaluation point
        """
        dt = X[1, 0] - X[0, 0]
        if "aboba" in self.model:
            x_pos = X[:, 1 : 1 + self.dim_x] + 0.5 * dt * X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        elif "euler" in self.model:
            x_pos = X[:, 1 : 1 + self.dim_x]
        else:
            x_pos = X[:, 1 : 1 + self.dim_x]
        return x_pos

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

        self.model = self.model.casefold()
        self.basis_type = self.basis_type.casefold()

        self.dim_x = (X.shape[1] - 1) // 2

        X = self.project_data(X)

        self.to_combine_ = False
        if self.featuresTransformer is None:
            if self.basis_type == "linear":
                self.featuresTransformer = FunctionTransformer(lambda x: x, validate=False).fit(X)
                self.nb_basis_elt_ = self.dim_x

            elif self.basis_type == "polynomial":
                degree = self.kwargs.get("degree", 3)
                if degree < 0:
                    raise ValueError("The number of basis element must be positive")
                self.featuresTransformer = PolynomialFeatures(degree=degree).fit(X)
                self.nb_basis_elt_ = self.featuresTransformer.n_output_features_

            elif self.basis_type == "bins":
                strategy = self.kwargs.get("strategy", "uniform")
                n_bins_arg = self.kwargs.get("n_bins", "auto")  # n_bins should be a number, we cannot have different number of bins for different direction
                if n_bins_arg == "auto":  # Automatique determination of the number of bins via maximum of sturges and freedman diaconis rules
                    # Sturges rules
                    n_bins = 1 + np.log2(X.shape[0])
                    for d in range(self.dim_x):
                        # Freedman–Diaconis rule
                        n_bins = max(n_bins, freedman_diaconis(X[:, d]))
                    n_bins = int(n_bins)
                elif isinstance(n_bins_arg, int):
                    n_bins = n_bins_arg
                else:
                    raise ValueError("The number of bins must be an integer")

                self.featuresTransformer = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy=strategy).fit(X)  # TODO use sparse array
                self.to_combine_ = True and (not self.dim_x == 1)  # No need for combinaison if only one dimensionnal datas
                self.nb_basis_elt_ = n_bins

            elif self.basis_type == "bsplines":
                n_knots = self.kwargs.get("n_knots", 5)
                degree = self.kwargs.get("degree", 3)
                periodic = self.kwargs.get("periodic", False)
                self.featuresTransformer = BSplineFeatures(n_knots, degree=degree, periodic=periodic).fit(X)
                self.to_combine_ = True and (not self.dim_x == 1)
                self.nb_basis_elt_ = self.featuresTransformer.nsplines

            elif self.basis_type == "custom":
                raise ValueError("No transformer have been passed as argument for custom transformer")
            else:
                raise ValueError("The basis type {} is not implemented.".format(self.basis_type))
        else:
            self.featuresTransformer = self.featuresTransformer.fit(X)
            self.nb_basis_elt_ = self.dim_x

        if self.to_combine_:  # If it is needed to combine the features
            self.combinations = _combinations(self.nb_basis_elt_per_dim, self.dim_x, False, False)
            self.ncomb_ = sum(1 for _ in combinations)
            self.nb_basis_elt_ = self.nb_basis_elt_per_dim ** self.dim_x
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
        X = check_array(X, ensure_min_samples=3)
        x_pos = self.project_data(X)
        n_samples, n_features = x_pos.shape

        if n_features != self.dim_x:
            raise ValueError("X shape does not match training shape")
        bk = self.featuresTransformer.transform(x_pos)

        if self.to_combine_:
            bk = self.combine(bk)
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
        check_is_fitted(self, "fitted_")
        y = self.featuresTransformer.transform(X)
        if self.to_combine_:
            y = self.combine(y)
        return y
