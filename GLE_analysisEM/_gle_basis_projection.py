"""
This the main estimator module
"""
import numpy as np
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from ._gle_potential_projection import GLE_PotentialTransform
from ._basis_features import LinearFeatures, BinsFeatures, BSplineFeatures


def _combinations(n_features, degree, interaction_only, include_bias):
    comb = combinations if interaction_only else combinations_w_r
    start = int(not include_bias)
    return chain.from_iterable(comb(range(n_features), i) for i in range(start, degree + 1))


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
        self.transformer = transformer
        self.kwargs = kwargs

    def _initialize(self):
        """
        Initialize the feature class
        """
        self.basis_type = self.basis_type.casefold()

        self.to_combine_ = False
        if self.transformer is None:
            if self.basis_type == "linear":
                self.featuresTransformer = LinearFeatures(to_center=self.kwargs.get("to_center", False))

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

            elif self.basis_type == "free_energy_kde":
                self.featuresTransformer = GLE_PotentialTransform(estimator="kde", dim_x=self.dim_x, bandwidth=self.kwargs.get("bandwidth", 1e-3), per=self.kwargs.get("periodic", False))

            elif self.basis_type == "free_energy_histogram" or self.basis_type == "free_energy":  # Default free energy set to histogram
                self.featuresTransformer = GLE_PotentialTransform(estimator="histogram", dim_x=self.dim_x, bins=self.kwargs.get("bins", "auto"), per=self.kwargs.get("periodic", False))

            elif self.basis_type == "custom":
                raise ValueError("No transformer have been passed as argument for custom transformer")
            else:
                raise ValueError("The basis type {} is not implemented.".format(self.basis_type))
        else:
            self.featuresTransformer = self.transformer

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
        if hasattr(self.featuresTransformer, "n_output_features_"):
            self.nb_basis_elt_ = self.featuresTransformer.n_output_features_
        else:
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

    def orthogonal_projection(self, X, y=None):
        """
        Get coefficients of the affine projection on the basis
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        y : {array-like, sparse-matrix}, shape (n_samples, n_features)
            Values at X of the function to project.
        """
        # Check is fit had been called
        check_is_fitted(self, "fitted_")

        # Input validation
        X = check_array(X)
        bk = self.transform(X)
        if y is None:
            y = X
        # For future we can try to implement minibatch via SGDRegressor and partial_fit
        self.regr_ = LinearRegression(fit_intercept=False).fit(bk, y=y)  # regressor is saved for latter used if needed
        return self.regr_.coef_.reshape(self.dim_x, -1)  # If this is a 1D array, that will becomes a 2D array

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
