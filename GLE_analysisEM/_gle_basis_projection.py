"""
This the main estimator module
"""
import numpy as np
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array


class GLE_BasisTransform(TransformerMixin, BaseEstimator):
    """ A simple transformer that give values of the linear basis along the trajectories.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions.

    model : str, default= "aboba"
        The statistical model to use

    basis_type : str, default={"name": "linear"}
        Give the wanted basis projection
        Must be one of::
            {"name": "linear"} : Linear basis.
            {"name": "polynomial"} : Polynomial basis.
            {"name": "hermite"} : Hermite basis.
            {"name": "BSplines", "":} : Hermite basis.

    degree : int, default=1
        The number of basis element to consider

    interaction_only : boolean, default = False
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

    Attributes
    ----------
    basis_type_  : str
        The type of the basis.
    """

    def __init__(self, model="aboba", basis_type="linear", degree=1, interaction_only=False):

        self.model = model

        self.basis_type = basis_type
        self.degree = degree
        self.interaction_only = interaction_only

    @staticmethod
    def _combinations(n_features, degree, interaction_only, include_bias):
        comb = combinations if interaction_only else combinations_w_r
        start = int(not include_bias)
        return chain.from_iterable(comb(range(n_features), i) for i in range(start, degree + 1))

    def _check_basis_type(self, **basis_params):
        """Check the inputed parameter for the basis type
        """
        if self.basis_type not in ["linear", "polynomial", "hermite", "BSplines"]:
            raise ValueError("The basis type {} is not one of implemented one.".format(self.basis_type))

        if self.degree < 0:
            raise ValueError("The number of basis element must be positive")
        self.include_zeroth_term_ = True
        if self.basis_type == "linear":
            self.degree = 1
            self.interaction_only = False
            self.include_zeroth_term_ = False
        self.model = self.model.casefold()

    def fit(self, X, y=None, **basis_params):
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
        self._check_basis_type(**basis_params)  # Check that input parameters are coherent

        self.dim_x = (X.shape[1] - 1) // 2
        combinations = self._combinations(self.dim_x, self.degree, self.interaction_only, self.include_zeroth_term_)
        self.n_output_features_ = sum(1 for _ in combinations)
        self.fitted_ = True
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

        if self.basis_type == "linear":
            bk = x_pos
        elif self.basis_type == "polynomial":
            bk = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)

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
            bk[:, current_col : current_col + n_features] = x_pos
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
                    np.multiply(bk[:, start:end], x_pos[:, feature_idx : feature_idx + 1], out=bk[:, current_col:next_col], casting="no")
                    current_col = next_col

                new_index.append(current_col)
                index = new_index

        elif self.basis_type == "hermite":
            bk = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
            bk = x_pos
        elif self.basis_type == "BSplines":
            bk = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
            bk = x_pos
        return np.hstack((X, bk))

    def predict(self, X):
        """Predict the hidden variables for the data samples in X using trained model.

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
