import pytest
import numpy as np

# from sklearn.utils import assert_array_equal
# from sklearn.utils import assert_allclose

from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, sufficient_stats, sufficient_stats_hidden, preprocessingTraj
from ..utils import loadTestDatas_est, generateRandomDefPosMat


@pytest.fixture
def data():
    return ["GLE_analysisEM/tests/0_trajectories.dat"]


def test_gen_random_mat():
    A = generateRandomDefPosMat()
    assert np.all(np.linalg.eigvals(A + A.T) > 0)


# def test_basis_proj(data):
#     transform = GLE_BasisTransform(dim_x=1)
#
#
# def test_user_input(data):
#     est = GLE_Estimator(init_params="user")
#
#
# def test_markov_input(data):
#     est = GLE_Estimator(init_params="markov")
#


def test_em_estimator(data):
    est = GLE_Estimator(verbose=1, C_init=np.identity(2))
    X, idx, Xh = loadTestDatas_est(data, 1, 1)
    basis = GLE_BasisTransform()
    X = basis.fit_transform(X)
    est.fit(X)
    assert est.dt == 5e-3
    assert hasattr(est, "converged_")

    # X = data[0]
    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
