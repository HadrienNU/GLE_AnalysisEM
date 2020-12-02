import pytest
import numpy as np

# from sklearn.utils import assert_array_equal
# from sklearn.utils import assert_allclose

from analysisEM import GLE_Estimator


@pytest.fixture
def data():
    return np.loadtxt("analysisEM/tests/test_traj.dat").reshape(1, -1)


def test_em_estimator_n_iter(data):
    # check that n_iter is the number of iteration performed.
    est = GLE_Estimator()
    max_iter = 1
    est.set_params(max_iter=max_iter)
    est.fit(data)
    assert est.n_iter_ == max_iter


def test_em_estimator(data):
    est = GLE_Estimator()
    assert est.dt == 5e-3

    est.fit(data)
    # assert hasattr(est, "is_fitted_")

    X = data[0]
    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
