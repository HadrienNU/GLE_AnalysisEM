import pytest
import numpy as np

# from sklearn.utils import assert_array_equal
# from sklearn.utils import assert_allclose

from GLE_analysisEM import GLE_Estimator, sufficient_stats, sufficient_stats_hidden
from ..utils import loadTestDatas_est, preprocessingTraj


@pytest.fixture
def data():
    return ["GLE_analysisEM/tests/0_trajectories.dat"]


def test_m_step(data):
    est = GLE_Estimator(verbose=1)
    time, traj_list_x, traj_list_v, traj_list_h = loadTestDatas_est(data, {"dim_x": est.dim_x, "dim_h": est.dim_h})
    traj_list = preprocessingTraj(traj_list_x, est.dt, est.dim_x, est.force)
    datas = 0.0
    for n, traj in enumerate(traj_list):
        datas_visible = sufficient_stats(traj, est.dim_x, est.dim_coeffs_force) / len(traj_list)
        datas += sufficient_stats_hidden(traj_list_h[n], Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)

    logL1 = est.loglikelihood(datas)
    est._m_step(datas)
    logL2 = est.loglikelihood(datas)
    assert logL2 > logL1
    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_e_step(data):

    est = GLE_Estimator(verbose=1, C_init=np.identity(2), A_init=np.array([[5, 1.0], [-2.0, 0.07]]))
    time, traj_list_x, traj_list_v, traj_list_h = loadTestDatas_est(data, {"Ntrajs": 5, "dim_x": est.dim_x, "dim_h": est.dim_h})
    traj_list = preprocessingTraj(traj_list_x, est.dt, est.dim_x, est.force)
    datas = 0.0
    for n, traj in enumerate(traj_list):
        datas_visible = sufficient_stats(traj, est.dim_x, est.dim_coeffs_force) / len(traj_list)
        datas += sufficient_stats_hidden(traj_list_h[n], Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)

    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    new_stat = sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force)
    # assert close new_stat, datas
    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_em_estimator(data):
    est = GLE_Estimator(verbose=1, OptimizeDiffusion=False, C_init=np.identity(2))
    assert est.dt == 5e-3

    est.fit(data)
    assert hasattr(est, "is_fitted_")

    # X = data[0]
    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
