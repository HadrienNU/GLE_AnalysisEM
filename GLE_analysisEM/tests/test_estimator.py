import pytest
import numpy as np

# from sklearn.utils import assert_array_equal
# from sklearn.utils import assert_allclose

from GLE_analysisEM import GLE_Estimator, sufficient_stats, sufficient_stats_hidden
from ..utils import loadTestDatas_est, preprocessingTraj, generateRandomDefPosMat


@pytest.fixture
def data():
    return ["GLE_analysisEM/tests/0_trajectories.dat"]


def test_gen_random_mat():
    A = generateRandomDefPosMat()
    assert np.all(np.linalg.eigvals(A + A.T) > 0)


def test_user_input(data):
    est = GLE_Estimator(init_params="user")


def test_markov_input(data):
    est = GLE_Estimator(init_params="markov")


def test_m_step(data):
    est = GLE_Estimator()
    est._check_initial_parameters()
    time, traj_list_x, traj_list_v, traj_list_h = loadTestDatas_est(data, {"dim_x": 1, "dim_h": 1})
    traj_list = preprocessingTraj(traj_list_x, 5e-3, 1, lambda x: -1 * x)
    datas = 0.0
    for n, traj in enumerate(traj_list):
        datas_visible = sufficient_stats(traj, est.dim_x, est.dim_coeffs_force) / len(traj_list)
        zero_sig = np.zeros((traj.attrs["lenTraj"], 2 * est.dim_h, 2 * est.dim_h))
        muh = np.hstack((np.roll(traj_list_h, -1, axis=0), traj_list_h))
        datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)
    est._initialize_parameters(datas_visible, np.random.default_rng())
    logL1 = est.loglikelihood(datas)
    est._m_step(datas)
    logL2 = est.loglikelihood(datas)
    assert logL2 > logL1


def test_e_step(data):

    est = GLE_Estimator(C_init=np.identity(2), A_init=np.array([[5, 1.0], [-2.0, 0.07]]), init_params="user")
    est._check_initial_parameters()
    time, traj_list_x, traj_list_v, traj_list_h = loadTestDatas_est(data, {"dim_x": 1, "dim_h": 1})
    traj_list = preprocessingTraj(traj_list_x, 5e-3, 1, lambda x: -1 * x)
    datas = 0.0
    for n, traj in enumerate(traj_list):
        datas_visible = sufficient_stats(traj, est.dim_x, est.dim_coeffs_force) / len(traj_list)
        zero_sig = np.zeros((traj.attrs["lenTraj"], 2 * est.dim_h, 2 * est.dim_h))
        muh = np.hstack((np.roll(traj_list_h, -1, axis=0), traj_list_h))
        datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)
    est._initialize_parameters(datas_visible, np.random.default_rng())

    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    new_stat = sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force)
    # assert close new_stat, datas
    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_em_estimator(data):
    est = GLE_Estimator(verbose=1, C_init=np.identity(2))
    assert est.dt == 5e-3
    time, traj_list_x, traj_list_v, traj_list_h = loadTestDatas_est(data, {"dim_x": 1, "dim_h": 1})
    est.fit(traj_list_x)
    assert hasattr(est, "converged_")

    # X = data[0]
    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
