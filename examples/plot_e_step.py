"""
===========================
E step, Kalman filter
===========================

Inner working of the E step :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, sufficient_stats, sufficient_stats_hidden
from GLE_analysisEM.utils import loadTestDatas_est, preprocessingTraj

est = GLE_Estimator(C_init=np.identity(2), A_init=np.array([[5, 1.0], [-2.0, 0.07]]), mu_init=np.zeros((1,)), sig_init=np.zeros((1, 1)), init_params="user")
est._check_initial_parameters()
time, X, _, traj_list_h = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat"], {"dim_x": 1, "dim_h": 1})
traj_list = preprocessingTraj(X, est.dt, est.dim_x, est.force)
datas = 0.0
for n, traj in enumerate(traj_list):
    datas_visible = sufficient_stats(traj, est.dim_x, est.dim_coeffs_force) / len(traj_list)
    zero_sig = np.zeros((traj.attrs["lenTraj"], 2 * est.dim_h, 2 * est.dim_h))
    muh = np.hstack((np.roll(traj_list_h, -1, axis=0), traj_list_h))
    datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)

est._initialize_parameters(datas_visible, np.random.default_rng())

muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
new_stat = sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force)


plt.plot(time[:-2], muh[:-2, 0], label="Prediction (with 2 \\sigma error lines)", color="blue")
plt.plot(time[:-2], muh[:-2, 0] + 2 * np.sqrt(Sigh[:-2, 0, 0]), "--", color="blue", linewidth=0.1)
plt.plot(time[:-2], muh[:-2, 0] - 2 * np.sqrt(Sigh[:-2, 0, 0]), "--", color="blue", linewidth=0.1)
# plt.errorbar(time[:-2], muh[:-2, 0], yerr=np.sqrt(Sigh[:-2, 0, 0]), label="Prediction")
plt.plot(time, traj_list_h[:, 0], label="Real", color="orange")
plt.legend(loc="upper right")
plt.show()
