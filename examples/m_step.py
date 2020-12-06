"""
===========================
M step
===========================

Inner working of the M step,  maximum likelihood estimation of the coefficients :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, sufficient_stats, sufficient_stats_hidden, preprocessingTraj
from GLE_analysisEM.utils import loadTestDatas_est


# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

est = GLE_Estimator(init_params="user", EnforceFDT=True, OptimizeDiffusion=True, C_init=np.identity(2), A_init=np.array([[5, 1.0], [-2.0, 0.07]]))
est._check_initial_parameters()

X, idx, Xh = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat"], 1, 1)
basis = GLE_BasisTransform(dim_x=1)
X = basis.fit_transform(X)

Xproc = preprocessingTraj(X, idx_trajs=idx, dim_x=est.dim_x)
traj_list = np.split(Xproc, idx)
traj_list_h = np.split(Xh, idx)

datas = 0.0
for n, traj in enumerate(traj_list):
    datas_visible = sufficient_stats(traj, est.dim_x, est.dim_coeffs_force) / len(traj_list)
    zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    muh = np.hstack((np.roll(traj_list_h[n], -1, axis=0), traj_list_h[n]))
    datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)
est._initialize_parameters(datas_visible, np.random.default_rng())
print(est._get_parameters())
logL1 = est.loglikelihood(datas)
est._m_step(datas)
logL2 = est.loglikelihood(datas)
print(logL1, logL2)
print(est._get_parameters())
# plt.plot(time[:-2], estimator.predict(X)[:, 0], label="Prediction")
# plt.plot(time, traj_list_h[:, 0], label="Real")
# plt.legend(loc="upper right")
# plt.show()
