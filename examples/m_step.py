"""
===========================
M step
===========================

Inner working of the M step,  maximum likelihood estimation of the coefficients :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd

# from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, sufficient_stats, sufficient_stats_hidden, preprocessingTraj
from GLE_analysisEM.utils import loadTestDatas_est


# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

dim_x = 1
dim_h = 1
model = "aboba"
force = -np.identity(dim_x)

basis = GLE_BasisTransform(basis_type="linear")
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, EnforceFDT=True, force_init=force, init_params="random", model=model)
X, idx, Xh = generator.sample(n_samples=10000, n_trajs=50, x0=0.0, v0=0.0, basis=basis)

# X, idx, Xh = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat"], 1, 1)
# est = GLE_Estimator(init_params="random", EnforceFDT=False, OptimizeDiffusion=True, C_init=np.identity(2), A_init=np.array([[5, 1.0], [-2.0, 0.07]]), force_init=np.array([-1]), dim_x=dim_x, dim_h=dim_h, model=model)
X = basis.fit_transform(X)

est = GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, model=model, EnforceFDT=False, OptimizeDiffusion=True)
est.dt = X[1, 0] - X[0, 0]
est._check_initial_parameters()
est._check_n_features(X)

Xproc, idx = preprocessingTraj(X, idx_trajs=idx, dim_x=est.dim_x, model=model)
traj_list = np.split(Xproc, idx)
traj_list_h = np.split(Xh, idx)

datas = 0.0
for n, traj in enumerate(traj_list):
    datas_visible = sufficient_stats(traj, est.dim_x)
    zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    muh = np.hstack((np.roll(traj_list_h[n], -1, axis=0), traj_list_h[n]))
    datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)
    # print(datas)
est._initialize_parameters(datas_visible, random_state=np.random.default_rng())
print(est.get_coefficients())
logL1, _ = est.loglikelihood(datas)
est._m_step(datas)
logL2, _ = est.loglikelihood(datas)
print(logL1, logL2)
print(est.get_coefficients())
# plt.plot(time[:-2], estimator.predict(X)[:, 0], label="Prediction")
# plt.plot(time, traj_list_h[:, 0], label="Real")
# plt.legend(loc="upper right")
# plt.show()
