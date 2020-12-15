"""
===========================
E step, Kalman filter
===========================

Inner working of the E step :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, sufficient_stats, sufficient_stats_hidden, preprocessingTraj

# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

dim_x = 1
dim_h = 1
model = "euler"
force = -np.identity(dim_x)

basis = GLE_BasisTransform(basis_type="linear")
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, EnforceFDT=True, force_init=force, init_params="random", model=model)
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=50, x0=0.0, v0=0.0, basis=basis)
traj_list_h = np.split(Xh, idx)
time = np.split(X, idx)[0][:-1, 0]

print(generator.get_coefficients())

X = basis.fit_transform(X)

est = GLE_Estimator(init_params="user", dim_x=dim_x, dim_h=dim_h, model=model)
est.set_init_coeffs(generator.get_coefficients())
est.dt = time[1] - time[0]
est._check_initial_parameters()


est._check_n_features(X)
Xproc, idx = preprocessingTraj(X, idx_trajs=idx, dim_x=est.dim_x, model=model)
traj_list = np.split(Xproc, idx)
#
# # Check velocity computation
# for n in range(dim_x):
#     plt.plot(time, traj_list[0][:, n * 2 + 1], label="v{}".format(n + 1))
#     plt.plot(X[:, 0], X[:, n * 2 + 2], label="v_true{}".format(n + 1))
#     plt.legend(loc="upper right")
# plt.show()


datas = 0.0
for n, traj in enumerate(traj_list):
    datas_visible = sufficient_stats(traj, est.dim_x)
    zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    muh = np.hstack((np.roll(traj_list_h[n][:-1, :], -1, axis=0), traj_list_h[n][:-1, :]))
    datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)

est._initialize_parameters(datas_visible / len(traj_list), np.random.default_rng())

print(datas)
new_stat = 0.0
for n, traj in enumerate(traj_list):
    datas_visible = sufficient_stats(traj, est.dim_x)
    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)
print(new_stat)

# np.savetxt("E_step.dat", np.vstack((X[:-1, 0], muh[:-1, 0], np.sqrt(Sigh[:-1, 0, 0]), Xh[:-1, 0])).T)

for k in range(dim_h):
    plt.plot(time, muh[:, k], label="Prediction (with \\pm 2 \\sigma error lines)", color="blue")
    plt.plot(time, muh[:, k] + 2 * np.sqrt(Sigh[:, k, k]), "--", color="blue", linewidth=0.1)
    plt.plot(time, muh[:, k] - 2 * np.sqrt(Sigh[:, k, k]), "--", color="blue", linewidth=0.1)
    plt.plot(time, traj_list_h[n][:-1, k], label="Real", color="orange")
    plt.legend(loc="upper right")
    plt.show()
