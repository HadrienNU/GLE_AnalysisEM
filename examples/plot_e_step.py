"""
===========================
E step, Kalman filter
===========================

Inner working of the E step :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, adder

# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

dim_x = 1
dim_h = 1
random_state = None
force = -np.identity(dim_x)
A = [[5, 1.0], [-1.0, 2.07]]
C = np.identity(dim_x + dim_h)  #
basis = GLE_BasisTransform(basis_type="linear")
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, basis=basis, C_init=C, force_init=force, init_params="random", random_state=random_state)
X, idx, Xh = generator.sample(n_samples=10000, n_trajs=10, x0=0.0, v0=0.0)
traj_list_h = np.split(Xh, idx)
time = np.split(X, idx)[0][:, 0]
for n, traj in enumerate(traj_list_h):
    traj_list_h[n] = traj_list_h[n][:-1, :]

print(generator.get_coefficients())

est = GLE_Estimator(init_params="user", dim_x=dim_x, dim_h=dim_h, basis=basis)
est.set_init_coeffs(generator.get_coefficients())
est.dt = time[1] - time[0]
est._check_initial_parameters()

Xproc, idx = est.model_class.preprocessingTraj(est.basis, X, idx_trajs=idx)
traj_list = np.split(Xproc, idx)
est.dim_coeffs_force = est.basis.nb_basis_elt_

datas = {}
for n, traj in enumerate(traj_list):
    datas_visible = est.model_class.sufficient_stats(traj, est.dim_x)
    zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    muh = np.hstack((np.roll(traj_list_h[n], -1, axis=0), traj_list_h[n]))
    adder(datas, est.model_class.sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force), len(traj_list))

est._initialize_parameters(None)
print(est.get_coefficients())
print("Real datas")
print(datas)
new_stat = {}
noise_corr = 0.0
for n, traj in enumerate(traj_list):
    datas_visible = est.model_class.sufficient_stats(traj, est.dim_x)
    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    adder(new_stat, est.model_class.sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force), len(traj_list))
print("Estimated datas")
print(new_stat)
print("Diff")
print((new_stat - datas) / np.abs(datas))


Pf = np.zeros((dim_x + dim_h, dim_x))
Pf[:dim_x, :dim_x] = 5e-3 * np.identity(dim_x)
YX = new_stat["xdx"].T - np.matmul(Pf, np.matmul(force, new_stat["bkx"]))
XX = new_stat["xx"]
A = -np.matmul(YX, np.linalg.inv(XX))


Pf = np.zeros((dim_x + dim_h, dim_x))
Pf[:dim_x, :dim_x] = 5e-3 * np.identity(dim_x)

# A = generator.friction_coeffs
print(A)
bkbk = np.matmul(Pf, np.matmul(np.matmul(force, np.matmul(new_stat["bkbk"], force.T)), Pf.T))
bkdx = np.matmul(Pf, np.matmul(force, new_stat["bkdx"]))
bkx = np.matmul(Pf, np.matmul(force, new_stat["bkx"]))

residuals = new_stat["dxdx"] + np.matmul(A, new_stat["xdx"]) + np.matmul(A, new_stat["xdx"]).T - bkdx.T - bkdx
residuals += np.matmul(A, np.matmul(new_stat["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk
print(residuals, generator.diffusion_coeffs)
# SST = 0.5 * (residuals + residuals.T)

fig, axs = plt.subplots(1, dim_h)
# plt.show()
for k in range(dim_h):
    axs.plot(time[:-1], muh[:, k], label="Prediction (with \\pm 2 \\sigma error lines)", color="blue")
    axs.plot(time[:-1], muh[:, k] + 2 * np.sqrt(Sigh[:, k, k]), "--", color="blue", linewidth=0.1)
    axs.plot(time[:-1], muh[:, k] - 2 * np.sqrt(Sigh[:, k, k]), "--", color="blue", linewidth=0.1)
    axs.plot(time[:-1], traj_list_h[n][:, k], label="Real", color="orange")
    axs.legend(loc="upper right")
plt.show()
