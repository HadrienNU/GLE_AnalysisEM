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
pd.set_option("display.max_colwidth", None)

dim_x = 1
dim_h = 1
model = "euler"
shift = 0
random_state = None
force = -np.identity(dim_x)
A = [[5, 1.0], [-1.0, 2.07]]
C = np.identity(dim_x + dim_h)  #
basis = GLE_BasisTransform(basis_type="linear")
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, EnforceFDT=False, C_init=C, force_init=force, init_params="random", model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=10000, n_trajs=100, x0=0.0, v0=0.0, basis=basis)
traj_list_h = np.split(Xh, idx)
time = np.split(X, idx)[0][:, 0]
for n, traj in enumerate(traj_list_h):
    traj_list_h[n] = traj_list_h[n][:-1, :]

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
    muh = np.hstack((np.roll(traj_list_h[n], -1, axis=0), traj_list_h[n]))
    datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)

est._initialize_parameters(None)
print(est.get_coefficients())
print("Real datas")
print(datas)
new_stat = 0.0
noise_corr = 0.0
for n, traj in enumerate(traj_list):
    datas_visible = sufficient_stats(traj, est.dim_x)
    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)
    n_c, n_m = est._get_noise_prop(traj)
    noise_corr += n_c / len(traj_list)
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

fig, axs = plt.subplots(1, 1)
# np.savetxt("E_step.dat", np.vstack((X[:-1, 0], muh[:-1, 0], np.sqrt(Sigh[:-1, 0, 0]), Xh[:-1, 0])).T)
axs.plot(noise_corr, label="Noise correlation")
fig, axs = plt.subplots(1, dim_h)
# plt.show()
for k in range(dim_h):
    axs.plot(time[:-1], muh[:, k], label="Prediction (with \\pm 2 \\sigma error lines)", color="blue")
    axs.plot(time[:-1], muh[:, k] + 2 * np.sqrt(Sigh[:, k, k]), "--", color="blue", linewidth=0.1)
    axs.plot(time[:-1], muh[:, k] - 2 * np.sqrt(Sigh[:, k, k]), "--", color="blue", linewidth=0.1)
    axs.plot(time[:-1], traj_list_h[n][:, k], label="Real", color="orange")
    axs.legend(loc="upper right")
plt.show()
