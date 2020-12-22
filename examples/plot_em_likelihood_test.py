"""
===========================
E step, Kalman filter
===========================

Inner working of the fit method :class:`GLE_analysisEM.GLE_Estimator.fit`
Plot the likelihood increase after one EM step
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
model = "aboba"
random_state = 42
force = -np.identity(dim_x)

basis = GLE_BasisTransform(basis_type="linear")
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, EnforceFDT=True, force_init=force, init_params="user", model=model, random_state=random_state, A_init=[[5, 1.0], [-2.0, 0.7]])
X, idx, Xh = generator.sample(n_samples=50000, n_trajs=5, x0=0.0, v0=0.0, basis=basis)
traj_list_h = np.split(Xh, idx)
X = basis.fit_transform(X)
print(generator.get_coefficients())

est = GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, model=model, OptimizeDiffusion=True, random_state=None, max_iter=15, A_init=[[5.5, 0.8], [-1.0, 0.08]], force_init=force)
# est.set_init_coeffs(generator.get_coefficients())
est.dt = X[1, 0] - X[0, 0]

est._check_initial_parameters()
est._check_n_features(X)
Xproc, idx = preprocessingTraj(X, idx_trajs=idx, dim_x=est.dim_x, model=model)
traj_list = np.split(Xproc, idx)
est._initialize_parameters(random_state=random_state)

datas_visible = 0.0
datas_true = 0.0
for n, traj in enumerate(traj_list):
    datas_vis = sufficient_stats(traj, est.dim_x)
    datas_visible += datas_vis / len(traj_list)
    zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    muh = np.hstack((np.roll(traj_list_h[n][:, :], -1, axis=0), traj_list_h[n][:, :]))
    datas_true += sufficient_stats_hidden(muh, zero_sig, traj, datas_vis, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)

print(datas_true)
to_plot_logL_true_true = generator.loglikelihood(datas_true)  # The true likelihood of the true datas
print("Real Likelihood: ", to_plot_logL_true_true)
to_plot_logL_true_datas = np.zeros((est.max_iter + 1,))
to_plot_logL_true_param = np.zeros((est.max_iter,))
to_plot_logL = np.zeros((est.max_iter, 2))
to_plot_logL_true_datas[0] = est.loglikelihood(datas_true)  # The initial likelihood on the true datas

for n_iter in range(est.max_iter):
    # print("E")
    new_stat = 0.0
    for traj in traj_list:
        muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
        new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)
    to_plot_logL[n_iter, 0] = est.loglikelihood(new_stat)  # + new_stat["hS"]  # The initial likelihood
    to_plot_logL_true_param[n_iter] = generator.loglikelihood(new_stat)  # + new_stat["hS"]  # The initial datas of the true param

    # print("M")
    est._m_step(new_stat)
    to_plot_logL[n_iter, 1] = est.loglikelihood(new_stat)  # + new_stat["hS"]  # After M step
    to_plot_logL_true_datas[n_iter + 1] = est.loglikelihood(datas_true)  # After M step on the true datas
print(new_stat)
print(generator.loglikelihood(new_stat))
print(est.get_coefficients())
plt.figure("Log likelihood")
plt.plot(to_plot_logL[:, 0], label="After E step")
plt.plot(to_plot_logL[:, 1], label="After M step")
plt.plot(to_plot_logL_true_datas, label="True Datas")
plt.plot(to_plot_logL_true_param, label="True Param")
# plt.plot(to_plot_logL[:, 2], label="After E step Norm")
# plt.plot(to_plot_logL[:, 3], label="After Mstep Norm")
plt.legend(loc="upper right")

plt.show()
