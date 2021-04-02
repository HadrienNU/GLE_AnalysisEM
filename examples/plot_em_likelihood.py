"""
===========================
Inner working
===========================

Inner working of the fit method :class:`GLE_analysisEM.GLE_Estimator.fit`
Plot the likelihood increase after one EM step
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, sufficient_stats, sufficient_stats_hidden

# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

dim_x = 1
dim_h = 1
model = "euler"
random_state = 42
force = -np.identity(dim_x)
nbTrajs = 10

to_plot_logL_true_true = np.zeros((nbTrajs, 1))
to_plot_logL_true_datas = np.zeros((nbTrajs, 3))
to_plot_logL_true_param = np.zeros((nbTrajs, 3))
to_plot_logL = np.zeros((nbTrajs, 4))

basis = GLE_BasisTransform(basis_type="linear")
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, basis=basis, EnforceFDT=True, force_init=force, init_params="random", model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=nbTrajs, x0=0.0, v0=0.0)
traj_list_h = np.split(Xh, idx)

# print(generator.get_coefficients())

# An estimator that hold the  real parameter for comparaison
true_est = GLE_Estimator(init_params="user", dim_x=dim_x, dim_h=dim_h, model=model, basis=basis)
true_est.set_init_coeffs(generator.get_coefficients())
true_est.dt = X[1, 0] - X[0, 0]
true_est._check_initial_parameters()
true_est._initialize_parameters(random_state=random_state)

est = GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, model=model, OptimizeDiffusion=True, basis=basis)
est.set_init_coeffs(generator.get_coefficients())
est.dt = X[1, 0] - X[0, 0]
est._check_initial_parameters()
Xproc, idx = est.model_class.preprocessingTraj(est.basis, X, idx_trajs=idx)
traj_list = np.split(Xproc, idx)


for n, traj in enumerate(traj_list):
    print(n)
    datas_visible = sufficient_stats(traj, est.dim_x)
    zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    muh = np.hstack((np.roll(traj_list_h[n][:-1, :], -1, axis=0), traj_list_h[n][:-1, :]))
    print("True")
    est._initialize_parameters(random_state=random_state)
    datas_true = sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force)

    to_plot_logL_true_true[n, 0] = true_est.loglikelihood(datas_true)  # The true likelihood of the true datas

    to_plot_logL_true_datas[n, 0] = est.loglikelihood(datas_true)  # The initial likelihood on the true datas

    # A first EM step
    print("E")
    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    new_stat = sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force)
    # hidenS = est.hidden_entropy(Sigh, est.dim_h)
    to_plot_logL[n, 0] = est.loglikelihood(new_stat) + new_stat["hS"]  # The initial likelihood
    to_plot_logL_true_param[n, 0] = true_est.loglikelihood(new_stat) + new_stat["hS"]  # The initial datas of the true param
    print("M")
    est._m_step(new_stat)
    to_plot_logL[n, 1] = est.loglikelihood(new_stat) + new_stat["hS"]  # After M step
    to_plot_logL_true_datas[n, 1] = est.loglikelihood(datas_true)  # After M step on the true datas
    # logL2 = est.loglikelihood(datas)
    # to_plot_logL[n,0] =
    # A second EM step
    print("E")
    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    new_stat = sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force)
    to_plot_logL[n, 2] = est.loglikelihood(new_stat) + new_stat["hS"]  # After ME step
    to_plot_logL_true_param[n, 1] = true_est.loglikelihood(new_stat) + new_stat["hS"]  # The true likelihood of the true param
    print("M")
    est._m_step(new_stat)
    to_plot_logL[n, 3] = est.loglikelihood(new_stat)  # After MEM step
    to_plot_logL_true_datas[n, 2] = est.loglikelihood(datas_true)  # The initial likelihood on the true datas


fig, (ax1, ax2, ax3) = plt.subplots(3)

# plt.plot(to_plot_logL[:, 0], label="Initial likelihood")
ax1.set_title("Estimated")
ax1.plot(to_plot_logL[:, 1] - to_plot_logL[:, 0], label="After M step")
ax1.plot(to_plot_logL[:, 2] - to_plot_logL[:, 0], label="After ME step")
ax1.plot(to_plot_logL[:, 3] - to_plot_logL[:, 0], label="After MEM step")
ax1.legend(loc="upper right")

ax2.set_title("True param")
# plt.plot(to_plot_logL_true_param[:, 0], label="Initial likelihood")
ax2.plot(to_plot_logL_true_param[:, 1] - to_plot_logL_true_param[:, 0], label="After E step")
ax2.plot(to_plot_logL_true_param[:, 2] - to_plot_logL_true_param[:, 1], label="After EE step")
ax2.legend(loc="upper right")

ax3.set_title("True datas")
# plt.plot(to_plot_logL_true_datas[:, 0], label="Initial likelihood")
ax3.plot(to_plot_logL_true_datas[:, 1] - to_plot_logL_true_datas[:, 0], label="After M step")
ax3.plot(to_plot_logL_true_datas[:, 2] - to_plot_logL_true_datas[:, 1], label="After MM step")
ax3.legend(loc="upper right")
plt.show()
