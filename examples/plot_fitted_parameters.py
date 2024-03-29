"""
===========================
Parameters estimation
===========================

Plot obtained values of the parameters versus the actual ones
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from GLE_analysisEM.post_processing import memory_kernel, forcefield, forcefield_plot2D, correlation, memory_timescales

from sklearn.preprocessing import FunctionTransformer

a = 0.025
b = 1.0


def dV(X):
    """
    Compute the force field
    """
    return 4 * a * np.power(X, 3) - 2 * b * X


dim_x = 1
dim_h = 2
random_state = None
force = -np.identity(dim_x)
# force = [[-0.25, -1], [1, -0.25]]
A = np.array([[5e-2, -1.0], [1.0, 0.1]])

# ------ Generation ------#
# pot_gen = GLE_BasisTransform(basis_type="linear")
pot_gen = GLE_BasisTransform(transformer=FunctionTransformer(dV))
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, basis=pot_gen, force_init=force, init_params="random", random_state=random_state)
X, idx, Xh = generator.sample(n_samples=20000, n_trajs=25, x0=0.0, v0=0.0)

# ------ Estimation ------#
# basis = GLE_BasisTransform(basis_type="linear")
basis = GLE_BasisTransform(basis_type="free_energy_kde")
estimator = GLE_Estimator(verbose=2, verbose_interval=10, dim_x=dim_x, dim_h=dim_h, basis=basis, n_init=1, random_state=7, tol=1e-7, no_stop=False, max_iter=100, multiprocessing=8)
estimator.fit(X, idx_trajs=idx)

# ------ Plotting ------#
fig, axs = plt.subplots(2, 2)


# ------ Force field ------#
axs[0, 0].set_title("Force field")
force_true = generator.get_coefficients()["force"]
force_fitted = estimator.get_coefficients()["force"]
if dim_x == 1:
    x_lims = [[-10, 10, 25]]
    xfx_true = forcefield(x_lims, pot_gen, force_true)
    xfx = forcefield(x_lims, basis, force_fitted)
    axs[0, 0].plot(xfx_true[:, 0], xfx_true[:, 1], label="True force field")
    axs[0, 0].plot(xfx[:, 0], xfx[:, 1], label="Fitted force field")

if dim_x == 2:
    x_lims = [[-2, 2, 10], [-2, 2, 10]]
    x_true, y_true, fx_true, fy_true = forcefield_plot2D(x_lims, basis, force_true)
    x, y, fx, fy = forcefield_plot2D(x_lims, basis, force_fitted)
    axs[0, 0].quiver(x_true, y_true, fx_true, fy_true, width=0.001, color="green", label="True force field")
    axs[0, 0].quiver(x, y, fx, fy, width=0.001, color="blue", label="Fitted force field")

axs[0, 0].legend(loc="upper right")

# ------ Memory kernel ------#
axs[0, 1].set_title("Memory kernel")
time, kernel = memory_kernel(1000, estimator.dt, estimator.get_coefficients(), dim_x)
time_true, kernel_true = memory_kernel(1000, generator.dt, generator.get_coefficients(), dim_x)


axs[0, 1].plot(time, kernel[:, 0, 0], label="Fitted memory kernel")
axs[0, 1].plot(time_true, kernel_true[:, 0, 0], label="True memory kernel")
axs[0, 1].legend(loc="upper right")

# ------ Memory eigenvalues ------#
axs[1, 1].set_title("Kernel Eigenvalues")
mem_ev = memory_timescales(estimator.get_coefficients(), dim_x=dim_x)
mem_ev_true = memory_timescales(generator.get_coefficients(), dim_x=dim_x)
axs[1, 1].scatter(np.real(mem_ev), np.imag(mem_ev), label="Ev fitted")
axs[1, 1].scatter(np.real(mem_ev_true), np.imag(mem_ev_true), label="Ev true")
# axs[1, 1].set_aspect(1)


def simulated_vacf(estimator):
    """
    Get vacf via numericall simulation of the model
    """
    Ntrajs = 5
    X, idx, Xh = estimator.sample(n_samples=10000, n_trajs=Ntrajs)
    traj_list = np.split(X, idx)
    vacf = 0.0
    for n, trj in enumerate(traj_list):
        vacf += correlation(trj[:, 1 + estimator.dim_x])
        time = trj[:, 0]
    vacf /= vacf[0]
    return time, vacf


# ------ Diffusion ------#
traj_list = np.split(X, idx)
vacf_num = 0.0
for n, trj in enumerate(traj_list):
    vacf_num += correlation(trj[:, 2])
    time = trj[:, 0]
vacf_num /= vacf_num[0]
time_sim, vacf_sim = simulated_vacf(estimator)
axs[1, 0].plot(time[: len(time_sim) // 2], vacf_sim, label="Fitted VACF")
axs[1, 0].set_title("Velocity autocorrelation function")
axs[1, 0].plot(time[: len(time) // 2], vacf_num, label="Numerical VACF")
axs[1, 0].legend(loc="upper right")

plt.show()
