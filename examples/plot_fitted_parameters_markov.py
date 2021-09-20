"""
===========================
Parameters estimation
===========================

An example plot of :class:`GLE_analysisEM.utils`
Plot obtained values of the parameters versus the actual ones when the parameters are estimated from a Markov model
"""
import numpy as np
from matplotlib import pyplot as plt

from GLE_analysisEM import GLE_BasisTransform, GLE_Estimator, GLE_PotentialTransform

from GLE_analysisEM import Markov_Estimator
from GLE_analysisEM.post_processing import forcefield, forcefield_plot2D, correlation

from sklearn.preprocessing import FunctionTransformer

a = 0.025
b = 1.0


def dV(X):
    """
    Compute the force field
    """
    return 4 * a * np.power(X, 3) - 2 * b * X


dim_x = 1
dim_h = 1
random_state = None
force = -np.identity(dim_x)
# force = [[-0.25, -1], [1, -0.25]]
A = np.array([[5e-2, -1.0], [1.0, 0.1]])

# ------ Generation ------#
# pot_gen = GLE_BasisTransform(basis_type="linear")
pot_gen = GLE_BasisTransform(transformer=FunctionTransformer(dV))
# pot_gen_polynom = GLE_BasisTransform(basis_type="polynomial", degree=3)
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=2, basis=pot_gen, init_params="random", force_init=force, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=20000, n_trajs=25, x0=0.0, v0=0.0)
print("---- Real ones ----")
print(generator.get_coefficients())

for n in range(dim_x):
    plt.plot(X[:, 0], X[:, n * 2 + 2], label="v{}".format(n + 1))
    plt.plot(X[:, 0], X[:, n * 2 + 1], label="x{}".format(n + 1))

plt.show()
# ------ Estimation ------#
# basis = GLE_BasisTransform(basis_type="linear")
basis = GLE_BasisTransform(basis_type="polynomial", degree=3).fit(X[1 : 1 + dim_x])
estimator = Markov_Estimator(init_params="random", verbose=2, verbose_interval=1, dim_x=dim_x, basis=basis, n_init=1, OptimizeForce=True, random_state=7)
estimator.fit(X, idx_trajs=idx)
# print(estimator.get_coefficients())

# Free energy

potential = GLE_PotentialTransform(estimator="histogram", dim_x=dim_x)
potential.fit(X)

# ------ Plotting ------#
fig, axs = plt.subplots(2)


# ------ Force field ------#
axs[0].set_title("Force field")
force_true = generator.get_coefficients()["force"]
force_fitted = estimator.get_coefficients()["force"]
if dim_x == 1:
    x_lims = [[-10, 10, 25]]
    xfx_true = forcefield(x_lims, pot_gen, force_true)
    xfx = forcefield(x_lims, basis, force_fitted)
    axs[0].plot(xfx_true[:, 0], xfx_true[:, 1], label="True force field")
    axs[0].plot(xfx[:, 0], xfx[:, 1], label="Fitted force field")

    x_lims = [[-8, 8, 150]]
    x_val = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2]).reshape(-1, 1)
    pot_val = potential.predict(x_val)
    axs[0].plot(x_val[:, 0], pot_val[:, 0], label="Fitted potential")

if dim_x == 2:
    x_lims = [[-2, 2, 10], [-2, 2, 10]]
    x_true, y_true, fx_true, fy_true = forcefield_plot2D(x_lims, basis, force_true)
    x, y, fx, fy = forcefield_plot2D(x_lims, basis, force_fitted)
    axs[0].quiver(x_true, y_true, fx_true, fy_true, width=0.001, color="green", label="True force field")
    axs[0].quiver(x, y, fx, fy, width=0.001, color="blue", label="Fitted force field")

axs[0].legend(loc="upper right")


def simulated_vacf(estimator, basis):
    """
    Get vacf via numericall simulation of the model
    """
    Ntrajs = 50
    X, idx, Xh = estimator.sample(n_samples=10000, n_trajs=Ntrajs)
    traj_list = np.split(X, idx)
    vacf = 0.0
    for n, trj in enumerate(traj_list):
        vacf += correlation(trj[:, 1 + estimator.dim_x])
        time = trj[:, 0]
    # vacf /= Ntrajs
    return time, vacf / vacf[0]


# ------ Diffusion ------#
traj_list = np.split(X, idx)
vacf_num = 0.0
for n, trj in enumerate(traj_list):
    vacf_num += correlation(trj[:, 2])
    time = trj[:, 0]
# vacf_num /= len(traj_list)
vacf_num /= vacf_num[0]
time_sim, vacf_sim = simulated_vacf(estimator, basis)
axs[1].plot(time[: len(time_sim) // 2], vacf_sim, label="Fitted VACF")
axs[1].set_title("Velocity autocorrelation function")
axs[1].plot(time[: len(time) // 2], vacf_num, label="Numerical VACF")


axs[1].legend(loc="upper right")

plt.show()
