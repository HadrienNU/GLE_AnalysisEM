"""
===========================
Parameters estimation
===========================

An example plot of :class:`GLE_analysisEM.utils`
Plot obtained values of the parameters versus the actual ones
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from GLE_analysisEM.utils import memory_kernel, forcefield, forcefield_plot2D, correlation

dim_x = 1
dim_h = 2
random_state = 42
model = "aboba"
force = -np.identity(dim_x)
# force = [[-0.25, -1], [1, -0.25]]

# ------ Generation ------#
pot_gen = GLE_BasisTransform(basis_type="linear")
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, EnforceFDT=False, force_init=force, init_params="random", model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=50, x0=0.0, v0=0.0, basis=pot_gen)
print(generator.get_coefficients())


# ------ Estimation ------#
basis = GLE_BasisTransform(basis_type="bsplines")
X = basis.fit_transform(X)
estimator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, model=model, n_init=10, EnforceFDT=False, random_state=None)
estimator.fit(X, idx_trajs=idx)
# print(estimator.get_coefficients())

# ------ Plotting ------#
fig, axs = plt.subplots(2, 2)


# ------ Force field ------#
axs[0, 0].set_title("Force field")
force_fitted = estimator.get_coefficients()["force"]
if dim_x == 1:
    x_lims = [[-2, 2, 25]]
    xfx_true = forcefield(x_lims, basis, force)
    xfx = forcefield(x_lims, basis, force_fitted)
    axs[0, 0].plot(xfx_true[:, 0], xfx_true[:, 1], label="True force field")
    axs[0, 0].plot(xfx[:, 0], xfx[:, 1], label="Fitted force field")

if dim_x == 2:
    x_lims = [[-2, 2, 10], [-2, 2, 10]]
    x_true, y_true, fx_true, fy_true = forcefield_plot2D(x_lims, basis, force)
    x, y, fx, fy = forcefield_plot2D(x_lims, basis, force_fitted)
    axs[0, 0].quiver(x_true, y_true, fx_true, fy_true, width=0.001, color="green", label="True force field")
    axs[0, 0].quiver(x, y, fx, fy, width=0.001, color="blue", label="Fitted force field")

axs[0, 0].legend(loc="upper right")

# ------ Memory kernel ------#
axs[0, 1].set_title("Memory kernel")
time, kernel = memory_kernel(500, estimator.dt, estimator.get_coefficients(), dim_x)
time_true, kernel_true = memory_kernel(500, generator.dt, generator.get_coefficients(), dim_x)


axs[0, 1].plot(time, kernel[:, 0, 0], label="Fitted memory kernel")
axs[0, 1].plot(time_true, kernel_true[:, 0, 0], label="True memory kernel")
axs[0, 1].legend(loc="upper right")
# ------ Diffusion ------#
traj_list = np.split(X, idx)
vacf_num = 0.0
for n, trj in enumerate(traj_list):
    vacf_num += correlation(trj[:, 2])
    time = trj[:, 0]
vacf_num /= len(traj_list)

axs[1, 0].set_title("Velocity autocorrelation function")
axs[1, 0].plot(time, vacf_num, label="Numerical VACF")
axs[1, 0].legend(loc="upper right")

plt.show()
