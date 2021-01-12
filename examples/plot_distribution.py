"""
===========================
Potential estimation via density
===========================

An example plot of :class:`GLE_analysisEM.GLE_PotentialTransform`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, GLE_PotentialTransform
from sklearn import linear_model
from sklearn.preprocessing import FunctionTransformer

a = 0.025
b = 1.0


def dV(X):
    """
    Compute the force field
    """
    return -4 * a * np.power(X, 3) + 2 * b * X


dim_x = 1
dim_h = 2
random_state = 23
model = "euler"
force = np.identity(dim_x)
max_iter = 1

# ------ Generation ------#
pot_gen = GLE_BasisTransform(transformer=FunctionTransformer(dV))
# pot_gen_polynom = GLE_BasisTransform(basis_type="polynomial", degree=3)

generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, EnforceFDT=False, force_init=force, init_params="random", model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=10, n_trajs=1, x0=0.0, v0=0.0, basis=pot_gen)
# print(generator.get_coefficients())


# ------ Potential estimation ------#

potential = GLE_PotentialTransform(estimator="histogram", dim_x=dim_x)
potential.fit(X)

# ------ Fitting to basis ------#
# y = potential.transform(X)  # Force values
basis = GLE_BasisTransform(basis_type="bsplines")
X = basis.fit_transform(X)
print(X.shape)
# reg = linear_model.LinearRegression()
# print(y.shape, X[:, 1 + 2 * dim_x :].shape)
# reg.fit(X[:, 1 + 2 * dim_x :], y[:, 1 + 2 * dim_x :])  # We fit bk versus force values
# print(reg.coef_)
# ------ Plotting ------#
fig, axs = plt.subplots(1, 2)


# ------ Potential ------#
axs[0].set_title("Potential")
if dim_x == 1:
    x_lims = [[-2, 2, 25]]
    x_val = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2]).reshape(-1, 1)
    pot_val = potential.predict(x_val)
    axs[0].plot(x_val[:, 0], pot_val[:, 0], label="Fitted potential")

elif dim_x == 2:
    x_lims = [[-2, 2, 10], [-2, 2, 10]]
    x_coords = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2])
    y_coords = np.linspace(x_lims[1][0], x_lims[1][1], x_lims[1][2])
    x, y = np.meshgrid(x_coords, y_coords)
    x_val = np.vstack((x.flatten(), y.flatten())).T
    print(x_val.shape)
    pot_val = potential.predict(x_val).reshape(x_lims[0][2], x_lims[1][2])
    axs[0].contour(x, y, pot_val, cmap="jet")
#
#     x_true, y_true, fx_true, fy_true = forcefield_plot2D(x_lims, basis, force)
#     x, y, fx, fy = forcefield_plot2D(x_lims, basis, force_fitted)
#     axs[0, 0].quiver(x_true, y_true, fx_true, fy_true, width=0.001, color="green", label="True force field")
#     axs[0, 0].quiver(x, y, fx, fy, width=0.001, color="blue", label="Fitted force field")
#
# axs[0].legend(loc="upper right")
#
# ------ Force field ------#
axs[1].set_title("Force field")
if dim_x == 1:
    x_lims = [[-2, 2, 25]]
    x_val = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2]).reshape(-1, 1)
    xfx = potential.transform(x_val)
    axs[1].plot(xfx[:, 0], xfx[:, 1], label="Fitted force field")

if dim_x == 2:
    x_lims = [[-2, 2, 10], [-2, 2, 10]]
    x_coords = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2])
    y_coords = np.linspace(x_lims[1][0], x_lims[1][1], x_lims[1][2])
    x, y = np.meshgrid(x_coords, y_coords)
    x_val = np.vstack((x.flatten(), y.flatten())).T
    xfx = potential.transform(x_val)
    f = xfx.reshape(x_lims[0][2], x_lims[1][2], 4)
    axs[1].quiver(x, y, f[:, :, 2], f[:, :, 3], width=0.001, color="blue", label="Fitted force field")


plt.show()
