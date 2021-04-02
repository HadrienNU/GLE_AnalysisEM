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
import pickle

import copy

dim_x = 1
dim_h = 1
random_state = 42
model = "aboba"
force = -np.identity(dim_x)
A = np.array([[0.5, 1.0], [-1.0, 5]])
A_save = A.copy()
C = np.identity(dim_x + dim_h)
# ------ Generation ------#
pot_gen = GLE_BasisTransform(basis_type="linear")
# pot_gen_polynom = GLE_BasisTransform(basis_type="polynomial", degree=3)
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, basis=pot_gen, EnforceFDT=False, force_init=force, init_params="user", A_init=A, C_init=C, model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=10000, n_trajs=25, x0=0.0, v0=0.0)
print(generator.get_coefficients())
inits_coeffs = generator.get_coefficients()
inits_coeffs_reset = copy.deepcopy(inits_coeffs)

# ------ Estimation ------#
basis = GLE_BasisTransform(basis_type="linear")
estimator = GLE_Estimator(verbose=2, verbose_interval=10, init_params="user", dim_x=dim_x, dim_h=dim_h, model=model, basis=basis, n_init=1, EnforceFDT=False, random_state=None, tol=1e-3, no_stop=False)
estimator.set_init_coeffs(generator.get_coefficients())
estimator.dt = generator.dt
estimator._initialize_parameters(random_state=None)
print(estimator.score(X, idx_trajs=idx))
# print(estimator.get_coefficients())

# inits_coeffs["Σ_0"] = 0.1
# ------ Plotting ------#
nb_points = 25
nb_plot = 4
fig, axs = plt.subplots(1, nb_plot)
inits_coeffs = copy.deepcopy(inits_coeffs_reset)
x_coords = np.linspace(0.4, 0.6, nb_points)
y_coords = np.linspace(1.0, 10, nb_points)
score_val = np.empty((y_coords.shape[0], x_coords.shape[0]))
for i, a in enumerate(x_coords):
    for j, b in enumerate(y_coords):
        A[0, 0] = a
        A[1, 1] = b
        inits_coeffs["A"] = A
        estimator.set_init_coeffs(inits_coeffs)

        estimator._initialize_parameters(random_state=None)
        score_val[j, i] = estimator.score(X, idx_trajs=idx)
        print(i, j, a, b, score_val[j, i])
        # print(estimator.get_coefficients()["A"])
axs[0].contour(x_coords, y_coords, np.log(score_val), cmap="jet", levels=100)
axs[0].set_title("Log log likelihood")
axs[0].set_xlabel("A[0,0]")
axs[0].set_ylabel("A[1,1]")


# axs[1].set_title("Cross section along A[1,1]")
# for i in range(nb_points):
#     axs[1].plot(y_coords, score_val[:, i])


inits_coeffs = copy.deepcopy(inits_coeffs_reset)
force_coords = np.linspace(-2, 0, 25)
score_val_force = np.empty(force_coords.shape)
for i, f in enumerate(force_coords):
    inits_coeffs["force"] = np.array([[f]])
    estimator.set_init_coeffs(inits_coeffs)
    estimator._initialize_parameters(random_state=None)
    score_val_force[i] = estimator.score(X, idx_trajs=idx)
axs[1].plot(force_coords, score_val_force)
axs[1].set_xlabel("Force")
axs[1].set_ylabel("ll")


inits_coeffs = copy.deepcopy(inits_coeffs_reset)
A = A_save.copy()
C = np.identity(dim_x + dim_h)
x_coords = np.linspace(-25, 25, nb_points)
y_coords = np.linspace(-25, 25, nb_points)
score_val = np.empty((y_coords.shape[0], x_coords.shape[0]))
for i, a in enumerate(x_coords):
    for j, b in enumerate(y_coords):
        A[0, 1] = a
        A[1, 0] = b
        inits_coeffs["A"] = A
        # inits_coeffs["µ_0"] = a
        # inits_coeffs["Σ_0"] = b
        estimator.set_init_coeffs(inits_coeffs)

        estimator._initialize_parameters(random_state=None)
        score_val[j, i] = estimator.score(X, idx_trajs=idx)
        print(i, j, a, b, score_val[j, i])
        # print(estimator.get_coefficients()["A"])
axs[2].contour(x_coords, y_coords, np.log(score_val), cmap="jet", levels=100)
axs[2].set_xlabel("A[0,1]")
axs[2].set_ylabel("A[1,0]")
# axs[2].set_xlabel("µ_0")
# axs[2].set_ylabel("Σ_0")
inits_coeffs = copy.deepcopy(inits_coeffs_reset)
A = A_save.copy()
C = np.identity(dim_x + dim_h)
x_coords = np.linspace(0, 10.0, nb_points)
y_coords = np.linspace(0.05, 1.5, nb_points)
score_val = np.empty((y_coords.shape[0], x_coords.shape[0]))
for i, a in enumerate(x_coords):
    for j, b in enumerate(y_coords):
        # C[0, 0] = a
        A[1, 1] = a
        C[1, 1] = b
        inits_coeffs["A"] = A
        inits_coeffs["C"] = C
        estimator.set_init_coeffs(inits_coeffs)

        estimator._initialize_parameters(random_state=None)
        score_val[j, i] = estimator.score(X, idx_trajs=idx)
        print(i, j, a, b, score_val[j, i])
        # print(estimator.get_coefficients()["A"])
axs[3].contour(x_coords, y_coords, np.log(score_val), cmap="jet", levels=100)
axs[3].set_title("Log log likelihood")

axs[3].set_xlabel("A[1,1]")
axs[3].set_ylabel("C[1,1]")

with open("fig_profile.pkl", "wb") as output:
    pickle.dump(fig, output)

# # -------------  Plotting trajs of EM -----------------
#
# fo = open("fit_trajs.pkl", "rb")
# fit_trajs = pickle.load(fo)
#
# for coeffs_list in fit_trajs:
#     len_iter = len(coeffs_list)
#
#     x = np.empty((nb_plot, len_iter))
#     y = np.empty((nb_plot, len_iter))
#     for n, step in enumerate(coeffs_list):
#         x[0, n] = step["A"][0, 0]
#         y[0, n] = step["A"][1, 1]
#         x[1, n] = step["force"][0, 0]
#         y[1, n] = step["ll"]
#         x[2, n] = step["A"][0, 1]
#         y[2, n] = step["A"][1, 0]
#         # x[2, n] = step["µ_0"][0]
#         # y[2, n] = step["Σ_0"][0, 0]
#         x[3, n] = step["A"][1, 1]
#         y[3, n] = step["C"][1, 1]
#     for i in range(nb_plot):
#         axs[i].plot(x[i, :], y[i, :], "-x", label="{}".format(n))
#
# with open("fig_profile_with_trajs.pkl", "wb") as output:
#     pickle.dump(fig, output)

plt.show()
