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
from GLE_analysisEM.utils import memory_kernel, forcefield, forcefield_plot2D, correlation, memory_timescales

import copy

dim_x = 1
dim_h = 1
random_state = 42
model = "aboba"
force = -np.identity(dim_x)

A = np.array([[5, 1.0], [-1.0, 0.5]])
C = np.identity(dim_x + dim_h)
# ------ Generation ------#
pot_gen = GLE_BasisTransform(basis_type="linear")
# pot_gen_polynom = GLE_BasisTransform(basis_type="polynomial", degree=3)
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, EnforceFDT=False, force_init=force, init_params="user", A_init=A, C_init=C, model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=10000, n_trajs=25, x0=0.0, v0=0.0, basis=pot_gen)
print(generator.get_coefficients())
inits_coeffs = generator.get_coefficients()
inits_coeffs_reset = copy.deepcopy(inits_coeffs)

# ------ Estimation ------#
basis = GLE_BasisTransform(basis_type="linear")
X = basis.fit_transform(X)
estimator = GLE_Estimator(verbose=2, verbose_interval=10, init_params="user", dim_x=dim_x, dim_h=dim_h, model=model, n_init=1, EnforceFDT=False, random_state=None, tol=1e-3, no_stop=False)
estimator.set_init_coeffs(generator.get_coefficients())
estimator.dt = generator.dt
estimator._initialize_parameters(random_state=None)
print(estimator.score(X, idx_trajs=idx))
# print(estimator.get_coefficients())

# ------ Plotting ------#
nb_points = 25
fig, axs = plt.subplots(1, 3)
x_coords = np.linspace(4.7, 5.2, nb_points)
y_coords = np.linspace(0.05, 10, nb_points)
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
axs[1].set_title("Cross section along A[1,1]")
for i in range(nb_points):
    axs[1].plot(y_coords, score_val[:, i])


inits_coeffs = inits_coeffs_reset
sig_coords = np.linspace(0.05, 10, 25)
score_val_sig = np.empty(sig_coords.shape)
for i, sig in enumerate(sig_coords):
    inits_coeffs["Σ_0"] = sig
    estimator.set_init_coeffs(inits_coeffs)
    estimator._initialize_parameters(random_state=None)
    score_val_sig[i] = estimator.score(X, idx_trajs=idx)
axs[2].plot(sig_coords, score_val_sig)

inits_coeffs = inits_coeffs_reset
sig_coords = np.linspace(-10, 10, 25)
score_val_sig = np.empty(sig_coords.shape)
for i, sig in enumerate(sig_coords):
    inits_coeffs["µ_0"] = sig
    estimator.set_init_coeffs(inits_coeffs)
    estimator._initialize_parameters(random_state=None)
    score_val_sig[i] = estimator.score(X, idx_trajs=idx)
axs[2].plot(sig_coords, score_val_sig)
axs[2].set_title("Loglikelihood along µ_0,Σ_0 ")
plt.show()
