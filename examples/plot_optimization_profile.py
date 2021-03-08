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

dim_x = 1
dim_h = 1
random_state = 42
model = "euler_fix_markov"
force = -np.identity(dim_x)
# force = [[-0.25, -1], [1, -0.25]]
A = np.array([[5e-2, -1.0], [1.0, 0.1]])

# ------ Generation ------#
pot_gen = GLE_BasisTransform(basis_type="linear")
# pot_gen = GLE_BasisTransform(transformer=FunctionTransformer(dV))
# pot_gen_polynom = GLE_BasisTransform(basis_type="polynomial", degree=3)
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, A_init=A, EnforceFDT=True, force_init=force, init_params="random", model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=20000, n_trajs=25, x0=0.0, v0=0.0, basis=pot_gen)
print(generator.get_coefficients())


# ------ Estimation ------#
basis = GLE_BasisTransform(basis_type="linear")
# basis = GLE_BasisTransform(basis_type="polynomial", degree=3)
X = basis.fit_transform(X)
estimator = GLE_Estimator(init_params="markov", verbose=2, verbose_interval=10, force_init=force, dim_x=dim_x, dim_h=dim_h, model=model, n_init=1, EnforceFDT=False, OptimizeForce=False, random_state=7, tol=1e-6, no_stop=False)
estimator.fit(X, idx_trajs=idx)
with open("fit_trajs.pkl", "wb") as output:
    pickle.dump(estimator.coeffs_list_all, output)
coeffs_list_all = estimator.coeffs_list_all
# coef_trajs = open("fit_trajs.pkl", "rb")
# coeffs_list_all = pickle.load(coef_trajs)
# ------ Plotting ------#
nb_plot = 4
fig, axs = plt.subplots(1, nb_plot)
# output = open("fig_profile.pkl", "rb")
# fig = pickle.load(output)
# axs = fig.axes
for coeffs_list in coeffs_list_all:
    len_iter = len(coeffs_list)

    x = np.empty((nb_plot, len_iter))
    y = np.empty((nb_plot, len_iter))
    for n, step in enumerate(coeffs_list):
        eigs_A = np.linalg.eigvals(step["A"])
        x[0, n] = step["A"][0, 0]
        y[0, n] = step["A"][1, 1]
        # x[1, n] = np.real(eigs_A[0])
        # y[1, n] = np.imag(eigs_A[0])
        # x[2, n] = np.real(eigs_A[1])
        # y[2, n] = np.imag(eigs_A[1])
        x[1, n] = step["force"][0, 0]
        y[1, n] = step["ll"]
        x[2, n] = step["µ_0"][0]
        y[2, n] = step["Σ_0"][0, 0]
        x[3, n] = step["A"][0, 1] * step["A"][1, 0]
        y[3, n] = step["C"][0, 0]
    for i in range(nb_plot):
        axs[i].plot(x[i, :], y[i, :], "-x", label="{}".format(n))
    # axs[1].plot(x[2, :], y[2, :], "-x", label="{}".format(n))
axs[0].set_xlabel("A[0,0]")
axs[0].set_ylabel("A[1,1]")
axs[1].set_xlabel("Force")
axs[1].set_ylabel("ll")
axs[2].set_xlabel("µ_0")
axs[2].set_ylabel("Σ_0")
axs[3].set_xlabel("A[0,1]*A[1,0]")
axs[3].set_ylabel("C[0,0]")
# axs[1].set_title("Cross section along A[1,1]")
# for i in range(nb_points):
#     axs[1].plot(y_coords, score_val[:, i])
#
#
# inits_coeffs = inits_coeffs_reset
# sig_coords = np.linspace(0.05, 10, 25)
# score_val_sig = np.empty(sig_coords.shape)
# for i, sig in enumerate(sig_coords):
#     inits_coeffs["Σ_0"] = sig
#     estimator.set_init_coeffs(inits_coeffs)
#     estimator._initialize_parameters(random_state=None)
#     score_val_sig[i] = estimator.score(X, idx_trajs=idx)
# axs[2].plot(sig_coords, score_val_sig)
#
# inits_coeffs = inits_coeffs_reset
# sig_coords = np.linspace(-10, 10, 25)
# score_val_sig = np.empty(sig_coords.shape)
# for i, sig in enumerate(sig_coords):
#     inits_coeffs["µ_0"] = sig
#     estimator.set_init_coeffs(inits_coeffs)
#     estimator._initialize_parameters(random_state=None)
#     score_val_sig[i] = estimator.score(X, idx_trajs=idx)
# axs[2].plot(sig_coords, score_val_sig)
# axs[2].set_title("Loglikelihood along µ_0,Σ_0 ")
plt.show()
