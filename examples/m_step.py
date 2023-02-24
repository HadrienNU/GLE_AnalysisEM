"""
===========================
M step
===========================

Inner working of the M step,  maximum likelihood estimation of the coefficients :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd

# from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, adder
from sklearn.preprocessing import FunctionTransformer

# from GLE_analysisEM.utils import loadTestDatas_est


# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# dim_x = 1
# dim_h = 1
# model = "aboba"
# force = -np.identity(dim_x)
#
a = 0.025
b = 1.0


def dV(X):
    """
    Compute the force field
    """
    return -4 * a * np.power(X, 3) + 2 * b * X


# generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, EnforceFDT=True, force_init=force, init_params="random", model=model, random_state=42)
# X, idx, Xh = generator.sample(n_samples=1000, n_trajs=500, x0=0.0, v0=0.0, basis=basis)

dim_x = 1
dim_h = 1
random_state = 42
force = -np.identity(dim_x)

A = np.array([[5e-5, 1.0], [-1.0, 0.5]])
C = np.identity(dim_x + dim_h)
# ------ Generation ------#
pot_gen = GLE_BasisTransform(basis_type="linear")
# pot_gen_polynom = GLE_BasisTransform(basis_type="polynomial", degree=3)
# pot_gen = GLE_BasisTransform(transformer=FunctionTransformer(dV))
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, basis=pot_gen, force_init=force, init_params="random", C_init=C, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=10000, n_trajs=25, x0=0.0, v0=0.0)

basis = GLE_BasisTransform(basis_type="linear")
# basis = GLE_BasisTransform(basis_type="polynomial", degree=3)

est = GLE_Estimator(init_params="user", dim_x=dim_x, dim_h=dim_h, basis=basis, OptimizeDiffusion=True)
est.set_init_coeffs(generator.get_coefficients())
est.dt = X[1, 0] - X[0, 0]
est._check_initial_parameters()


traj_list_h = np.split(Xh, idx)
# for n, traj in enumerate(traj_list_h):
#     traj_list_h[n] = traj_list_h[n][:-1, :]  # For euler

Xproc, idx = est.model_class.preprocessingTraj(basis, X, idx_trajs=idx)
traj_list = np.split(Xproc, idx)

est._initialize_parameters(random_state=42)
datas = {}
for n, traj in enumerate(traj_list):
    datas_visible = est.model_class.sufficient_stats(traj, est.dim_x)
    zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    # muh = np.hstack((np.roll(traj_list_h[n], -1, axis=0), traj_list_h[n]))
    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    adder(datas, est.model_class.sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force), len(traj_list))
    # print(datas)

print(generator.get_coefficients())
logL1 = est.loglikelihood(datas)
print(logL1)

est._m_step(datas)
logL3 = est.loglikelihood(datas)
print("Analytic", est.get_coefficients())
print("Diff")
anal_coef = est.get_coefficients()
