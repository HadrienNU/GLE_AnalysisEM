"""
===========================
Likelihood Evolution
===========================

Evolution of the log_likelihood along fit for train and test trajectories
"""
import numpy as np
import pandas as pd
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform

from matplotlib import pyplot as plt

# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

dim_x = 1
dim_h = 1
random_state = 42
model = "euler_fv"
force = -np.identity(dim_x)
max_iter = 10
N_big_steps = 150
ntrajs = 25

pot_gen = GLE_BasisTransform(basis_type="linear")

# Trajectory generation
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, EnforceFDT=True, force_init=force, init_params="random", model=model, basis=pot_gen, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=10000, n_trajs=ntrajs, x0=0.0, v0=0.0)
X_val, idx_val, Xh_val = generator.sample(n_samples=10000, n_trajs=10, x0=0.0, v0=0.0)
print("Real parameters", generator.get_coefficients())

initial_ll = generator.score(X, idx_trajs=idx)
initial_ll_val = generator.score(X_val, idx_trajs=idx_val)
print("Initial ll", initial_ll, initial_ll_val)

basis = GLE_BasisTransform(basis_type="linear")
# Trajectory estimation
estimator = GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, basis=basis, model=model, EnforceFDT=False, OptimizeDiffusion=True, no_stop=True, max_iter=max_iter, n_init=1, random_state=None, verbose=0, multiprocessing=8)
# We set some initial conditions, check for stability
# estimator.set_init_coeffs(generator.get_coefficients())

logL_train = np.empty((N_big_steps * max_iter,))
logL_val = np.empty((N_big_steps * max_iter,))
for i in range(N_big_steps):
    print("Step {}".format(i))
    estimator.set_params(warm_start=True)
    estimator.fit(X, idx_trajs=idx)
    estimator.get_coefficients()
    logL_train[i * max_iter : (i + 1) * max_iter] = estimator.logL[0, :]
    logL_val[i * max_iter : (i + 1) * max_iter] = estimator.score(X_val, idx_trajs=idx_val)

print(estimator.get_coefficients())


plt.plot(logL_train[1:], label="Log L train")
plt.plot(logL_val[1:], label="Log L validation")
plt.plot([initial_ll] * N_big_steps * max_iter, label="Initial ll train")
plt.plot([initial_ll_val] * N_big_steps * max_iter, label="Initial ll validation")
plt.legend(loc="upper right")
plt.show()
