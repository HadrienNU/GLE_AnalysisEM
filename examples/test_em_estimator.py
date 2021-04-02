"""
===========================
Running GLE Estimator
===========================

Result of the fit as a function of the number of trajectories for :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform

# from GLE_analysisEM.utils import loadTestDatas_est
# , "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"
# X, idx, Xh = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"], 1, 1)
# estimator = GLE_Estimator(verbose=3, init_params="user", EnforceFDT=False, OptimizeDiffusion=False, dim_h=1, A_init=np.array([[5, 1.0], [-2.0, 0.07]]), C_init=np.identity(2), force_init=[-1], no_stop=True, n_init=1, random_state=42)


# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

dim_x = 1
dim_h = 1
random_state = 42
model = "euler_fv"
force = -np.identity(dim_x)
max_iter = 200

ntrajs = 150

basis = GLE_BasisTransform(basis_type="linear", model=model)

# Trajectory generation
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, EnforceFDT=True, force_init=force, init_params="user", model=model, random_state=random_state, A_init=[[5, 1.0], [-2.0, 0.07]])
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=ntrajs, x0=0.0, v0=0.0, basis=basis)
print("Real parameters", generator.get_coefficients())

X = basis.fit_transform(X)
print("Initial ll", generator.score(X, idx_trajs=idx))


# Trajectory estimation
estimator = GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, model=model, EnforceFDT=False, OptimizeDiffusion=True, no_stop=True, max_iter=max_iter, n_init=1, random_state=random_state + 1, verbose=1, verbose_interval=50, multiprocessing=32)
# We set some initial conditions, check for stability
# estimator.set_init_coeffs(generator.get_coefficients())
estimator.fit(X, idx_trajs=idx)
print(estimator.get_coefficients())

# plt.plot(estimator.logL[0], label="Log L")
# plt.legend(loc="upper right")
# plt.show()
