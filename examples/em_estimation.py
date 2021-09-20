"""
===========================
Running GLE Estimator
===========================

An example of how to run estimation :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform


# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

dim_x = 1
dim_h = 1
random_state = 42
force = -np.identity(dim_x)
max_iter = 200

ntrajs = 25

pot_gen = GLE_BasisTransform(basis_type="linear")

# Trajectory generation
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, force_init=force, init_params="random", basis=pot_gen, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=ntrajs, x0=0.0, v0=0.0)
print("Real parameters", generator.get_coefficients())

basis = GLE_BasisTransform(basis_type="free_energy")
# Estimation of parameters
estimator = GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, basis=basis, OptimizeDiffusion=True, no_stop=True, max_iter=max_iter, n_init=1, random_state=random_state + 1, verbose=1, verbose_interval=50, multiprocessing=8)
estimator.fit(X, idx_trajs=idx)
print("Estimated parameters", estimator.get_coefficients())
