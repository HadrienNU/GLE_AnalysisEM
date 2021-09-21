"""
===========================
Running GLE Estimator
===========================

An example of how to run estimation :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform

dim_x = 1
dim_h = 1
random_state = 42
force = -np.identity(dim_x)

ntrajs = 25

pot_gen = GLE_BasisTransform(basis_type="linear")

# Trajectory generation
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, force_init=force, init_params="random", basis=pot_gen, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=ntrajs, x0=0.0, v0=0.0)
print("Real parameters", generator.get_coefficients())


# Estimation of parameters
basis = GLE_BasisTransform(basis_type="free_energy")  # Choice of basis for mean force
estimator = GLE_Estimator(dim_x=dim_x, dim_h=dim_h, basis=basis, max_iter=100, n_init=1, random_state=random_state + 1, verbose=1, verbose_interval=50)
estimator.fit(X, idx_trajs=idx)
print("Estimated parameters", estimator.get_coefficients())
