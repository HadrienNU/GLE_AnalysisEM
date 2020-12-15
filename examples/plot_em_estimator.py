"""
===========================
Running GLE Estimator
===========================

An example plot of :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
model = "aboba"
force = -np.identity(dim_x)
basis = GLE_BasisTransform(basis_type="linear", model=model)
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, EnforceFDT=False, force_init=force, init_params="random", model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=2, x0=0.0, v0=0.0, basis=basis)
X = basis.fit_transform(X)

estimator = GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, model=model, EnforceFDT=False, OptimizeDiffusion=True, no_stop=False, n_init=1, random_state=random_state, verbose=2)
estimator.set_init_coeffs(generator.get_coefficients())
print(generator.get_coefficients())
estimator.fit(X, idx_trajs=idx)

for n in range(estimator.logL.shape[0]):
    plt.plot(estimator.logL[n], label="Iter {}".format(n + 1))
    plt.plot(estimator.logL_norm[n], label="Iter {} Normed".format(n + 1))
print(estimator.get_coefficients())
# plt.plot(X[:, 0], estimator.predict(X)[:, 0], label="Prediction")
# plt.plot(X[:, 0], Xh[:, 0], label="Real")
plt.legend(loc="upper right")
plt.show()
