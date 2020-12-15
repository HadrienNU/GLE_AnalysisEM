"""
===========================
Memory kernel estimation
===========================

An example plot of :class:`GLE_analysisEM.utils.memory_kernel`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from GLE_analysisEM.utils import memory_kernel

dim_x = 1
dim_h = 1
random_state = None
model = "aboba"
force = -np.identity(dim_x)
max_iter = 10

# X, idx, Xh = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"], 1, 1)
basis = GLE_BasisTransform()
generator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, EnforceFDT=False, force_init=force, init_params="random", model=model, random_state=random_state)
X, idx, Xh = generator.sample(n_samples=5000, n_trajs=20, x0=0.0, v0=0.0, basis=basis)

X = basis.fit_transform(X)
estimator = GLE_Estimator(verbose=2, dim_x=dim_x, dim_h=dim_h, model=model, EnforceFDT=False)
estimator.fit(X)
time, kernel = memory_kernel(5000, estimator.dt, estimator.get_coefficients(), dim_x)
time_true, kernel_true = memory_kernel(5000, generator.dt, generator.get_coefficients(), dim_x)

plt.plot(time, kernel[:, 0, 0], label="Fitted memory kernel")
plt.plot(time_true, kernel_true[:, 0, 0], label="True memory kernel")
plt.legend(loc="upper right")
plt.show()
