"""
===========================
Memory kernel estimation
===========================

An example plot of :class:`GLE_analysisEM.utils.memory_kernel`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from GLE_analysisEM.utils import loadTestDatas_est, memory_kernel

X, idx, Xh = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"], 1, 1)
basis = GLE_BasisTransform()
X = basis.fit_transform(X)
estimator = GLE_Estimator(verbose=1, C_init=np.identity(2), EnforceFDT=False)
estimator.fit(X)
time, kernel = memory_kernel(5000, 5e-3, estimator._get_parameters(), 1)
time_true, kernel_true = memory_kernel(5000, 5e-3, {"C": np.identity(2), "A": np.array([[5, 1.0], [-2.0, 0.07]])}, 1)

plt.plot(time, kernel[:, 0, 0], label="Fitted memory kernel")
plt.plot(time_true, kernel_true[:, 0, 0], label="True memory kernel")
plt.legend(loc="upper right")
plt.show()
