"""
===========================
Plotting GLE Estimator
===========================

An example plot of :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_LinearBasis
from GLE_analysisEM.utils import loadTestDatas_est

X, idx, Xh = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"], 1, 1)
basis = GLE_LinearBasis(dim_x=1)
X_basis = basis.fit_transform(X)
X = np.hstack((X, X_basis))
estimator = GLE_Estimator(verbose=1, EnforceFDT=False)
estimator.fit(X, idx_trajs=idx)
# plt.plot(estimator.nlogL)
plt.plot(X[:, 0], estimator.predict(X)[:, 0], label="Prediction")
plt.plot(X[:, 0], Xh[:, 0], label="Real")
plt.legend(loc="upper right")
plt.show()
