"""
===========================
Running GLE Estimator
===========================

An example plot of :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from GLE_analysisEM.utils import loadTestDatas_est

X, idx, Xh = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"], 1, 1)
basis = GLE_BasisTransform()
X = basis.fit_transform(X)
estimator = GLE_Estimator(verbose=2, EnforceFDT=True, C_init=np.identity(2), force_init=np.array([-1]), no_stop=True, n_init=2)
estimator.fit(X, idx_trajs=idx)

for n in range(estimator.logL.shape[0]):
    plt.plot(estimator.logL[n], label="Iter {}".format(n + 1))
    plt.plot(estimator.logL_norm[n], label="Iter {} Normed".format(n + 1))

# plt.plot(X[:, 0], estimator.predict(X)[:, 0], label="Prediction")
# plt.plot(X[:, 0], Xh[:, 0], label="Real")
plt.legend(loc="upper right")
plt.show()
