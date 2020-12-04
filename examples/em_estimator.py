"""
===========================
Plotting GLE Estimator
===========================

An example plot of :class:`GLE_analysisEM.GLE_Estimator`
"""

from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator
from GLE_analysisEM.utils import loadTestDatas_est

time, X, traj_list_v, traj_list_h = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat"], {"dim_x": 1, "dim_h": 1})
estimator = GLE_Estimator(verbose=1, EnforceFDT=False)
estimator.fit(X)
print(estimator.logL)
# plt.plot(estimator.nlogL)
plt.plot(time[:-2], estimator.predict(X)[:, 0], label="Prediction")
plt.plot(time, traj_list_h[:, 0], label="Real")
plt.legend(loc="upper right")
plt.show()
