"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator

# X = np.arange(100).reshape(1, -1)
X = np.loadtxt("../GLE_analysisEM/tests/test_traj.dat").reshape(1, -1)
estimator = GLE_Estimator(verbose=1)
estimator.fit(X)
print(estimator.nlogL)
# plt.plot(estimator.nlogL)
plt.plot(estimator.predict(X))
plt.show()
