"""
===========================
Generating GLE Samples
===========================

Generation of sample trajectory via :class:`GLE_analysisEM.GLE_Estimator.sample`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform


basis = GLE_BasisTransform(basis_type="linear")
estimator = GLE_Estimator(verbose=1, EnforceFDT=False, C_init=np.identity(2), A_init=np.array([[5, 1.0], [-2.0, 0.07]]), force_init=np.array([-1]), mu_init=np.zeros((1,)), sig_init=np.zeros((1, 1)), init_params="user")
X, h = estimator.sample(n_samples=5000, x0=0.0, v0=0.0, basis=basis)

plt.plot(X[:, 0], h[:, 0], label="h")
plt.plot(X[:, 0], X[:, 2], label="v")
plt.plot(X[:, 0], X[:, 1], label="x")


plt.legend(loc="upper right")
plt.show()
