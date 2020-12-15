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
# generator = GLE_Estimator(verbose=1, EnforceFDT=False, C_init=np.identity(2), A_init=np.array([[5, 1.0], [-2.0, 0.07]]), force_init=np.array([-1]), mu_init=np.zeros((1,)), sig_init=np.zeros((1, 1)), init_params="user")
dim_x = 1
dim_h = 1
model = "euler"
force = -np.identity(dim_x)
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, model=model, EnforceFDT=True, force_init=force, init_params="random")

X, idx, h = generator.sample(n_samples=5000, x0=0.0, v0=0.0, basis=basis)
print(X.shape)
for n in range(dim_h):
    plt.plot(X[:, 0], h[:, n], label="h{}".format(n + 1))
# plt.plot(X[:, 0], h[:, 1], label="h2")
for n in range(dim_x):
    plt.plot(X[:, 0], X[:, n * 2 + 2], label="v{}".format(n + 1))
    plt.plot(X[:, 0], X[:, n * 2 + 1], label="x{}".format(n + 1))


plt.legend(loc="upper right")
plt.show()