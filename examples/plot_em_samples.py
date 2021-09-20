"""
===========================
Generating GLE Samples
===========================

Generation of sample trajectory via :class:`GLE_analysisEM.GLE_Estimator.sample`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from sklearn.preprocessing import FunctionTransformer


a = 0.025
b = 1.0


def dV(X):
    """
    Compute the force field
    """
    return -4 * a * np.power(X, 3) + 2 * b * X


dim_x = 1
dim_h = 1
model = "euler"
force = np.identity(dim_x)

basis = GLE_BasisTransform(transformer=FunctionTransformer(dV))
generator = GLE_Estimator(verbose=1, dim_x=dim_x, dim_h=dim_h, model=model, basis=basis, force_init=force, init_params="random", multiprocessing=4)
X, idx, h = generator.sample(n_samples=5000, x0=0.0, v0=0.0)
print(generator.get_coefficients())
for n in range(dim_h):
    plt.plot(X[:, 0], h[:, n], label="h{}".format(n + 1))

for n in range(dim_x):
    plt.plot(X[:, 0], X[:, n * 2 + 2], label="v{}".format(n + 1))
    plt.plot(X[:, 0], X[:, n * 2 + 1], label="x{}".format(n + 1))


plt.legend(loc="upper right")
plt.show()
