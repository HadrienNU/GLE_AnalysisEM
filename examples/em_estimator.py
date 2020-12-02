"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from analysisEM import GLE_Estimator, GLE_Transformer

X = np.arange(50, dtype=np.float).reshape(-1, 1)
X /= 50
estimator = GLE_Transformer()
X_transformed = estimator.fit_transform(X)

plt.plot(X.flatten(), label="Original Data")
plt.plot(X_transformed.flatten(), label="Transformed Data")
plt.title("Plots of original and transformed data")

plt.legend(loc="best")
plt.grid(True)
plt.xlabel("Index")
plt.ylabel("Value of Data")

plt.show()


X = np.arange(100).reshape(100, 1)
y = np.zeros((100,))
estimator = GLE_Estimator()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
