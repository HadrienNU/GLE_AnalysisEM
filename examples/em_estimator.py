"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator


X = np.arange(100).reshape(1, -1)
estimator = GLE_Estimator()
estimator.fit(X)
plt.plot(estimator.predict(X))
plt.show()
