"""
===========================
Plotting GLE Estimator
===========================

An example plot of :class:`GLE_analysisEM.GLE_Estimator`
"""
import pandas as pd

# from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator
from GLE_analysisEM.utils import loadTestDatas_est
from sklearn.model_selection import train_test_split, GridSearchCV

time, X, traj_list_v, traj_list_h = loadTestDatas_est(["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"], {"dim_x": 1, "dim_h": 1})

X_train, X_test = train_test_split(X)
print(X_train.shape, X_test.shape)
estimator = GLE_Estimator(verbose=1, EnforceFDT=False)
# putting together a parameter grid to search over using grid search
params = {"dim_h": [1, 2], "EnforceFDT": [True, False]}  # setting up the grid search
gs = GridSearchCV(estimator, params, n_jobs=-1, cv=2)
gs.fit(X_train)
print("Best parameters set found on development set:")
print(gs.best_params_)
# building a dataframe from cross-validation data
df_cv_scores = pd.DataFrame(gs.cv_results_).sort_values(by="rank_test_score")
print(df_cv_scores)
# print(estimator.logL)
# plt.plot(estimator.nlogL)
# plt.plot(time[:-2], estimator.predict(X)[:, 0], label="Prediction")
# plt.plot(time, traj_list_h[0, :, 0], label="Real")
# plt.legend(loc="upper right")
# plt.show()
