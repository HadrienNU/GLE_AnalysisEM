"""
===========================
Plotting GLE Estimator
===========================

An example plot of :class:`GLE_analysisEM.GLE_Estimator`
"""
import pandas as pd

# from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from GLE_analysisEM.utils import loadTestDatas_est
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

paths = ["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"]
paths_train, paths_test = train_test_split(paths)

X_train, idx_train, Xh_train = loadTestDatas_est(paths_train, 1, 1)
X_test, idx_test, Xh_test = loadTestDatas_est(paths_test, 1, 1)

pipe = Pipeline([("basis", GLE_BasisTransform(dim_x=1)), ("em", GLE_Estimator(verbose=1, EnforceFDT=False))])
# TODO: Make a pipeline
basis = GLE_BasisTransform(dim_x=1)
X = basis.fit_transform(X)

print(X_train.shape, X_test.shape)
estimator = GLE_Estimator(verbose=1, EnforceFDT=False)
# putting together a parameter grid to search over using grid search
params = {"em__dim_h": [1, 2], "em__EnforceFDT": [True, False]}  # setting up the grid search
gs = GridSearchCV(estimator, params, n_jobs=4, cv=2)
gs.fit(X_train, idx_trajs=idx_train)
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
