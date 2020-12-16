"""
===========================
Scikit-learn Interface
===========================

An example of cross validation for :class:`GLE_analysisEM.GLE_Estimator`
"""
import pandas as pd

# from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
from GLE_analysisEM.utils import loadTestDatas_est
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

dim_x = 1
dim_h = 2
random_state = 42
model = "aboba"

max_iter = 10

paths = ["../GLE_analysisEM/tests/0_trajectories.dat", "../GLE_analysisEM/tests/1_trajectories.dat", "../GLE_analysisEM/tests/2_trajectories.dat"]
paths_train, paths_test = train_test_split(paths)

X_train, idx_train, Xh_train = loadTestDatas_est(paths_train, 1, 1)
X_test, idx_test, Xh_test = loadTestDatas_est(paths_test, 1, 1)

pipe = Pipeline([("basis", GLE_BasisTransform(basis_type="linear", model=model)), ("em", GLE_Estimator(init_params="random", dim_x=dim_x, dim_h=dim_h, model=model, no_stop=True, max_iter=max_iter, n_init=1, random_state=None, verbose=1))])

print(X_train.shape, X_test.shape)
estimator = GLE_Estimator(verbose=1, EnforceFDT=False)
# putting together a parameter grid to search over using grid search
params = {"em__dim_h": [1, 2], "em__EnforceFDT": [True, False]}  # setting up the grid search
gs = GridSearchCV(estimator, params, n_jobs=4, cv=2)  # TODO change the cv to get trajectories cross-validation
gs.fit(X_train, idx_trajs=idx_train)
print("Best parameters set found on development set:")
print(gs.best_coeffs_)
# building a dataframe from cross-validation data
df_cv_scores = pd.DataFrame(gs.cv_results_).sort_values(by="rank_test_score")
print(df_cv_scores)
