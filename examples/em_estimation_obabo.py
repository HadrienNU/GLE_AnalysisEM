#!/home/huetleon/.miniconda3/envs/GLE_hidden_velocity/bin/python3
"""
===========================
Running GLE Estimator
===========================
An example of how to run estimation :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform
import numpy as np
import pandas as pd
import scipy.linalg
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, datas_loaders
from glob import glob 

dim_x = 1
hidden_var = 0

dim_h = 0 #dim_x + hidden_var
force = -np.identity(dim_x)
A = [[300.]]
friction = A
print(f"original A = {np.exp(-A[0][0]*0.0005 / 2)}")
datapath="../Langevin_obabo_Harmonic_Force/Langevin_12/"

model = "obabo"

run = 1
verbose = 10
no_stop = True
max_iter = 10**3

paths = glob(datapath+"*_traj.dat")
maxlenght = None


X, idx = datas_loaders.loadData(paths, dim_x, maxlenght=maxlenght)


# Estimation of parameters
basis = GLE_BasisTransform(basis_type="linear")  # Choice of basis for mean force
estimator = GLE_Estimator(dim_x = dim_x, dim_h = dim_h, basis = basis, n_init = 1, verbose = verbose, verbose_interval = 10, model = model, no_stop = no_stop, max_iter = max_iter)
estimator.fit(X, idx_trajs=idx)
print("Estimated parameters", estimator.get_coefficients())

