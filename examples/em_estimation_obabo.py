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
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, datas_loaders, adder
from glob import glob 

dim_x = 1
hidden_var = 0

dim_h = 0 #dim_x + hidden_var
force = -np.identity(dim_x)

datapath="../Langevin_obabo_Harmonic_Force/Langevin_12/"

model = "obabo"

run = 1
verbose = 2
paths = glob(datapath+"*_traj.dat")
maxlenght = None

X, idx = datas_loaders.loadData(paths, dim_x, maxlenght=maxlenght)


# Estimation of parameters
basis = GLE_BasisTransform(basis_type="linear")  # Choice of basis for mean force
estimator = GLE_Estimator(dim_x=dim_x, dim_h=dim_h, basis=basis, n_init=1, verbose=2, verbose_interval=10, model=model)
estimator.fit(X, idx_trajs=idx)
print("Estimated parameters", estimator.get_coefficients())

