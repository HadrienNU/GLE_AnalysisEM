#!/home/huetleon/.miniconda3/envs/GLE_hidden_velocity/bin/python3
"""
===========================
E step, Kalman filter
===========================

Inner working of the E step :class:`GLE_analysisEM.GLE_Estimator`
"""
import numpy as np
import pandas as pd
import scipy.linalg
from matplotlib import pyplot as plt
from GLE_analysisEM import GLE_Estimator, GLE_BasisTransform, datas_loaders, adder
from glob import glob

# Printing options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

datapath="../Langevin_obabo_Harmonic_Force/Langevin_12/"

dim_x = 1
hidden_var = 0
dim_h = 0 # dim_x + hidden_var # hidden speeds and hidden var   
random_state = None
force = -np.identity(dim_x) * 500
A = [[300.]]
friction = A
print(f"original A = {np.exp(-A[0][0]*0.0005/2)}")
print("C'est le bon fichier")
C = np.identity(dim_x + dim_h) / 0.0029102486697617723  

basis = GLE_BasisTransform(basis_type="linear")
Ntrajs = 0
run = 1
verbose = 2
paths = glob(datapath+"*_traj.dat")
maxlenght = None

if Ntrajs > 0:
    paths = paths[:Ntrajs]
Ntrajs = len(paths)
#print(Ntrajs)
#print(paths)
X, idx = datas_loaders.loadData(paths, dim_x, maxlenght=maxlenght)
#print (X.shape)
n = 0

xva_list = []
path = paths[0]
trj=np.loadtxt(str(path))

x = trj[:,1]
v = trj[:,2]

time = np.split(X, idx)[0][:, 0]

est = GLE_Estimator(model = 'obabo', init_params = "user", dim_x = dim_x, dim_h = dim_h, basis = basis, 
                    A_init = A , C_init = C , force_init = force, sig_init = C, mu_init = np.zeros((dim_x + dim_h)),  random_state = None, verbose = verbose, )
est.dt = time[1] - time[0]
est._check_initial_parameters()

Xproc, idx = est.model_class.preprocessingTraj(est.basis, X, idx_trajs=idx)

traj_list = np.split(Xproc, idx)
est.dim_coeffs_force = est.basis.nb_basis_elt_

datas = {}
print(traj_list[0])
#for n, traj in enumerate(traj_list):
    #datas_visible = sufficient_stats(traj, est.dim_x)
    #zero_sig = np.zeros((len(traj), 2 * est.dim_h, 2 * est.dim_h))
    #muh = np.hstack((np.roll(traj_list_h[n], -1, axis=0), traj_list_h[n]))
    #datas += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / len(traj_list)

est._initialize_parameters(None)
print(est.get_coefficients())
print("Real datas")
print(datas)
new_stat = {}
noise_corr = {}
for n, traj in enumerate(traj_list):
    datas_visible = est.model_class.sufficient_stats(traj, est.dim_x)
    muh , Sigh = est._e_step(traj)
    #muh , Sigh = filtersmoother
      # Compute hidden variable distribution
    adder(new_stat, est.model_class.sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h_kalman, est.dim_coeffs_force), len(traj_list))

print("Estimated datas")
print(new_stat)
print("Diff")
#print((new_stat - datas) / np.abs(datas))

force_m, friction_m, diffusion_m = est.model_class.m_step(None, None, None, new_stat, est.dim_h, est.dt, est.OptimizeDiffusion, est.OptimizeForce)

traj_list = np.split(Xproc, idx)

C_f = 0
BBT = 0
ABT = 0
for n, traj in enumerate(traj_list):
    bk = traj[:-1, 3: 4]
    bk_plus = traj[:-1, 4: 5]
    q  =  traj[:-1, 0: 1]
    q_plus = traj[:-1, 1: 2]
    v  =  traj[:-1, 2: 3]
    B = np.hstack( (est.dt * v, est.dt**2 / 2 * bk))
    #print ("dt =", est.dt)
    A = q_plus-q 
    #print("B.shape = ", B.shape)
    #print("A.shape = ", B.shape)
    BBT += np.mean(B[:, :, np.newaxis] * B[:, np.newaxis, :], axis = 0 )/len(traj_list)
    #print("BBT.shape =", BBT.shape)
    ABT += np.mean(A[:, :, np.newaxis] * B[:, np.newaxis, :], axis = 0 )/len(traj_list)

invBBT = np.linalg.inv(BBT)
C_f = np.matmul(ABT,invBBT)

print(C_f)

A = C_f[0,0]

force_coeffs = C_f[0,1]

print(A, force_coeffs)

print("Input friction = ", friction)

print("Input force_coeff = ", force)

print("Output friction = ", A)

print("Output force_coeff = ", force_coeffs)

print("M Output friction = ", force_m)

print("M Output force_coeff = ", friction_m)

print("M Output diffusion_coeff = ",  diffusion_m)

print(Sigh[:, 0, 0])
fig, axs = plt.subplots(1, dim_h + dim_x)
## plt.show()
axs.plot(time[:-4], muh[:-3, 0] , label="Prediction (with \\pm 2 \\sigma error lines)", color="blue")
axs.plot(time[:-4], muh[:-3, 0] + 2 * np.sqrt(Sigh[:-3, 0, 0]), "--", color="blue", linewidth=0.1)
axs.plot(time[:-4], muh[:-3, 0] - 2 * np.sqrt(Sigh[:-3, 0, 0]), "--", color="blue", linewidth=0.1)
axs.plot(time[:-4], Sigh[:-3, 0, 0], label="Sigma error", color="cyan")
#axs.plot(time[:-1], mutilde[:, 1] , label="Mutilde_v", color="red")
#axs.plot(time[:-1], mutilde[:, 0] , label="Mutilde_x", color="darkgreen")
axs.plot(time[:-4], v[:-2,0], label="Real_v", color="orange")
#axs.plot(time[:-1], x[:-1], label="Real_x", color="green")
axs.legend(loc="upper right")
plt.show()
