"""
This the main estimator module
"""
import numpy as np
import pandas as pd
import scipy.linalg


def preprocessingTraj_aboba(X, idx_trajs=[], dim_x=1):
    """
    From position and velocity array compute everything that is needed for the following computation
    """
    dt = X[1, 0] - X[0, 0]

    projmat = np.zeros((dim_x, 2 * dim_x))
    projmat[:dim_x, :dim_x] = 0.5 * dt / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)
    projmat[:dim_x, dim_x : 2 * dim_x] = 1.0 / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)
    P = projmat.copy()
    P[:dim_x, dim_x : 2 * dim_x] = (1 + ((0.5 * dt) ** 2 / (1 + (0.5 * dt) ** 2))) * np.identity(dim_x)

    xv_plus_proj = (np.matmul(projmat, np.roll(X[:, 1 : 1 + 2 * dim_x], -1, axis=0).T)).T
    xv_proj = np.matmul(P, X[:, 1 : 1 + 2 * dim_x].T).T
    v = X[:, 1 + dim_x : 1 + 2 * dim_x]
    bk = X[:, 1 + 2 * dim_x :]
    return np.hstack((xv_plus_proj, xv_proj, v, bk))


def sufficient_stats_aboba(traj, dim_x):
    """
    Given a sample of trajectory, compute the averaged values of the sufficient statistics
    Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
    """

    xval = traj[:, 2 * dim_x : 3 * dim_x]
    dx = traj[:, :dim_x] - traj[:, dim_x : 2 * dim_x]
    bk = traj[:, 3 * dim_x :]
    xx = np.mean(xval[:-1, :, np.newaxis] * xval[:-1, np.newaxis, :], axis=0)
    xdx = np.mean(xval[:-1, :, np.newaxis] * dx[:-1, np.newaxis, :], axis=0)
    dxdx = np.mean(dx[:-1, :, np.newaxis] * dx[:-1, np.newaxis, :], axis=0)
    bkx = np.mean(bk[:-1, :, np.newaxis] * xval[:-1, np.newaxis, :], axis=0)
    bkdx = np.mean(bk[:-1, :, np.newaxis] * dx[:-1, np.newaxis, :], axis=0)
    bkbk = np.mean(bk[:-1, :, np.newaxis] * bk[:-1, np.newaxis, :], axis=0)

    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": bkbk})  # / (lenTraj - 1)


def sufficient_stats_hidden_aboba(muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force):
    """
    Compute the sufficient statistics averaged over the hidden variable distribution
    Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
    """

    xx = np.zeros((dim_x + dim_h, dim_x + dim_h))
    xx[:dim_x, :dim_x] = old_stats["xx"]
    xdx = np.zeros_like(xx)
    xdx[:dim_x, :dim_x] = old_stats["xdx"]
    dxdx = np.zeros_like(xx)
    dxdx[:dim_x, :dim_x] = old_stats["dxdx"]
    bkx = np.zeros((dim_force, dim_x + dim_h))
    bkx[:, :dim_x] = old_stats["bkx"]
    bkdx = np.zeros_like(bkx)
    bkdx[:, :dim_x] = old_stats["bkdx"]

    xval = traj[:, 2 * dim_x : 3 * dim_x]
    dx = traj[:, :dim_x] - traj[:, dim_x : 2 * dim_x]
    bk = traj[:, 3 * dim_x :]

    dh = muh[:, :dim_h] - muh[:, dim_h:]

    Sigh_tptp = np.mean(Sigh[:-1, :dim_h, :dim_h], axis=0)
    Sigh_ttp = np.mean(Sigh[:-1, dim_h:, :dim_h], axis=0)
    Sigh_tt = np.mean(Sigh[:-1, dim_h:, dim_h:], axis=0)

    muh_tptp = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
    muh_ttp = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
    muh_tpt = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
    muh_tt = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)

    xx[dim_x:, dim_x:] = Sigh_tt + muh_tt
    xx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * xval[:-1, np.newaxis, :], axis=0)

    xdx[dim_x:, dim_x:] = Sigh_ttp + muh_ttp - Sigh_tt - muh_tt
    xdx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * dx[:-1, np.newaxis, :], axis=0)
    xdx[:dim_x, dim_x:] = np.mean(xval[:-1, :, np.newaxis] * dh[:-1, np.newaxis, :], axis=0)

    dxdx[dim_x:, dim_x:] = Sigh_tptp + muh_tptp - 2 * Sigh_ttp - muh_ttp - muh_tpt + Sigh_tt + muh_tt
    dxdx[dim_x:, :dim_x] = np.mean(dh[:-1, :, np.newaxis] * dx[:-1, np.newaxis, :], axis=0)

    bkx[:, dim_x:] = np.mean(bk[:-1, :, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
    bkdx[:, dim_x:] = np.mean(bk[:-1, :, np.newaxis] * dh[:-1, np.newaxis, :], axis=0)

    xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
    dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"], "µ_0": muh[0, dim_h:], "Σ_0": Sigh[0, dim_h:, dim_h:]})


def mle_derivative_expA_FDT(theta, dxdx, xdx, xx, bkbk, bkdx, bkx, dim_tot):
    """
    Compute the value of the derivative with respect to expA only for the term related to the FDT (i.e. Sigma)
    """
    expA = theta[:-1].reshape((dim_tot, dim_tot))
    kbT = theta[-1]
    deriv_expA = np.zeros_like(theta)
    # k is the chosen derivative
    YY = dxdx - 2 * (bkdx + bkdx.T) + 4 * bkbk
    YX = xdx.T - 2 * bkx + bkdx.T - 2 * bkbk
    XX = xx + bkx + bkx.T + bkbk
    Id = np.identity(dim_tot)
    invSSTexpA = np.linalg.inv(Id - np.matmul(expA, expA.T)) / kbT
    combYX = YY + np.matmul(expA - Id, np.matmul(XX, expA.T - Id)) - np.matmul(YX, expA.T - Id) - np.matmul(YX, expA.T - Id).T

    for k in range(dim_tot ** 2):
        DexpA_flat = np.zeros((dim_tot ** 2,))
        DexpA_flat[k] = 1.0
        DexpA = DexpA_flat.reshape((dim_tot, dim_tot))
        deriv_expA[k] = 2 * np.trace(np.matmul(invSSTexpA, np.matmul(np.matmul(expA, Id - np.matmul(combYX, invSSTexpA)), DexpA.T)))
        deriv_expA[k] += np.trace(np.matmul(invSSTexpA, np.matmul(YX - np.matmul(expA - Id, XX), DexpA.T)))
    deriv_expA[-1] = dim_tot / kbT - np.trace(np.matmul(combYX, invSSTexpA)) / kbT
    # print(deriv_expA)
    return deriv_expA


def mle_FDT(theta, dxdx, xdx, xx, bkbk, bkdx, bkx, dim_tot):
    """Value of the ml
    """
    expA = theta[:-1].reshape((dim_tot, dim_tot))
    kbT = theta[-1]
    # k is the chosen derivative
    YY = dxdx - 2 * (bkdx + bkdx.T) + 4 * bkbk
    YX = xdx.T - 2 * bkx + bkdx.T - 2 * bkbk
    XX = xx + bkx + bkx.T + bkbk
    Id = np.identity(dim_tot)
    invSSTexpA = np.linalg.inv(Id - np.matmul(expA, expA.T)) / kbT
    # print(theta, 1 / np.linalg.det(invSSTexpA))
    combYX = YY + np.matmul(expA - Id, np.matmul(XX, expA.T - Id)) - np.matmul(YX, expA.T - Id) - np.matmul(YX, expA.T - Id).T
    return np.trace(np.matmul(combYX, invSSTexpA)) - np.log(np.linalg.det(invSSTexpA))


def compute_expectation_estep_aboba(traj, expA, force_coeffs, dim_x, dim_h, dt):
    """
    Compute the value of mutilde and Xtplus
    Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
    """
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = 0.5 * dt * np.identity(dim_x)
    mutilde = (
        np.matmul(np.identity(dim_x + dim_h)[:, :dim_x], traj[:, dim_x : 2 * dim_x].T - traj[:, 2 * dim_x : 3 * dim_x].T) + np.matmul(expA[:, :dim_x], traj[:, 2 * dim_x : 3 * dim_x].T) + np.matmul(expA + np.identity(dim_x + dim_h), np.matmul(Pf, np.matmul(force_coeffs, traj[:, 3 * dim_x :].T)))
    ).T

    return traj[:, :dim_x], mutilde


def m_step_aboba(sufficient_stat, expA, SST, coeffs_force, dim_x, EnforceFDT, OptimizeDiffusion, dim_h, dt):
    """M step.
    TODO:   -Select dimension of fitted parameters from the sufficient stats
            -Allow to select statistical model (Euler/ ABOBA)
    """
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = 0.5 * dt * np.identity(dim_x)

    bkbk = np.matmul(Pf, np.matmul(np.matmul(coeffs_force, np.matmul(sufficient_stat["bkbk"], coeffs_force.T)), Pf.T))
    bkdx = np.matmul(Pf, np.matmul(coeffs_force, sufficient_stat["bkdx"]))
    bkx = np.matmul(Pf, np.matmul(coeffs_force, sufficient_stat["bkx"]))
    Id = np.identity(dim_x + dim_h)
    if not EnforceFDT:

        YX = sufficient_stat["xdx"].T - 2 * bkx + bkdx.T - 2 * bkbk
        XX = sufficient_stat["xx"] + bkx + bkx.T + bkbk
        expA = Id + np.matmul(YX, np.linalg.inv(XX))
    else:
        theta0 = expA.ravel()  # Starting point of the scipy root algorithm
        # To find the better value of the parameters based on the means values
        sol = scipy.optimize.root(mle_derivative_expA_FDT, theta0, args=(sufficient_stat["dxdx"], sufficient_stat["xdx"], sufficient_stat["xx"], bkbk, bkdx, bkx, np.linalg.inv(SST), dim_x + dim_h), method="hybr")
        if not sol.success:
            print(sol)
            raise ValueError("M step did not converge")
        expA = sol.x.reshape((dim_x + dim_h, dim_x + dim_h))

    # Optimize based on  the variance of the sufficients statistics
    if OptimizeDiffusion:
        residuals = sufficient_stat["dxdx"] - np.matmul(expA - Id, sufficient_stat["xdx"]) - np.matmul(expA - Id, sufficient_stat["xdx"]).T - np.matmul(expA + Id, bkdx) - np.matmul(expA + Id, bkdx).T
        residuals += np.matmul(expA - Id, np.matmul(sufficient_stat["xx"], (expA - Id).T)) + np.matmul(expA + Id, np.matmul(bkx, (expA - Id).T)) + np.matmul(expA + Id, np.matmul(bkx, (expA - Id).T)).T
        residuals += np.matmul(expA + Id, np.matmul(bkbk, (expA + Id).T))
        if EnforceFDT:  # In which case we only optimize the temperature
            kbT = (dim_x + dim_h) / np.trace(np.matmul(np.linalg.inv(SST), residuals))  # Update the temperature
            SST = kbT * (Id - np.matmul(expA, expA.T))
        else:  # In which case we optimize the full diffusion matrix
            SST = residuals


def loglikelihood_aboba(suff_datas, expA, SST, coeffs_force, dim_x, dim_h, dt):
    """
    Return the current value of the log-likelihood
    """
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = 0.5 * dt * np.identity(dim_x)

    bkbk = np.matmul(Pf, np.matmul(np.matmul(coeffs_force, np.matmul(suff_datas["bkbk"], coeffs_force.T)), Pf.T))
    bkdx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkdx"]))
    bkx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkx"]))

    Id = np.identity(dim_x + dim_h)
    m1 = suff_datas["dxdx"] - np.matmul(expA - Id, suff_datas["xdx"]) - np.matmul(expA - Id, suff_datas["xdx"]).T - np.matmul(expA + Id, bkdx).T - np.matmul(expA + Id, bkdx)
    m1 += np.matmul(expA - Id, np.matmul(suff_datas["xx"], (expA - Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)).T + np.matmul(expA + Id, np.matmul(bkbk, (expA + Id).T))

    logdet = (dim_x + dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(SST))
    quad_part = -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1))
    return quad_part - 0.5 * logdet, quad_part


def ABOBA_generator(nsteps=50, dt=5e-3, dim_x=1, dim_h=1, x0=0.0, v0=0.0, expA=None, SST=None, force_coeffs=None, muh0=0.0, sigh0=0.0, basis=None, rng=np.random.default_rng()):
    """
    Integrate the equation of nsteps steps
    """

    x_traj = np.empty((nsteps, dim_x))
    p_traj = np.empty((nsteps, dim_x))
    h_traj = np.empty((nsteps, dim_h))
    t_traj = np.reshape(np.arange(0.0, nsteps) * dt, (-1, 1))
    x_traj[0, :] = x0
    p_traj[0, :] = v0
    h_traj[0, :] = rng.multivariate_normal(muh0, sigh0)

    for n in range(1, nsteps):
        xhalf = x_traj[n - 1, :] + 0.5 * dt * p_traj[n - 1, :]
        force_t = np.reshape(np.matmul(force_coeffs, basis.predict(np.reshape(xhalf, (1, -1)))), (dim_x,))
        phalf = p_traj[n - 1, :] + 0.5 * dt * force_t

        gaussp, gaussh = np.split(rng.multivariate_normal(np.zeros((dim_x + dim_h,)), SST), [dim_x])
        phalfprime = np.matmul(expA[0:dim_x, 0:dim_x], phalf) + np.matmul(expA[0:dim_x, dim_x:], h_traj[n - 1, :]) + gaussp
        h_traj[n, :] = np.matmul(expA[dim_x:, 0:dim_x], phalf) + np.matmul(expA[dim_x:, dim_x:], h_traj[n - 1, :]) + gaussh

        p_traj[n, :] = phalfprime + 0.5 * dt * force_t
        x_traj[n, :] = xhalf + 0.5 * dt * p_traj[n, :]
    return np.hstack((t_traj, x_traj, p_traj)), h_traj
