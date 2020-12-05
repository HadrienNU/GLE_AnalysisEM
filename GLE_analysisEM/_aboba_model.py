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


def sufficient_stats_aboba(traj, dim_x, dim_force):
    """
    Given a sample of trajectory, compute the averaged values of the sufficient statistics
    Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
    """

    xx = np.zeros((dim_x, dim_x))
    xdx = np.zeros_like(xx)
    dxdx = np.zeros_like(xx)
    bkx = np.zeros((dim_force, dim_x))
    bkdx = np.zeros_like(bkx)
    bkbk = np.zeros((dim_force, dim_force))

    lenTraj = len(traj)
    for i in range(lenTraj - 1):  # The -1 comes from the last values removed
        xx += np.outer(traj[i, 2 * dim_x : 3 * dim_x], traj[i, 2 * dim_x : 3 * dim_x])
        xdx += np.outer(traj[i, 2 * dim_x : 3 * dim_x], traj[i, :dim_x] - traj[i, dim_x : 2 * dim_x])
        dxdx += np.outer(traj[i, :dim_x] - traj[i, dim_x : 2 * dim_x], traj[i, :dim_x] - traj[i, dim_x : 2 * dim_x])
        bkx += np.outer(traj[i, 3 * dim_x :], traj[i, 2 * dim_x : 3 * dim_x])
        bkdx += np.outer(traj[i, 3 * dim_x :], traj[i, :dim_x] - traj[i, dim_x : 2 * dim_x])
        bkbk += np.outer(traj[i, 3 * dim_x :], traj[i, 3 * dim_x :])

    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": bkbk}) / (lenTraj - 1)


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

    lenTraj = len(traj)
    for i in range(lenTraj - 1):  # The -1 comes from the last values removed
        # print(muh[i, dim_h:])
        xx[dim_x:, dim_x:] += Sigh[i, dim_h:, dim_h:] + np.outer(muh[i, dim_h:], muh[i, dim_h:])
        xx[dim_x:, :dim_x] += np.outer(muh[i, dim_h:], traj[i, 2 * dim_x : 3 * dim_x])

        xdx[dim_x:, dim_x:] += Sigh[i, dim_h:, :dim_h] + np.outer(muh[i, dim_h:], muh[i, :dim_h]) - Sigh[i, dim_h:, dim_h:] - np.outer(muh[i, dim_h:], muh[i, dim_h:])
        xdx[dim_x:, :dim_x] += np.outer(muh[i, dim_h:], traj[i, :dim_x] - traj[i, dim_x : 2 * dim_x])
        xdx[:dim_x, dim_x:] += np.outer(traj[i, 2 * dim_x : 3 * dim_x], muh[i, :dim_h] - muh[i, dim_h:])

        dxdx[dim_x:, dim_x:] += Sigh[i, :dim_h, :dim_h] + np.outer(muh[i, :dim_h], muh[i, :dim_h]) - 2 * Sigh[i, dim_h:, :dim_h] - np.outer(muh[i, dim_h:], muh[i, :dim_h]) - np.outer(muh[i, :dim_h], muh[i, dim_h:]) + Sigh[i, dim_h:, dim_h:] + np.outer(muh[i, dim_h:], muh[i, dim_h:])
        dxdx[dim_x:, :dim_x] += np.outer(muh[i, :dim_h] - muh[i, dim_h:], traj[i, :dim_x] - traj[i, dim_x : 2 * dim_x])

        bkx[:, dim_x:] += np.outer(traj[i, 3 * dim_x :], muh[i, dim_h:])
        bkdx[:, dim_x:] += np.outer(traj[i, 3 * dim_x :], muh[i, :dim_h] - muh[i, dim_h:])

    # Normalisation
    xx[dim_x:, :] /= lenTraj - 1

    xdx[dim_x:, :] /= lenTraj - 1
    xdx[:dim_x, dim_x:] /= lenTraj - 1

    dxdx[dim_x:, :] /= lenTraj - 1

    bkx[:, dim_x:] /= lenTraj - 1
    bkdx[:, dim_x:] /= lenTraj - 1

    xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
    dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"], "µ_0": muh[0, dim_h:], "Σ_0": Sigh[0, dim_h:, dim_h:]})


def mle_derivative_expA_FDT(theta, dxdx, xdx, xx, bkbk, bkdx, bkx, invSST, dim_tot):
    """
    Compute the value of the derivative with respect to expA only for the term related to the FDT (i.e. Sigma)
    """
    expA = theta.reshape((dim_tot, dim_tot))
    deriv_expA = np.zeros_like(theta)
    # k is the chosen derivative
    YY = dxdx - 2 * (bkdx + bkdx.T) + 4 * bkbk
    YX = xdx.T - 2 * bkx + bkdx.T - 2 * bkbk
    XX = xx + bkx + bkx.T + bkbk
    Id = np.identity(dim_tot)
    invSSTexpA = np.linalg.inv(Id - np.matmul(expA, expA.T))
    combYX = YY + np.matmul(expA - Id, np.matmul(XX, expA.T - Id)) - np.matmul(YX, expA.T - Id) - np.matmul(YX, expA.T - Id).T

    for k in range(dim_tot ** 2):
        DexpA_flat = np.zeros((dim_tot ** 2,))
        DexpA_flat[k] = 1.0
        DexpA = DexpA_flat.reshape((dim_tot, dim_tot))
        deriv_expA[k] = 2 * np.trace(np.matmul(invSST, np.matmul(np.matmul(expA, Id - np.matmul(combYX, invSSTexpA)), DexpA.T)))
        deriv_expA[k] += np.trace(np.matmul(invSST, np.matmul(YX - np.matmul(expA - Id, XX), DexpA.T)))
    return deriv_expA


def compute_expectation_estep_aboba(traj, expA, dim_x, dim_h, dt):
    """
    Compute the value of mutilde and Xtplus
    Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
    """
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = 0.5 * dt * np.identity(dim_x)

    mutilde = (np.matmul(np.identity(dim_x + dim_h)[:, :dim_x], traj[:, dim_x : 2 * dim_x].T - traj[:, 2 * dim_x : 3 * dim_x].T) + np.matmul(expA[:, :dim_x], traj[:, 2 * dim_x : 3 * dim_x].T) + np.matmul(expA + np.identity(dim_x + dim_h), np.matmul(Pf, traj[:, 3 * dim_x :].T))).T

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


def loglikelihood_aboba(suff_datas, expA, SST, coeffs_force, dim_x, dim_h, dt, OptimizeDiffusion):
    """
    Return the current value of the negative log-likelihood
    """
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = 0.5 * dt * np.identity(dim_x)

    bkbk = np.matmul(Pf, np.matmul(np.matmul(coeffs_force, np.matmul(suff_datas["bkbk"], coeffs_force.T)), Pf.T))
    bkdx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkdx"]))
    bkx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkx"]))

    Id = np.identity(dim_x + dim_h)
    m1 = suff_datas["dxdx"] - np.matmul(expA - Id, suff_datas["xdx"]) - np.matmul(expA - Id, suff_datas["xdx"]).T - np.matmul(expA + Id, bkdx).T - np.matmul(expA + Id, bkdx)
    m1 += np.matmul(expA - Id, np.matmul(suff_datas["xx"], (expA - Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)).T + np.matmul(expA + Id, np.matmul(bkbk, (expA + Id).T))
    if OptimizeDiffusion:
        logdet = (dim_x + dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(SST))
        return -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1)) - 0.5 * logdet
    else:
        return -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1))
