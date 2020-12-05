"""
This the main estimator module
"""
import numpy as np
import pandas as pd
import scipy.linalg


def preprocessingTraj(X, dt, dim_x):
    """
    From a flat array compute everythong that is needed for the following computation
    """
    X = check_array(X, ensure_min_features=4, allow_nd=True)
    traj_list = []
    for xv in X:
        x, v = np.hsplit(xv, 2)
        tps = dt * np.arange(x.shape[0])
        # v = (np.roll(x, -1, axis=0) - x) / dt
        # print(v)
        # xv_np = np.hstack((x, v))
        xhalf = xr.DataArray(x + 0.5 * dt * v, coords={"t": tps}, dims=["t", "space"])
        projmat = np.zeros((dim_x, 2 * dim_x))
        projmat[:dim_x, :dim_x] = 0.5 * dt / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)
        projmat[:dim_x, dim_x : 2 * dim_x] = 1.0 / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)

        P = projmat.copy()
        P[:dim_x, dim_x : 2 * dim_x] = (1 + ((0.5 * dt) ** 2 / (1 + (0.5 * dt) ** 2))) * np.identity(dim_x)
        xv_plus_proj = (np.matmul(projmat, np.roll(xv, -1, axis=0).T)).T
        xv_proj = np.matmul(P, xv.T).T

        xv = xr.Dataset({"xv_plus_proj": (["t", "dim_x"], xv_plus_proj), "xv_proj": (["t", "dim_x"], xv_proj), "v": (["t", "dim_x"], v)}, coords={"t": tps})
        xv.attrs["lenTraj"] = x.shape[0]
        traj_list.append(xv)
    return traj_list, xhalf  # TODO mettre xhalf sous la forme [nb_traj,nb_timestep, dim_x]


def sufficient_stats_aboba(traj, dim_x, dim_force):
    """
    Given a sample of trajectory, compute the averaged values of the sufficient statistics
    """

    diffs = traj["xv_plus_proj"].values - traj["xv_proj"].values

    x_val_proj = traj["v"].values

    xx = np.zeros((dim_x, dim_x))
    xdx = np.zeros_like(xx)
    dxdx = np.zeros_like(xx)
    bkx = np.zeros((dim_force, dim_x))
    bkdx = np.zeros_like(bkx)
    bkbk = np.zeros((dim_force, dim_force))

    bk = traj["bk"].data
    lenTraj = traj.attrs["lenTraj"]
    for i in range(lenTraj - 1):  # The -1 comes from the last values removed
        xx += np.outer(x_val_proj[i], x_val_proj[i])
        xdx += np.outer(x_val_proj[i], diffs[i])
        dxdx += np.outer(diffs[i], diffs[i])
        bkx += np.outer(bk[i], x_val_proj[i])
        bkdx += np.outer(bk[i], diffs[i])
        bkbk += np.outer(bk[i], bk[i])

    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": bkbk}) / (lenTraj - 1)


def sufficient_stats_hidden_aboba(muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force):
    """
    Compute the sufficient statistics averaged over the hidden variable distribution
    """

    dim_tot = dim_x + dim_h

    xx = np.zeros((dim_tot, dim_tot))
    xx[:dim_x, :dim_x] = old_stats["xx"]
    xdx = np.zeros_like(xx)
    xdx[:dim_x, :dim_x] = old_stats["xdx"]
    dxdx = np.zeros_like(xx)
    dxdx[:dim_x, :dim_x] = old_stats["dxdx"]
    bkx = np.zeros((dim_force, dim_tot))
    bkx[:, :dim_x] = old_stats["bkx"]
    bkdx = np.zeros_like(bkx)
    bkdx[:, :dim_x] = old_stats["bkdx"]

    bk = traj["bk"].data

    diffs_xv = traj["xv_plus_proj"].values - traj["xv_proj"].values

    x_val_proj = traj["v"].values

    lenTraj = traj.attrs["lenTraj"]

    for i in range(lenTraj - 1):  # The -1 comes from the last values removed
        # print(muh[i, dim_h:])
        xx[dim_x:, dim_x:] += Sigh[i, dim_h:, dim_h:] + np.outer(muh[i, dim_h:], muh[i, dim_h:])
        xx[dim_x:, :dim_x] += np.outer(muh[i, dim_h:], x_val_proj[i])

        xdx[dim_x:, dim_x:] += Sigh[i, dim_h:, :dim_h] + np.outer(muh[i, dim_h:], muh[i, :dim_h]) - Sigh[i, dim_h:, dim_h:] - np.outer(muh[i, dim_h:], muh[i, dim_h:])
        xdx[dim_x:, :dim_x] += np.outer(muh[i, dim_h:], diffs_xv[i])
        xdx[:dim_x, dim_x:] += np.outer(x_val_proj[i], muh[i, :dim_h] - muh[i, dim_h:])

        dxdx[dim_x:, dim_x:] += Sigh[i, :dim_h, :dim_h] + np.outer(muh[i, :dim_h], muh[i, :dim_h]) - 2 * Sigh[i, dim_h:, :dim_h] - np.outer(muh[i, dim_h:], muh[i, :dim_h]) - np.outer(muh[i, :dim_h], muh[i, dim_h:]) + Sigh[i, dim_h:, dim_h:] + np.outer(muh[i, dim_h:], muh[i, dim_h:])
        dxdx[dim_x:, :dim_x] += np.outer(muh[i, :dim_h] - muh[i, dim_h:], diffs_xv[i])

        bkx[:, dim_x:] += np.outer(bk[i], muh[i, dim_h:])
        bkdx[:, dim_x:] += np.outer(bk[i], muh[i, :dim_h] - muh[i, dim_h:])

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
    TODO
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
    """
    Xt = traj["xv_proj"].values
    Xtplus = traj["xv_plus_proj"].values
    Vt = traj["v"].values
    bkt = traj["bk"].values
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = 0.5 * dt * np.identity(dim_x)

    mutilde = (np.matmul(np.identity(dim_x + dim_h)[:, :dim_x], Xt.T - Vt.T) + np.matmul(expA[:, :dim_x], Vt.T) + np.matmul(expA + np.identity(dim_x + dim_h), np.matmul(Pf, bkt.T))).T

    return Xtplus, mutilde


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
