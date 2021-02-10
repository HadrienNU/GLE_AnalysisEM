"""
This the main estimator module
"""
import numpy as np
import scipy.linalg


def projA(expA, dim_x, dim_h, dt):
    """
    From full matrix project onto correct model
    """
    # print((-scipy.linalg.logm(expA) / dt))
    A = -scipy.linalg.logm(expA) / dt
    A[dim_x:, :dim_x] = 0
    A[:dim_x, dim_x:] = 0
    min_dim = min(dim_x, dim_h)
    A[dim_x : dim_x + min_dim, :min_dim] = -np.eye(min_dim)
    A[:min_dim, dim_x : dim_x + min_dim] = np.eye(min_dim)
    return scipy.linalg.expm(-1 * dt * A)


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
    return np.hstack((xv_plus_proj, xv_proj, v, bk)), idx_trajs


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
    """Value of the ml"""
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

    return traj[:, :dim_x], mutilde, expA[:, dim_x:]


def m_step_aboba(sufficient_stat, dim_x, dim_h, dt, EnforceFDT, OptimizeDiffusion, OptimizeForce):
    """M step.
    TODO:   -Select dimension of fitted parameters from the sufficient stats
    """
    Id = np.identity(dim_x + dim_h)

    invbkbk = np.linalg.inv(sufficient_stat["bkbk"])
    YX = sufficient_stat["xdx"].T - np.matmul(sufficient_stat["bkdx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
    XX = sufficient_stat["xx"] - np.matmul(sufficient_stat["bkx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
    expA = Id + np.matmul(YX, np.linalg.inv(XX))

    # expA = projA(expA, dim_x, dim_h, dt)

    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = 0.5 * dt * np.identity(dim_x)

    X = np.matmul(expA + Id, Pf)
    # print(np.matmul(sufficient_stat["bkdx"].T, invbkbk))
    force_coeffs = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, np.matmul(sufficient_stat["bkdx"].T, invbkbk) - np.matmul(expA - Id, np.matmul(sufficient_stat["bkx"].T, invbkbk))))

    if OptimizeDiffusion:  # Optimize Diffusion based on the variance of the sufficients statistics
        bkbk = np.matmul(Pf, np.matmul(np.matmul(force_coeffs, np.matmul(sufficient_stat["bkbk"], force_coeffs.T)), Pf.T))
        bkdx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkdx"]))
        bkx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkx"]))

        residuals = sufficient_stat["dxdx"] - np.matmul(expA - Id, sufficient_stat["xdx"]) - np.matmul(expA - Id, sufficient_stat["xdx"]).T - np.matmul(expA + Id, bkdx).T - np.matmul(expA + Id, bkdx)
        residuals += np.matmul(expA - Id, np.matmul(sufficient_stat["xx"], (expA - Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)).T + np.matmul(expA + Id, np.matmul(bkbk, (expA + Id).T))
        SST = 0.5 * (residuals + residuals.T)

    else:
        SST = 1.0
    # if EnforceFDT:  # In case we want the FDT the starting seed is the computation without FDT
    #     theta0 = friction_coeffs.ravel()  # Starting point of the scipy root algorithm
    #     theta0 = np.hstack((theta0, (dim_x + dim_h) / np.trace(np.matmul(np.linalg.inv(diffusion_coeffs), (Id - np.matmul(friction_coeffs, friction_coeffs.T))))))
    #
    #     # To find the better value of the parameters based on the means values
    #     # sol = scipy.optimize.root(mle_derivative_expA_FDT, theta0, args=(sufficient_stat["dxdx"], sufficient_stat["xdx"], sufficient_stat["xx"], bkbk, bkdx, bkx, dim_x + dim_h), method="lm")
    #     # cons = scipy.optimize.NonlinearConstraint(detConstraints, 1e-10, np.inf)
    #     sol = scipy.optimize.minimize(mle_FDT, theta0, args=(sufficient_stat["dxdx"], sufficient_stat["xdx"], sufficient_stat["xx"], bkbk, bkdx, bkx, dim_x + dim_h), method="Nelder-Mead")
    #     if not sol.success:
    #         warnings.warn("M step did not converge" "{}".format(sol), ConvergenceWarning)
    #     friction_coeffs = sol.x[:-1].reshape((dim_x + dim_h, dim_x + dim_h))
    #     diffusion_coeffs = sol.x[-1] * (Id - np.matmul(friction_coeffs, friction_coeffs.T))

    return expA, force_coeffs, SST


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
    return quad_part - 0.5 * logdet


def ABOBA_generator(nsteps=50, dt=5e-3, dim_x=1, dim_h=1, x0=None, v0=None, expA=None, SST=None, force_coeffs=None, muh0=0.0, sigh0=0.0, basis=None, rng=np.random.default_rng()):
    """
    Integrate the equation of nsteps steps
    """
    if x0 is None:
        x0 = np.zeros((dim_x,))
    if v0 is None:
        v0 = np.zeros((dim_x,))
    x_traj = np.empty((nsteps, dim_x))
    p_traj = np.empty((nsteps, dim_x))
    h_traj = np.empty((nsteps, dim_h))
    t_traj = np.reshape(np.arange(0.0, nsteps) * dt, (-1, 1))
    x_traj[0, :] = x0
    p_traj[0, :] = v0
    h_traj[0, :] = rng.multivariate_normal(muh0, sigh0)
    S = np.linalg.cholesky(SST)
    for n in range(1, nsteps):
        xhalf = x_traj[n - 1, :] + 0.5 * dt * p_traj[n - 1, :]
        force_t = np.matmul(force_coeffs, basis.predict(np.reshape(xhalf, (1, -1)))[0])  # The [0] because predict return an n_timestep*n_features array
        phalf = p_traj[n - 1, :] + 0.5 * dt * force_t
        gauss = np.matmul(S, rng.standard_normal(size=dim_x + dim_h))
        # gaussp, gaussh = np.split(rng.multivariate_normal(np.zeros((dim_x + dim_h,)), SST), [dim_x])

        phalfprime = np.matmul(expA[0:dim_x, 0:dim_x], phalf) + np.matmul(expA[0:dim_x, dim_x:], h_traj[n - 1, :]) + gauss[:dim_x]
        h_traj[n, :] = np.matmul(expA[dim_x:, 0:dim_x], phalf) + np.matmul(expA[dim_x:, dim_x:], h_traj[n - 1, :]) + gauss[dim_x:]

        p_traj[n, :] = phalfprime + 0.5 * dt * force_t
        x_traj[n, :] = xhalf + 0.5 * dt * p_traj[n, :]
    return np.hstack((t_traj, x_traj, p_traj)), h_traj
