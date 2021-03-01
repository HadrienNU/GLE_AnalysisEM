"""
This the main estimator module
"""
import numpy as np


def expected_features(dim_x):
    return 1 + 2 * dim_x


def compute_expectation_estep_euler_nl(traj, A, force_coeffs, dim_x, dim_h, dt):
    """
    Compute the value of mutilde and Xtplus
    Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
    """
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = dt * np.identity(dim_x)
    mutilde = (np.matmul(-A[:, :dim_x], traj[:, 2 * dim_x : 3 * dim_x].T) + np.matmul(Pf, np.matmul(force_coeffs, traj[:, 3 * dim_x :].T))).T
    mutilde += np.matmul(np.identity(dim_x + dim_h)[:, :dim_x], traj[:, dim_x : 2 * dim_x].T).T  # mutilde is X_t+f(X_t) - A*X_t
    return traj[:, :0], mutilde[:, dim_x:], np.identity(dim_h) - A[dim_x:, dim_x:]


def m_step_euler_nl(sufficient_stat, dim_x, dim_h, dt, EnforceFDT, OptimizeDiffusion, OptimizeForce):
    """M step.
    TODO:   -Select dimension of fitted parameters from the sufficient stats
    """

    invbkbk = np.linalg.inv(sufficient_stat["bkbk"])
    YX = sufficient_stat["xdx"].T - np.matmul(sufficient_stat["bkdx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
    XX = sufficient_stat["xx"] - np.matmul(sufficient_stat["bkx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
    A = -np.matmul(YX, np.linalg.inv(XX))

    force_coeffs = (np.matmul(sufficient_stat["bkdx"].T, invbkbk) / dt - np.matmul(A, np.matmul(sufficient_stat["bkx"].T, invbkbk)) / dt)[:dim_x, :]

    if OptimizeDiffusion:  # Optimize Diffusion based on the variance of the sufficients statistics
        Pf = np.zeros((dim_x + dim_h, dim_x))
        Pf[:dim_x, :dim_x] = dt * np.identity(dim_x)

        bkbk = np.matmul(Pf, np.matmul(np.matmul(force_coeffs, np.matmul(sufficient_stat["bkbk"], force_coeffs.T)), Pf.T))
        bkdx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkdx"]))
        bkx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkx"]))

        residuals = sufficient_stat["dxdx"] + np.matmul(A, sufficient_stat["xdx"]) + np.matmul(A, sufficient_stat["xdx"]).T - bkdx.T - bkdx
        residuals += np.matmul(A, np.matmul(sufficient_stat["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk
        SST = 0.5 * (residuals + residuals.T)[dim_x:, dim_x:]
    else:
        SST = 1
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

    return A, force_coeffs, SST


def loglikelihood_euler_nl(suff_datas, A, SST, coeffs_force, dim_x, dim_h, dt):
    """
    Return the current value of the log-likelihood
    """
    Pf = np.zeros((dim_x + dim_h, dim_x))
    Pf[:dim_x, :dim_x] = dt * np.identity(dim_x)

    bkbk = np.matmul(Pf, np.matmul(np.matmul(coeffs_force, np.matmul(suff_datas["bkbk"], coeffs_force.T)), Pf.T))
    bkdx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkdx"]))
    bkx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkx"]))

    m1 = suff_datas["dxdx"] + np.matmul(A, suff_datas["xdx"]) + np.matmul(A, suff_datas["xdx"]).T - bkdx.T - bkdx
    m1 += np.matmul(A, np.matmul(suff_datas["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk

    logdet = dim_h * np.log(2 * np.pi) + np.log(np.linalg.det(SST))
    quad_part = -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1[dim_x:, dim_x:]))
    return quad_part - 0.5 * logdet


def euler_generator_nl(nsteps=50, dt=5e-3, dim_x=1, dim_h=1, x0=0.0, v0=0.0, friction=None, SST=None, force_coeffs=None, muh0=0.0, sigh0=0.0, basis=None, rng=np.random.default_rng()):
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
        x_traj[n, :] = x_traj[n - 1, :] + dt * p_traj[n - 1, :]
        force_t = dt * np.matmul(force_coeffs, basis.predict(np.reshape(x_traj[n - 1, :], (1, -1)))[0])
        # gaussh = rng.multivariate_normal(np.zeros((dim_h,)), SST)
        gaussh = np.matmul(S, rng.standard_normal(size=dim_h))
        h_traj[n, :] = h_traj[n - 1, :] - np.matmul(friction[dim_x:, :dim_x], p_traj[n - 1, :]) - np.matmul(friction[dim_x:, dim_x:], h_traj[n - 1, :]) + gaussh
        p_traj[n, :] = p_traj[n - 1, :] - np.matmul(friction[:dim_x, :dim_x], p_traj[n - 1, :]) - np.matmul(friction[:dim_x, dim_x:], h_traj[n - 1, :]) + force_t
    return np.hstack((t_traj, x_traj, p_traj)), h_traj
