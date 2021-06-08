"""
This the main estimator module
"""
import numpy as np
import scipy.linalg
import warnings
from sklearn.exceptions import ConvergenceWarning

from ._model_class import AbstractModel


class ABOBAModel(AbstractModel):
    def _convert_user_coefficients(self, A, C, dt):
        """
        Convert the user provided coefficients into the local one
        """
        friction = scipy.linalg.expm(-1 * dt * A)
        diffusion = C - np.matmul(friction, np.matmul(C, friction.T))

        return friction, diffusion

    def _convert_local_coefficients(self, friction_coeffs, diffusion_coeffs, dt):
        """
        Convert the estimator coefficients into the user one
        """
        if not np.isfinite(np.sum(friction_coeffs)) or not np.isfinite(np.sum(diffusion_coeffs)):  # Check for NaN value
            warnings.warn("NaN of infinite value in friction or diffusion coefficients.")
            return friction_coeffs, diffusion_coeffs

        A = -scipy.linalg.logm(friction_coeffs) / dt
        C = scipy.linalg.solve_discrete_lyapunov(friction_coeffs, diffusion_coeffs)

        return A, C

    def preprocessingTraj(self, basis, X, idx_trajs=[]):
        """
        From position and velocity array compute everything that is needed for the following computation
        """
        dt = X[1, 0] - X[0, 0]

        projmat = np.zeros((self.dim_x, 2 * self.dim_x))
        projmat[: self.dim_x, : self.dim_x] = 0.5 * dt / (1 + (0.5 * dt) ** 2) * np.identity(self.dim_x)
        projmat[: self.dim_x, self.dim_x : 2 * self.dim_x] = 1.0 / (1 + (0.5 * dt) ** 2) * np.identity(self.dim_x)
        P = projmat.copy()
        P[: self.dim_x, self.dim_x : 2 * self.dim_x] = (1 + ((0.5 * dt) ** 2 / (1 + (0.5 * dt) ** 2))) * np.identity(self.dim_x)

        xv_plus_proj = (np.matmul(projmat, np.roll(X[:, 1 : 1 + 2 * self.dim_x], -1, axis=0).T)).T
        xv_proj = np.matmul(P, X[:, 1 : 1 + 2 * self.dim_x].T).T
        v = X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        bk = basis.fit_transform(X[:, 1 : 1 + self.dim_x] + 0.5 * dt * v)
        return np.hstack((xv_plus_proj, xv_proj, v, bk)), idx_trajs

    def compute_expectation_estep(self, traj, expA, force_coeffs, dim_h, dt):
        """
        Compute the value of mutilde and Xtplus
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """
        Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = 0.5 * dt * np.identity(self.dim_x)
        mutilde = (
            np.matmul(np.identity(self.dim_x + dim_h)[:, : self.dim_x], traj[:, self.dim_x : 2 * self.dim_x].T - traj[:, 2 * self.dim_x : 3 * self.dim_x].T)
            + np.matmul(expA[:, : self.dim_x], traj[:, 2 * self.dim_x : 3 * self.dim_x].T)
            + np.matmul(expA + np.identity(self.dim_x + dim_h), np.matmul(Pf, np.matmul(force_coeffs, traj[:, 3 * self.dim_x :].T)))
        ).T

        return traj[:, : self.dim_x], mutilde, expA[:, self.dim_x :]

    def m_step(self, expA_old, SST_old, coeffs_force_old, sufficient_stat, dim_h, dt, OptimizeDiffusion, OptimizeForce):
        """M step.
        TODO:   -Select dimension of fitted parameters from the sufficient stats
        """
        Id = np.identity(self.dim_x + dim_h)

        invbkbk = np.linalg.inv(sufficient_stat["bkbk"])
        YX = sufficient_stat["xdx"].T - np.matmul(sufficient_stat["bkdx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
        XX = sufficient_stat["xx"] - np.matmul(sufficient_stat["bkx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
        expA = Id + np.matmul(YX, np.linalg.inv(XX))

        # expA = projA(expA, self.dim_x, dim_h, dt)

        Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = 0.5 * dt * np.identity(self.dim_x)

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

        return expA, force_coeffs, SST

    def loglikelihood(self, suff_datas, expA, SST, coeffs_force, dim_h, dt):
        """
        Return the current value of the log-likelihood
        """
        Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = 0.5 * dt * np.identity(self.dim_x)

        bkbk = np.matmul(Pf, np.matmul(np.matmul(coeffs_force, np.matmul(suff_datas["bkbk"], coeffs_force.T)), Pf.T))
        bkdx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkdx"]))
        bkx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkx"]))

        Id = np.identity(self.dim_x + dim_h)
        m1 = suff_datas["dxdx"] - np.matmul(expA - Id, suff_datas["xdx"]) - np.matmul(expA - Id, suff_datas["xdx"]).T - np.matmul(expA + Id, bkdx).T - np.matmul(expA + Id, bkdx)
        m1 += np.matmul(expA - Id, np.matmul(suff_datas["xx"], (expA - Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)) + np.matmul(expA - Id, np.matmul(bkx.T, (expA + Id).T)).T + np.matmul(expA + Id, np.matmul(bkbk, (expA + Id).T))

        logdet = (self.dim_x + dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(SST))
        quad_part = -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1))
        return quad_part - 0.5 * logdet

    def generator_one_step(self, x_t, p_t, h_t, dt, friction, force_coeffs, basis, gauss):
        xhalf = x_t + 0.5 * dt * p_t
        force_t = np.matmul(force_coeffs, basis.transform(np.reshape(xhalf, (1, -1)))[0])  # The [0] because transform return an n_timestep*n_features array
        phalf = p_t + 0.5 * dt * force_t
        phalfprime = np.matmul(friction[0 : self.dim_x, 0 : self.dim_x], phalf) + np.matmul(friction[0 : self.dim_x, self.dim_x :], h_t) + gauss[: self.dim_x]
        h_tp = np.matmul(friction[self.dim_x :, 0 : self.dim_x], phalf) + np.matmul(friction[self.dim_x :, self.dim_x :], h_t) + gauss[self.dim_x :]

        p_tp = phalfprime + 0.5 * dt * force_t
        x_tp = xhalf + 0.5 * dt * p_tp
        return x_tp, p_tp, h_tp

    def projA(self, expA, dim_h, dt):
        """
        From full matrix project onto correct model
        """
        # print((-scipy.linalg.logm(expA) / dt))
        A = -scipy.linalg.logm(expA) / dt
        # A[dim_x:, :dim_x] = 0
        A[: self.dim_x, self.dim_x :] = 0
        min_dim = min(self.dim_x, dim_h)
        # A[self.dim_x : self.dim_x + min_dim, :min_dim] = -np.eye(min_dim)
        A[:min_dim, self.dim_x : self.dim_x + min_dim] = np.eye(min_dim)
        return scipy.linalg.expm(-1 * dt * A)

    def mle_derivative_expA_FDT(self, theta, dxdx, xdx, xx, bkbk, bkdx, bkx, dim_tot):
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

    def mle_FDT(self, theta, dxdx, xdx, xx, bkbk, bkdx, bkx, dim_tot):
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

    def negloglike_tominimize(self, theta, suff_datas, dim_h, dt):
        expA = theta[: (self.dim_x + dim_h) ** 2].reshape(self.dim_x + dim_h, self.dim_x + dim_h)
        SST = theta[(self.dim_x + dim_h) ** 2 : 2 * (self.dim_x + dim_h) ** 2].reshape(self.dim_x + dim_h, self.dim_x + dim_h)
        SST = 0.5 * (SST + SST.T)
        coeffs_force = theta[2 * (self.dim_x + dim_h) ** 2 :].reshape(self.dim_x, -1)
        return -self.loglikelihood(suff_datas, expA, SST, coeffs_force, dim_h, dt)

    def negloglike_tominimize_FDT(self, theta, suff_datas, dim_h, dt):
        expA = theta[: (self.dim_x + dim_h) ** 2].reshape(self.dim_x + dim_h, self.dim_x + dim_h)
        kbT = theta[(self.dim_x + dim_h) ** 2]
        coeffs_force = theta[(self.dim_x + dim_h) ** 2 + 1 :].reshape(self.dim_x, -1)
        SST = kbT * (np.eye(self.dim_x + dim_h) - np.matmul(expA, np.matmul(np.eye(self.dim_x + dim_h), expA.T)))
        return -self.loglikelihood(suff_datas, expA, SST, coeffs_force, dim_h, dt)

    def m_step_num(self, expA, SST, coeffs_force, sufficient_stat, dim_h, dt, OptimizeDiffusion, OptimizeForce):
        """
        Do numerical maximization instead of analytical one
        """
        theta0 = np.concatenate((expA.flatten(), SST.flatten(), coeffs_force.flatten()))
        sol = scipy.optimize.minimize(self.negloglike_tominimize, theta0, args=(sufficient_stat, self.dim_x, dim_h, dt), method="Nelder-Mead", options={"maxfev": 5000})
        expA_sol = sol.x[: (self.dim_x + dim_h) ** 2].reshape(self.dim_x + dim_h, self.dim_x + dim_h)
        SST_sol = sol.x[(self.dim_x + dim_h) ** 2 : 2 * (self.dim_x + dim_h) ** 2].reshape(self.dim_x + dim_h, self.dim_x + dim_h)
        force_coeffs_sol = sol.x[2 * (self.dim_x + dim_h) ** 2 :].reshape(self.dim_x, -1)
        if not sol.success:
            warnings.warn("M step did not converge" "{}".format(sol), ConvergenceWarning)
        return expA_sol, force_coeffs_sol, SST_sol
