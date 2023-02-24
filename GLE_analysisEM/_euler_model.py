"""
This the main estimator module
"""
import numpy as np
import scipy.linalg
import warnings
from ._model_class import AbstractModel


class EulerModel(AbstractModel):
    def _convert_user_coefficients(self, A, C, dt):
        """
        Convert the user provided coefficients into the local one
        """
        friction = A * dt
        diffusion = np.matmul(friction, C) + np.matmul(C, friction.T)
        return friction, diffusion

    def _convert_local_coefficients(self, friction_coeffs, diffusion_coeffs, dt):
        """
        Convert the estimator coefficients into the user one
        """
        if not np.isfinite(np.sum(friction_coeffs)) or not np.isfinite(np.sum(diffusion_coeffs)):  # Check for NaN value
            warnings.warn("NaN of infinite value in friction or diffusion coefficients.")
            return friction_coeffs, diffusion_coeffs

        A = friction_coeffs / dt
        C = scipy.linalg.solve_continuous_lyapunov(friction_coeffs, diffusion_coeffs)

        return A, C

    def preprocessingTraj(self, basis, X, idx_trajs=[]):
        dt = X[1, 0] - X[0, 0]
        v = (np.roll(X[:, 1 : 1 + self.dim_x], -1, axis=0) - X[:, 1 : 1 + self.dim_x]) / dt
        bk = basis.fit_transform(X[:, 1 : 1 + self.dim_x])
        v_plus = np.roll(v, -1, axis=0)
        Xtraj = np.hstack((v_plus, v, v, bk))

        # Remove the last element of each trajectory
        traj_list = np.split(Xtraj, idx_trajs)
        Xtraj_new = None
        idx_new = []
        for trj in traj_list:
            if Xtraj_new is None:
                Xtraj_new = trj[:-1, :]
            else:
                idx_new.append(len(Xtraj_new))
                Xtraj_new = np.vstack((Xtraj_new, trj[:-1, :]))

        return Xtraj_new, idx_new

    def compute_expectation_estep(self, traj, A, force_coeffs, dim_h, dt, diffusion_coeffs):
        """
        Compute the value of mutilde and Xtplus
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """
        Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)
        mutilde = (np.matmul(-A[:, : self.dim_x], traj[:, 2 * self.dim_x : 3 * self.dim_x].T) + np.matmul(Pf, np.matmul(force_coeffs, traj[:, 3 * self.dim_x :].T))).T
        mutilde += np.matmul(np.identity(self.dim_x + dim_h)[:, : self.dim_x], traj[:, self.dim_x : 2 * self.dim_x].T).T  # mutilde is X_t+f(X_t) - A*X_t
        return traj[:, : self.dim_x], mutilde, np.identity(self.dim_x + dim_h)[:, self.dim_x :] - A[:, self.dim_x :], diffusion_coeffs

    def m_step(self, expA_old, SST_old, coeffs_force_old, sufficient_stat, dim_h, dt, OptimizeDiffusion, OptimizeForce):
        """M step.
        TODO:   -Select dimension of fitted parameters from the sufficient stats
        """
        if OptimizeForce:
            invbkbk = np.linalg.inv(sufficient_stat["bkbk"])
            YX = sufficient_stat["xdx"].T - np.matmul(sufficient_stat["bkdx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
            XX = sufficient_stat["xx"] - np.matmul(sufficient_stat["bkx"].T, np.matmul(invbkbk, sufficient_stat["bkx"]))
            A = -np.matmul(YX, np.linalg.inv(XX))

            force_coeffs = (np.matmul(sufficient_stat["bkdx"].T, invbkbk) / dt - np.matmul(A, np.matmul(sufficient_stat["bkx"].T, invbkbk)) / dt)[: self.dim_x, :]
        else:
            force_coeffs = coeffs_force_old
            Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
            Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)
            YX = sufficient_stat["xdx"].T - np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkx"]))
            XX = sufficient_stat["xx"]
            A = -np.matmul(YX, np.linalg.inv(XX))

        if OptimizeDiffusion:  # Optimize Diffusion based on the variance of the sufficients statistics
            Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
            Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)

            bkbk = np.matmul(Pf, np.matmul(np.matmul(force_coeffs, np.matmul(sufficient_stat["bkbk"], force_coeffs.T)), Pf.T))
            bkdx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkdx"]))
            bkx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkx"]))

            residuals = sufficient_stat["dxdx"] + np.matmul(A, sufficient_stat["xdx"]) + np.matmul(A, sufficient_stat["xdx"]).T - bkdx.T - bkdx
            residuals += np.matmul(A, np.matmul(sufficient_stat["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk
            SST = 0.5 * (residuals + residuals.T)
        else:
            SST = 1
        return A, force_coeffs, SST

    def loglikelihood(self, suff_datas, A, SST, coeffs_force, dim_h, dt):
        """
        Return the current value of the log-likelihood
        """
        Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)

        bkbk = np.matmul(Pf, np.matmul(np.matmul(coeffs_force, np.matmul(suff_datas["bkbk"], coeffs_force.T)), Pf.T))
        bkdx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkdx"]))
        bkx = np.matmul(Pf, np.matmul(coeffs_force, suff_datas["bkx"]))

        m1 = suff_datas["dxdx"] + np.matmul(A, suff_datas["xdx"]) + np.matmul(A, suff_datas["xdx"]).T - bkdx.T - bkdx
        m1 += np.matmul(A, np.matmul(suff_datas["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk

        logdet = (self.dim_x + dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(SST))
        quad_part = -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1))
        # print(SST, np.linalg.det(SST))
        return quad_part - 0.5 * logdet

    def generator_one_step(self, x_t, p_t, h_t, dt, friction, force_coeffs, basis, gauss):
        x_tp = x_t + dt * p_t
        force_t = dt * np.matmul(force_coeffs, basis.transform(np.reshape(x_t, (1, -1)))[0])

        h_tp = h_t - np.matmul(friction[self.dim_x :, : self.dim_x], p_t) - np.matmul(friction[self.dim_x :, self.dim_x :], h_t) + gauss[self.dim_x :]
        p_tp = p_t - np.matmul(friction[: self.dim_x, : self.dim_x], p_t) - np.matmul(friction[: self.dim_x, self.dim_x :], h_t) + force_t + gauss[: self.dim_x]
        return x_tp, p_tp, h_tp

    @staticmethod
    def sufficient_stats(traj, dim_x):
        """
        Given a sample of trajectory, compute the averaged values of the sufficient statistics
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """

        xval = traj[:-1, 2 * dim_x : 3 * dim_x]
        dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
        bk = traj[:-1, 3 * dim_x :]
        xx = np.mean(xval[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
        xdx = np.mean(xval[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        dxdx = np.mean(dx[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkx = np.mean(bk[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
        bkdx = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkbk = np.mean(bk[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)

        return {"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": bkbk, "µ_0": 0, "Σ_0": 1, "hS": 0}

    @staticmethod
    def sufficient_stats_hidden(muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force, model="aboba"):
        """
        Compute the sufficient statistics averaged over the hidden variable distribution
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """
        # print("Suff_stats")
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

        xval = traj[:-1, 2 * dim_x : 3 * dim_x]
        dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
        bk = traj[:-1, 3 * dim_x :]

        dh = muh[:-1, :dim_h] - muh[:-1, dim_h:]

        Sigh_tptp = np.mean(Sigh[:-1, :dim_h, :dim_h], axis=0)
        Sigh_ttp = np.mean(Sigh[:-1, dim_h:, :dim_h], axis=0)
        Sigh_tpt = np.mean(Sigh[:-1, :dim_h, dim_h:], axis=0)
        Sigh_tt = np.mean(Sigh[:-1, dim_h:, dim_h:], axis=0)

        muh_tptp = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
        muh_ttp = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
        muh_tpt = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
        muh_tt = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)

        xx[dim_x:, dim_x:] = Sigh_tt + muh_tt
        xx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * xval[:, np.newaxis, :], axis=0)

        xdx[dim_x:, dim_x:] = Sigh_ttp + muh_ttp - Sigh_tt - muh_tt
        xdx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        xdx[:dim_x, dim_x:] = np.mean(xval[:, :, np.newaxis] * dh[:, np.newaxis, :], axis=0)

        dxdx[dim_x:, dim_x:] = Sigh_tptp + muh_tptp - Sigh_ttp - Sigh_tpt - muh_ttp - muh_tpt + Sigh_tt + muh_tt
        dxdx[dim_x:, :dim_x] = np.mean(dh[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)

        bkx[:, dim_x:] = np.mean(bk[:, :, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
        bkdx[:, dim_x:] = np.mean(bk[:, :, np.newaxis] * dh[:, np.newaxis, :], axis=0)

        xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
        dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

        detd = np.linalg.det(Sigh[:-1, :, :])
        dets = np.linalg.det(Sigh[:-1, dim_h:, dim_h:])
        hSdouble = 0.5 * np.log(detd[detd > 0.0]).mean()
        hSsimple = 0.5 * np.log(dets[dets > 0.0]).mean()
        # TODO take care of initial value that is missing
        return {"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"], "µ_0": muh[0, dim_h:], "Σ_0": Sigh[0, dim_h:, dim_h:], "hS": 0.5 * dim_h * (1 + np.log(2 * np.pi)) + hSdouble - hSsimple}


class EulerForceVisibleModel(EulerModel):
    def m_step(self, A_old, SST_old, coeffs_force_old, sufficient_stat, dim_h, dt, OptimizeDiffusion, OptimizeForce):
        """M step.
        TODO:   -Select dimension of fitted parameters from the sufficient stats
        """
        Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)

        if OptimizeForce:
            invxx = np.linalg.inv(sufficient_stat["xx"][: self.dim_x, : self.dim_x])
            Ybk = sufficient_stat["bkdx"][:, : self.dim_x].T - np.matmul(sufficient_stat["xdx"][: self.dim_x, : self.dim_x].T, np.matmul(invxx, sufficient_stat["bkx"][:, : self.dim_x].T))

            bkbk = sufficient_stat["bkbk"] - np.matmul(sufficient_stat["bkx"][:, : self.dim_x], np.matmul(invxx, sufficient_stat["bkx"][:, : self.dim_x].T))
            # print(Ybk, sufficient_stat["bkdx"][:, : self.dim_x].T, -np.matmul(sufficient_stat["xdx"][: self.dim_x, : self.dim_x].T, np.matmul(invxx, sufficient_stat["bkx"][:, : self.dim_x].T)))
            # print(np.matmul(Ybk, np.linalg.inv(dt * bkbk)))
            force_coeffs = (np.matmul(Ybk, np.linalg.inv(dt * bkbk)))[: self.dim_x, :]
        else:
            force_coeffs = coeffs_force_old

        YX = sufficient_stat["xdx"].T - np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkx"]))
        XX = sufficient_stat["xx"]
        A = -np.matmul(YX, np.linalg.inv(XX))

        if OptimizeDiffusion:  # Optimize Diffusion based on the variance of the sufficients statistics
            Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
            Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)

            bkbk = np.matmul(Pf, np.matmul(np.matmul(force_coeffs, np.matmul(sufficient_stat["bkbk"], force_coeffs.T)), Pf.T))
            bkdx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkdx"]))
            bkx = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkx"]))

            residuals = sufficient_stat["dxdx"] + np.matmul(A, sufficient_stat["xdx"]) + np.matmul(A, sufficient_stat["xdx"]).T - bkdx.T - bkdx
            residuals += np.matmul(A, np.matmul(sufficient_stat["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk
            SST = 0.5 * (residuals + residuals.T)
        else:
            SST = 1

        return A, force_coeffs, SST
