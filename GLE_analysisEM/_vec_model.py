"""
This is the obabo model module where velocity is a hidden variable of the problem.
"""
import numpy as np
import scipy.linalg
import warnings
from ._model_class import AbstractModel


class VEC_Model(AbstractModel):
    hidden_v = True

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
        x = X[:, 1 : 1 + self.dim_x]
        x_plus = np.roll(x, -1, axis=0)
        bk = basis.fit_transform(x)
        bk_plus = basis.fit_transform(x_plus)
        Xtraj = np.hstack((x, x_plus, bk, bk_plus))
        self.dim_basis = basis.nb_basis_elt_
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

    def compute_expectation_estep(self, traj, A_coeffs, force_coeffs, dim_h, dt, diffusion_coeffs):
        """
        Compute the value of mutilde and Xtplus
        Datas are stacked as (x, x plus proj , bk , bk plus proj)
        return Xtplus, mutilde, R, SIG_TETHA
        """

        Basis_l = self.dim_basis
        mutilde = np.zeros((traj.shape[0], 2 * self.dim_x + dim_h))
        R = np.zeros((2 * self.dim_x + dim_h, self.dim_x + dim_h))
        S_mat = np.zeros((2 * self.dim_x + dim_h, 2 * self.dim_x + dim_h))
        sig = np.zeros((2 * self.dim_x + dim_h, 2 * self.dim_x + dim_h))

        q = traj[:, : self.dim_x]
        q_plus = traj[:, self.dim_x : 2 * self.dim_x]

        bk = traj[:, 2 * self.dim_x : (2 + Basis_l) * self.dim_x]
        bk_plus = traj[:, (2 + Basis_l) * self.dim_x : (2 + 2 * Basis_l) * self.dim_x]

        force = np.matmul(bk, force_coeffs.T)
        force_plus = np.matmul(bk_plus, force_coeffs.T)

        Avv = A_coeffs[: self.dim_x, : self.dim_x]

        mutilde[:, : self.dim_x] = q + (dt**2 / 2 * force)

        mutilde[:, self.dim_x : 2 * self.dim_x] = dt / 2 * (np.matmul(force, Avv.T).T + force_plus)

        R[: self.dim_x, : self.dim_x] = dt * (np.identity(self.dim_x) - 0.5 * Avv)

        R[self.dim_x :, :] = np.identity(self.dim_x + dim_h) - A_coeffs + 0.5 * np.matmul(A_coeffs, A_coeffs)

        sig[: self.dim_x, : self.dim_x] = diffusion_coeffs[: self.dim_x, : self.dim_x]
        sig[self.dim_x : 2 * self.dim_x, : self.dim_x] = diffusion_coeffs[: self.dim_x, : self.dim_x]
        sig[: self.dim_x, self.dim_x : 2 * self.dim_x] = diffusion_coeffs[: self.dim_x, : self.dim_x]
        sig[self.dim_x :, self.dim_x :] = diffusion_coeffs

        S_mat[: self.dim_x, : self.dim_x] = dt / (2 * np.sqrt(3)) * np.identity(self.dim_x)

        S_mat[self.dim_x : 2 * self.dim_x, : self.dim_x] = dt / 2 * np.identity(self.dim_x)
        S_mat[: self.dim_x, self.dim_x : 2 * self.dim_x] = -Avv / (2 * np.sqrt(3))
        S_mat[self.dim_x :, self.dim_x :] = np.identity(self.dim_x + dim_h) - 0.5 * A_coeffs
        sig_tetha = S_mat @ sig @ S_mat.T

        # sig_tetha[: self.dim_x, : self.dim_x] = (dt**2 / 3) * diffusion_coeffs[: self.dim_x, : self.dim_x]
        #
        # sig_tetha[self.dim_x : 2 * self.dim_x, : self.dim_x] = (dt / 2) * diffusion_coeffs[self.dim_x :, : self.dim_x] - (5 * dt / 12) * A_coeffs @ diffusion_coeffs[self.dim_x :, : self.dim_x]
        # sig_tetha[: self.dim_x, self.dim_x : 2 * self.dim_x] = sig_tetha[self.dim_x : 2 * self.dim_x, : self.dim_x].T
        #
        # sig_tetha[self.dim_x :, self.dim_x :] = (np.identity(self.dim_x + dim_h) - 0.5 * A_coeffs) @ diffusion_coeffs @ (np.identity(self.dim_x + dim_h) - 0.5 * A_coeffs).T + (1 / 6.0) * A_coeffs @ diffusion_coeffs @ A_coeffs.T

        return q_plus, mutilde, R, sig_tetha

    def m_step(self, expA_old, SST_old, coeffs_force_old, sufficient_stat, dim_h, dt, OptimizeDiffusion, OptimizeForce):  # TODO : a faire
        """M step.
        TODO:   -Select dimension of fitted parameters from the sufficient stats
        """
        if OptimizeForce:
            invBBT = np.linalg.inv(sufficient_stat["BBT"])
            ABT = sufficient_stat["ABT"]
            # invbkbk = np.linalg.inv(sufficient_stat["bkbk"])
            C = np.matmul(ABT, invBBT)

            A = C[: self.dim_x, : self.dim_x] / dt

            force_coeffs = C[: self.dim_x, self.dim_x :] / dt**2 * 2
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

            bkq = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkq"]))

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
        bkbk = np.matmul(coeffs_force, np.matmul(suff_datas["bkbk"], coeffs_force.T))
        bkdx = np.matmul(coeffs_force, suff_datas["bkdx"])  # force * a.T
        bkx = np.matmul(coeffs_force, suff_datas["bkx"])  # force * v.T

        bkq = np.matmul(coeffs_force, suff_datas["bkq"])  # force * x.T

        # m1 = suff_datas["dxdx"] + np.matmul(A, suff_datas["xdx"]) + np.matmul(A, suff_datas["xdx"]).T - bkdx.T - bkdx
        # m1 += np.matmul(A, np.matmul(suff_datas["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk

        # logdet = (self.dim_x + dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(SST))
        # quad_part = -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1))
        # print(SST, np.linalg.det(SST))
        # print(quad_part.shape)
        return 0  # quad_part - 0.5 * logdet

    def generator_one_step(self, x_t, p_t, h_t, dt, friction, force_coeffs, basis, gauss):
        # S_mat = np.zeros((2 * self.dim_x + dim_h, 2 * self.dim_x + dim_h))
        # S_mat[: self.dim_x, : self.dim_x] = dt / (2 * np.sqrt(3)) * np.identity(self.dim_x)
        # S_mat[self.dim_x : 2 * self.dim_x, : self.dim_x] = dt / 2 * np.identity(self.dim_x)
        # S_mat[: self.dim_x, self.dim_x : 2 * self.dim_x] = -friction[: self.dim_x, : self.dim_x] / (2 * np.sqrt(3))
        # S_mat[self.dim_x :, self.dim_x :] = np.identity(self.dim_x + dim_h) - 0.5 * friction
        force_t = dt * np.matmul(force_coeffs, basis.transform(np.reshape(x_t, (1, -1)))[0])
        x_tp = x_t + dt * (np.identity(self.dim_x) + 0.5 * friction) @ p_t + 0.5 * dt**2 * force_t
        # h_tp = h_t - np.matmul(friction[self.dim_x :, : self.dim_x], p_t) - np.matmul(friction[self.dim_x :, self.dim_x :], h_t) + gauss[self.dim_x :]
        # p_tp = p_t - np.matmul(friction[: self.dim_x, : self.dim_x], p_t) - np.matmul(friction[: self.dim_x, self.dim_x :], h_t) + force_t + gauss[: self.dim_x]
        pass
        # return x_tp, p_tp, h_tp

    def sufficient_stats(self, traj, dim_x):
        """
        Given a sample of trajectory, compute the averaged values of the sufficient statistics
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """

        # xval = traj[:-1, 2 * dim_x : 3 * dim_x]
        # dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
        dim_bk = int(len(traj[0, 2 * dim_x :]) / 2)

        # print(dim_bk, type(dim_bk))
        bk = traj[:-1, 2 * dim_x : 2 * dim_x + dim_bk]
        # xx = np.mean(xval[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
        # xdx = np.mean(xval[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        # dxdx = np.mean(dx[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        # bkx = np.mean(bk[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
        # bkdx = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkbk = np.mean(bk[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)

        return {"dxdx": np.zeros((dim_x, dim_x)), "xdx": np.zeros((dim_x, dim_x)), "xx": np.zeros((dim_x, dim_x)), "bkx": np.zeros((dim_bk, dim_x)), "bkdx": np.zeros((dim_bk, dim_x)), "bkbk": bkbk, "µ_0": 0, "Σ_0": 1, "hS": 0}

    def sufficient_stats_hidden(self, muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force, model="obabo"):
        """
        Compute the sufficient statistics averaged over the hidden variable distribution
        Datas are stacked as (x, x_plus, bk, bk_plus, original_v)
        """
        # print("Suff_stats")
        xx = np.zeros((dim_h, dim_h))
        # xx = np.zeros((dim_x + dim_h, dim_x + dim_h))
        # xx[:dim_x, :dim_x] = old_stats["xx"]
        xdx = np.zeros((dim_h, dim_h))
        # xdx[:dim_x, :dim_x] = old_stats["xdx"]
        # dxdx = np.zeros_like(xx)
        dxdx = np.zeros((dim_h, dim_h))
        # dxdx[:dim_x, :dim_x] = old_stats["dxdx"]
        bkx = np.zeros((dim_force, dim_h))
        # bkx[:, :dim_x] = old_stats["bkx"]
        bkdx = np.zeros_like(bkx)
        # bkdx[:, :dim_x] = old_stats["bkdx"]
        # xval = traj[:-1, 2 * dim_x : 3 * dim_x]
        # dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
        q = traj[:-1, :dim_x]
        q_plus = traj[:-1, dim_x : 2 * dim_x]
        bk = traj[:-1, 2 * dim_x : 2 * dim_x + dim_force * dim_x]
        # print(bk, q , bk - q)
        # bk_plus = traj[:-1, 2 * dim_x + dim_force * dim_x : 2 * dim_x + 2 * dim_force * dim_x]

        x = muh[:-1, dim_h:]
        # x = traj[:,2 * dim_x + 2 * dim_force * dim_x :]
        # dx = x[1:] - x[:-1]
        dx = muh[:-1, :dim_h] - muh[:-1, dim_h:]
        # x = x[:-1]
        Sigh_tptp = np.mean(Sigh[:-1, :dim_h, :dim_h], axis=0)
        Sigh_ttp = np.mean(Sigh[:-1, dim_h:, :dim_h], axis=0)
        Sigh_tpt = np.mean(Sigh[:-1, :dim_h, dim_h:], axis=0)
        Sigh_tt = np.mean(Sigh[:-1, dim_h:, dim_h:], axis=0)

        muh_tptp = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
        muh_ttp = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
        muh_tpt = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
        muh_tt = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)

        xx[:, :] = Sigh_tt + muh_tt
        ## xx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * xval[:, np.newaxis, :], axis=0)

        xdx[:, :] = Sigh_ttp + muh_ttp - Sigh_tt - muh_tt
        ## xdx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        ## xdx[:dim_x, dim_x:] = np.mean(xval[:, :, np.newaxis] * dh[:, np.newaxis, :], axis=0)

        dxdx[:, :] = Sigh_tptp + muh_tptp - Sigh_ttp - Sigh_tpt - muh_ttp - muh_tpt + Sigh_tt + muh_tt
        ## dxdx[dim_x:, :dim_x] = np.mean(dh[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)

        bkx[:, :] = np.mean(bk[:, :, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
        bkdx[:, :] = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkq = np.mean(bk[:, :, np.newaxis] * q[:, np.newaxis, :], axis=0)

        # xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
        # dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

        B = np.hstack((x, bk))
        # print(f"B = {B, B.shape}")
        BBT = np.mean(B[:, :, np.newaxis] * B[:, np.newaxis, :], axis=0)
        # print(f"BBT = {BBT, BBT.shape}")
        A = q_plus - q
        ABT = np.mean(A[:, :, np.newaxis] * B[:, np.newaxis, :], axis=0)
        # print(f"ABT = {ABT, ABT.shape}")

        detd = np.linalg.det(Sigh[:-1, :, :])
        dets = np.linalg.det(Sigh[:-1, dim_h:, dim_h:])
        hSdouble = 0.5 * np.log(detd[detd > 0.0]).mean()
        hSsimple = 0.5 * np.log(dets[dets > 0.0]).mean()
        # TODO take care of initial value that is missing
        return {"BBT": BBT, "ABT": ABT, "q": q, "bkq": bkq, "dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"], "µ_0": muh[0, dim_h:], "Σ_0": Sigh[0, dim_h:, dim_h:], "hS": 0.5 * dim_h * (1 + np.log(2 * np.pi)) + hSdouble - hSsimple}
