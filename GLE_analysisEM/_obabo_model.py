"""
This is the obabo model module where velocity is a hidden variable of the problem.
"""
import numpy as np
import scipy.linalg
import warnings
from ._model_class import AbstractModel


class OBABO_Model(AbstractModel):
    hidden_v = True

    def _convert_user_coefficients(self, A, C, dt):
        """
        Convert the user provided coefficients into the local one
        """
        friction = scipy.linalg.expm(-A * dt / 2)
        friction2 = np.identity(friction.shape[0]) - scipy.linalg.expm(-A * dt)
        diffusion = np.matmul(friction2, C) + np.matmul(C, friction2.T)
        return friction, diffusion

    def _convert_local_coefficients(self, friction_coeffs, diffusion_coeffs, dt):
        """
        Convert the estimator coefficients into the user one
        """
        if not np.isfinite(np.sum(friction_coeffs)) or not np.isfinite(
            np.sum(diffusion_coeffs)
        ):  # Check for NaN value
            warnings.warn(
                "NaN of infinite value in friction or diffusion coefficients."
            )
            return friction_coeffs, diffusion_coeffs

        A = -scipy.linalg.logm(friction_coeffs) * 2 / dt
        friction2 = np.identity(friction_coeffs.shape[0]) - np.matmul(
            friction_coeffs, friction_coeffs
        )
        C = scipy.linalg.solve_continuous_lyapunov(friction2, diffusion_coeffs)

        return A, C

    def preprocessingTraj(self, basis, X, idx_trajs=[]):
        dt = X[1, 0] - X[0, 0]
        # v = (np.roll(X[:, 1 : 1 + self.dim_x], -1, axis=0) - X[:, 1 : 1 + self.dim_x]) / dt
        # v_plus = np.roll(v, -1, axis=0)
        x = X[:, 1 : 1 + self.dim_x]
        x_plus = np.roll(x, -1, axis=0)
        bk = basis.fit_transform(x)
        bk_plus = basis.fit_transform(x_plus)
        bkm = (bk + bk_plus) / 2
        v = X[:, 1 + self.dim_x : 1 + 2 * self.dim_x]
        Xtraj = np.hstack((x, x_plus, v, bk, bkm))
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

    def compute_expectation_estep(
        self, traj, A_coeffs, force_coeffs, dim_h, dt, diffusion_coeffs
    ):
        """
        Compute the value of mutilde and Xtplus
        Datas are stacked as (x, x plus proj , bk , bk plus proj)
        return Xtplus, mutilde, R, SIG_TETHA
        """
        ### size of output
        Basis_l = self.dim_basis
        mutilde = np.zeros((len(traj), 2 * self.dim_x + dim_h))
        R = np.zeros((2 * self.dim_x + dim_h, self.dim_x + dim_h))
        sig_tetha = np.zeros((2 * self.dim_x + dim_h, 2 * self.dim_x + dim_h))

        # print(self.dim_x, dim_h)

        q = traj[:, : self.dim_x]
        q_plus = traj[:, self.dim_x : 2 * self.dim_x]

        bk = traj[:, 3 * self.dim_x : (3 + Basis_l) * self.dim_x]
        bk_plus = traj[:, (3 + Basis_l) * self.dim_x :]
        # bkm = (bk + bk_plus) / 2
        force = np.matmul(bk, force_coeffs.T)
        force_plus = np.matmul(bk_plus, force_coeffs.T)

        AVV = A_coeffs[: self.dim_x, : self.dim_x]
        AHV = A_coeffs[self.dim_x :, : self.dim_x]
        AVH = A_coeffs[: self.dim_x, self.dim_x :]
        AHH = A_coeffs[self.dim_x :, self.dim_x :]

        mutilde_q_np1 = q + (dt**2 / 2 * force)
        # print(A_fric)
        # print(force_coeffs)
        mutilde_v_np1 = dt / 2 * np.matmul((force + force_plus), AVV.T)

        # print(mutilde_q_np1.shape)
        # print(mutilde_v_np1.shape)

        mutilde[:, : 2 * self.dim_x] = np.hstack((mutilde_q_np1, mutilde_v_np1))

        AVV2 = np.matmul(AVV, AVV)

        R = np.vstack((dt * AVV, AVV2))
        # print(diffusion_coeffs)
        sig_tetha[:, :] = np.asarray(
            [
                [0, dt**2 * diffusion_coeffs[0, 0]],
                [diffusion_coeffs[0, 0], diffusion_coeffs[0, 0] * AVV2[0, 0]],
            ]
        ).reshape((2 * self.dim_x + dim_h, 2 * self.dim_x + dim_h))
        return q_plus, mutilde, R, sig_tetha

    def m_step(
        self,
        expA_old,
        SST_old,
        coeffs_force_old,
        sufficient_stat,
        dim_h,
        dt,
        OptimizeDiffusion,
        OptimizeForce,
    ):
        """M step.
        TODO:   -Select dimension of fitted parameters from the sufficient stats
        """
        if OptimizeForce:
            invBBT = np.linalg.inv(sufficient_stat["BBT"])
            ABT = sufficient_stat["ABT"]
            # invbkbk = np.linalg.inv(sufficient_stat["bkbk"])
            C = np.matmul(ABT, invBBT)

            A = scipy.linalg.sqrtm(
                C[: self.dim_x, : self.dim_x] + np.identity(self.dim_x)
            )

            force_coeffs = C[: self.dim_x, self.dim_x :] / A / dt
        else:
            force_coeffs = coeffs_force_old
            Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
            Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)
            YX = sufficient_stat["xdx"].T - np.matmul(
                Pf, np.matmul(force_coeffs, sufficient_stat["bkx"])
            )
            XX = sufficient_stat["xx"]
            A = -np.matmul(YX, np.linalg.inv(XX))

        if OptimizeDiffusion:
            # Optimize Diffusion based on the variance of the sufficients statistics

            ### Size of arrays

            bkbk = np.zeros((self.dim_x + dim_h, self.dim_x + dim_h))
            bkdx = np.zeros((self.dim_x + dim_h, self.dim_x))
            bkx = np.zeros_like(bkdx)

            ### Def traj

            bkbk[: self.dim_x, : self.dim_x] = np.matmul(
                force_coeffs, np.matmul(sufficient_stat["bkbk"], force_coeffs.T)
            )

            bkdx[: self.dim_x, : self.dim_x] = np.matmul(
                force_coeffs, sufficient_stat["bkdx"]
            )
            bkx[: self.dim_x, : self.dim_x] = np.matmul(
                force_coeffs, sufficient_stat["bkx"]
            )

            # bkq = np.matmul(Pf, np.matmul(force_coeffs, sufficient_stat["bkq"]))
            A2 = np.zeros(A.shape[0])  # - np.matmul(A, A)
            residuals = (
                sufficient_stat["dxdx"]
                + np.matmul(A2, sufficient_stat["xdx"])
                + np.matmul(A2, sufficient_stat["xdx"]).T
                - bkdx.T
                - bkdx
            )
            residuals += (
                np.matmul(A2, np.matmul(sufficient_stat["xx"], A2.T))
                - np.matmul(A2, bkx.T)
                - np.matmul(A2, bkx.T).T
                + bkbk
            )
            SST = 0.5 * (residuals + residuals.T)
        else:
            SST = np.identity(self.dim_x + dim_h) * 48.206473477
        print(A, force_coeffs, SST)
        return A, force_coeffs, SST

    def loglikelihood(self, suff_datas, A, SST, coeffs_force, dim_h, dt):
        """
        Return the current value of the log-likelihood
        """
        bkbk = np.matmul(coeffs_force, np.matmul(suff_datas["bkbk"], coeffs_force.T))
        bkdx = np.matmul(coeffs_force, suff_datas["bkdx"])  # force * a.T
        bkx = np.matmul(coeffs_force, suff_datas["bkx"])  # force * v.T

        # m1 = suff_datas["dxdx"] + np.matmul(A, suff_datas["xdx"]) + np.matmul(A, suff_datas["xdx"]).T - bkdx.T - bkdx
        # m1 += np.matmul(A, np.matmul(suff_datas["xx"], A.T)) - np.matmul(A, bkx.T) - np.matmul(A, bkx.T).T + bkbk

        # logdet = (self.dim_x + dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(SST))
        # quad_part = -np.trace(np.matmul(np.linalg.inv(SST), 0.5 * m1))
        # print(SST, np.linalg.det(SST))
        # print(quad_part.shape)
        # print(A, coeffs_force)
        return 0  # quad_part - 0.5 * logdet

    def generator_one_step(
        self, x_t, p_t, h_t, dt, friction, force_coeffs, basis, gauss
    ):
        # x_tp = x_t + dt * p_t
        # force_t = dt * np.matmul(force_coeffs, basis.transform(np.reshape(x_t, (1, -1)))[0])

        # h_tp = h_t - np.matmul(friction[self.dim_x :, : self.dim_x], p_t) - np.matmul(friction[self.dim_x :, self.dim_x :], h_t) + gauss[self.dim_x :]
        # p_tp = p_t - np.matmul(friction[: self.dim_x, : self.dim_x], p_t) - np.matmul(friction[: self.dim_x, self.dim_x :], h_t) + force_t + gauss[: self.dim_x]
        pass
        # return x_tp, p_tp, h_tp

    def sufficient_stats(self, traj, dim_x):
        """
        Given a sample of trajectory, compute the averaged values of the sufficient statistics
        Datas are stacked as (x, x_plus, v, bk, bkm)
        """

        # xval = traj[:-1, 2 * dim_x : 3 * dim_x]
        # dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
        dim_bk = int(len(traj[0, 3 * dim_x :]) / 2)

        # print(dim_bk, type(dim_bk))
        bk = traj[:-1, 3 * dim_x : 3 * dim_x + dim_bk]
        bkm = traj[:-1, 3 * dim_x + dim_bk :]

        bkbk = np.mean(bk[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)
        bkmbkm = np.mean(bkm[:, :, np.newaxis] * bkm[:, np.newaxis, :], axis=0)

        return {
            "dxdx": np.zeros((dim_x, dim_x)),
            "xdx": np.zeros((dim_x, dim_x)),
            "xx": np.zeros((dim_x, dim_x)),
            "bkx": np.zeros((dim_bk, dim_x)),
            "bkdx": np.zeros((dim_bk, dim_x)),
            "bkbk": bkbk,
            "bkmbkm": bkmbkm,
            "µ_0": 0,
            "Σ_0": 1,
            "hS": 0,
        }

    def sufficient_stats_hidden(
        self, muh, Sigh, traj, old_stats, dim_x, dim_h_kalman, dim_force, model="obabo"
    ):
        """
        Compute the sufficient statistics averaged over the hidden variable distribution
        Datas are stacked as (x, x_plus, v, bk, bkm)
        """

        ### Sizes of output array  ###

        xx = np.zeros((dim_h_kalman, dim_h_kalman))
        xdx = np.zeros_like(xx)
        dxdx = np.zeros_like(xx)

        bkx = np.zeros((dim_force, dim_h_kalman))
        bkdx = np.zeros_like(bkx)

        bkmx = np.zeros_like(bkx)
        bkmdx = np.zeros_like(bkx)

        ### Data From traj and old_stat ###

        bkbk = old_stats["bkbk"]
        bkmbkm = old_stats["bkmbkm"]

        q = traj[:-1, :dim_x]
        q_plus = traj[:-1, dim_x : 2 * dim_x]

        bk = traj[:-1, 3 * dim_x : (3 + dim_force) * dim_x]
        bkm = traj[:-1, (3 + dim_force) * dim_x :]

        dq = q_plus - q
        dqbk = np.mean(dq[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)

        ### Data From e_step ###

        x = muh[:-1, dim_h_kalman:]
        dx = muh[:-1, :dim_h_kalman] - muh[:-1, dim_h_kalman:]
        dqx = np.mean(dq[:, :, np.newaxis] * x[:, np.newaxis, :], axis=0)

        Sigh_tptp = np.mean(Sigh[:-1, :dim_h_kalman, :dim_h_kalman], axis=0)
        Sigh_ttp = np.mean(Sigh[:-1, dim_h_kalman:, :dim_h_kalman], axis=0)
        Sigh_tpt = np.mean(Sigh[:-1, :dim_h_kalman, dim_h_kalman:], axis=0)
        Sigh_tt = np.mean(Sigh[:-1, dim_h_kalman:, dim_h_kalman:], axis=0)

        muh_tptp = np.mean(
            muh[:-1, :dim_h_kalman, np.newaxis] * muh[:-1, np.newaxis, :dim_h_kalman],
            axis=0,
        )
        muh_ttp = np.mean(
            muh[:-1, dim_h_kalman:, np.newaxis] * muh[:-1, np.newaxis, :dim_h_kalman],
            axis=0,
        )
        muh_tpt = np.mean(
            muh[:-1, :dim_h_kalman, np.newaxis] * muh[:-1, np.newaxis, dim_h_kalman:],
            axis=0,
        )
        muh_tt = np.mean(
            muh[:-1, dim_h_kalman:, np.newaxis] * muh[:-1, np.newaxis, dim_h_kalman:],
            axis=0,
        )

        xx[:, :] = Sigh_tt + muh_tt

        xdx[:, :] = Sigh_ttp + muh_ttp - Sigh_tt - muh_tt

        dxdx[:, :] = (
            Sigh_tptp
            + muh_tptp
            - Sigh_ttp
            - Sigh_tpt
            - muh_ttp
            - muh_tpt
            + Sigh_tt
            + muh_tt
        )

        bkx[:, :] = np.mean(bk[:, :, np.newaxis] * x[:, np.newaxis, :], axis=0)
        bkdx[:, :] = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkmx[:, :] = np.mean(bkm[:, :, np.newaxis] * x[:, np.newaxis, :], axis=0)
        bkmdx[:, :] = np.mean(bkm[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)

        # xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
        # dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

        # B = np.hstack((x, bk))
        # print(f"B = {B, B.shape}")
        BBT = np.vstack((np.hstack((xx, bkmx.T)), np.hstack((bkmx, bkmbkm))))
        # print(f"BBT = {BBT, BBT.shape}")
        # a = q_plus-q

        ABT = np.hstack((xdx.T, bkmdx.T))
        # np.mean(A[:, :, np.newaxis] * B[:, np.newaxis, :], axis=0)
        # print(f"ABT = {ABT, ABT.shape}")

        detd = np.linalg.det(Sigh[:-1, :, :])
        dets = np.linalg.det(Sigh[:-1, dim_h_kalman:, dim_h_kalman:])
        hSdouble = 0.5 * np.log(detd[detd > 0.0]).mean()
        hSsimple = 0.5 * np.log(dets[dets > 0.0]).mean()
        # TODO take care of initial value that is missing
        return {
            "BBT": BBT,
            "ABT": ABT,
            "dxdx": dxdx,
            "dqx": dqx,
            "dqdx": dqbk,
            "xdx": xdx,
            "xx": xx,
            "bkx": bkx,
            "bkmx": bkmx,
            "bkdx": bkdx,
            "bkmdx": bkmdx,
            "bkbk": bkbk,
            "bkmbkm": bkmbkm,
            "µ_0": muh[0, dim_h_kalman:],
            "Σ_0": Sigh[0, dim_h_kalman:, dim_h_kalman:],
            "hS": 0.5 * dim_h_kalman * (1 + np.log(2 * np.pi)) + hSdouble - hSsimple,
        }
