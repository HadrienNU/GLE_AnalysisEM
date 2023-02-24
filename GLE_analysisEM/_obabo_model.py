"""
This the main estimator module
"""
import numpy as np
import scipy.linalg
import warnings
from ._model_class import AbstractModel


class OBABO_Model(AbstractModel):
    def _convert_user_coefficients(self, A, C, dt):
        """
        Convert the user provided coefficients into the local one
        """
        friction = scipy.linalg.expm(-A * dt / 2)
        diffusion = np.matmul(friction, C) + np.matmul(C, friction.T)
        return friction, diffusion

    def _convert_local_coefficients(self, friction_coeffs, diffusion_coeffs, dt):
        """
        Convert the estimator coefficients into the user one
        """
        if not np.isfinite(np.sum(friction_coeffs)) or not np.isfinite(np.sum(diffusion_coeffs)):  # Check for NaN value
            warnings.warn("NaN of infinite value in friction or diffusion coefficients.")
            return friction_coeffs, diffusion_coeffs

        A = -scipy.linalg.logm(friction_coeffs) * 2 / dt
        C = scipy.linalg.solve_continuous_lyapunov(friction_coeffs, diffusion_coeffs)

        return A, C

    def preprocessingTraj(self, basis, X, idx_trajs=[]):
        dt = X[1, 0] - X[0, 0]
        # v = (np.roll(X[:, 1 : 1 + self.dim_x], -1, axis=0) - X[:, 1 : 1 + self.dim_x]) / dt
        # v_plus = np.roll(v, -1, axis=0)
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
        Pf = np.zeros((self.dim_x + dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = dt * np.identity(self.dim_x)
        # mutilde = (np.matmul(-A[:, : self.dim_x], traj[:, 2 * self.dim_x : 3 * self.dim_x].T) + np.matmul(Pf, np.matmul(force_coeffs, traj[:, 3 * self.dim_x :].T))).T

        # NEW mutilde = ( 1 + dt**2 / 2 * SOMME(ck bk(x_n)) :$
        #                    e^(-gamma dt) * dt/2 * SOMME(ck bk(x_n)) + e^(-gamma dt) * dt/2 * SOMME(ck bk(x_n+1))
        Basis_l = self.dim_basis
        x = traj[:, : self.dim_x]
        x_plus = traj[:, self.dim_x : 2 * self.dim_x]
        bk = traj[:, 2 * self.dim_x : (2 + Basis_l) * self.dim_x]
        bk_plus = traj[:, (2 + Basis_l) * self.dim_x : (2 + 2 * Basis_l) * self.dim_x]
        force = np.matmul(bk, force_coeffs.T)
        force_plus = np.matmul(bk_plus, force_coeffs.T)
        A_fric = A_coeffs[:, : self.dim_x]
        x_np1 = traj[:, : self.dim_x] + (dt**2 / 2 * force)
        print(A_fric)
<<<<<<< HEAD
        v_np1 = dt / 2 * np.matmul((force + force_plus), A_fric.T)
        mutilde = np.hstack((x_np1, v_np1))

=======
        print(force_coeffs)
        v_np1 =  dt / 2 * np.matmul((force + force_plus),  A_fric.T)
        mutilde = np.hstack(( x_np1 , v_np1 ))
        
>>>>>>> eb3d11d4f1611abea6682399e823b80ed6c67ff1
        A2 = np.matmul(A_coeffs, A_coeffs)
        R = np.vstack((dt * A_coeffs, A2))  # .reshape((self.dim_x + dim_h, dim_h))
        Id = np.identity(self.dim_x)
        SIG_TETHA = np.asarray([[0 * Id, dt * diffusion_coeffs * Id], [diffusion_coeffs * Id, np.matmul(A_coeffs[:, : self.dim_x], Id)]]).reshape((self.dim_x + dim_h, self.dim_x + dim_h))

        return x_plus, mutilde, R, SIG_TETHA

    def m_step(self, expA_old, SST_old, coeffs_force_old, sufficient_stat, dim_h, dt, OptimizeDiffusion, OptimizeForce):
        """M step.
        TODO:   -Select dimension of fitted parameters from the sufficient stats
        """
        if OptimizeForce:
            invBBT = np.linalg.inv(sufficient_stat["BBT"])
            ABT = sufficient_stat["ABT"]
            # invbkbk = np.linalg.inv(sufficient_stat["bkbk"])
            C = np.matmul(ABT, invBBT)

<<<<<<< HEAD
            A = scipy.linalg.expm(scipy.linalg.logm(C[:dim_h, :dim_h] + np.identity(dim_h)) / 2)

            force_coeffs = C[:dim_x, dim_h:] / dt / A[0, 0]
=======
            force_coeffs = C[:dim_h,dim_h:]  / dt / A[0,0] 
>>>>>>> eb3d11d4f1611abea6682399e823b80ed6c67ff1
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

    def sufficient_stats(traj, dim_x):
        """
        Given a sample of trajectory, compute the averaged values of the sufficient statistics
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """

        # xval = traj[:-1, 2 * dim_x : 3 * dim_x]
        # dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
        dim_bk = int(len(traj[0, 2 * dim_x :]) / 2)
        print(dim_bk, type(dim_bk))
        bk = traj[:-1, 2 * dim_x : 2 * dim_x + dim_bk]
        # xx = np.mean(xval[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
        # xdx = np.mean(xval[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        # dxdx = np.mean(dx[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        # bkx = np.mean(bk[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
        # bkdx = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkbk = np.mean(bk[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)

        return {"dxdx": np.zeros((dim_x, dim_x)), "xdx": np.zeros((dim_x, dim_x)), "xx": np.zeros((dim_x, dim_x)), "bkx": np.zeros((dim_bk, dim_x)), "bkdx": np.zeros((dim_bk, dim_x)), "bkbk": bkbk, "µ_0": 0, "Σ_0": 1, "hS": 0}

    def sufficient_stats_hidden(muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force, model="obabo"):
        """
        Compute the sufficient statistics averaged over the hidden variable distribution
        Datas are stacked as (x, x_plus, bk, bk_plus)
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
        bk = traj[:-1, 2 * dim_x : 2 * dim_x + dim_force * dim_x]
        bk_plus = traj[:-1, 2 * dim_x + dim_force * dim_x :]

        x = muh[:-1, dim_h:]
        dx = muh[:-1, :dim_h] - muh[:-1, dim_h:]

        Sigh_tptp = np.mean(Sigh[:-1, :dim_h, :dim_h], axis=0)
        Sigh_ttp = np.mean(Sigh[:-1, dim_h:, :dim_h], axis=0)
        Sigh_tpt = np.mean(Sigh[:-1, :dim_h, dim_h:], axis=0)
        Sigh_tt = np.mean(Sigh[:-1, dim_h:, dim_h:], axis=0)

        muh_tptp = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
        muh_ttp = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
        muh_tpt = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
        muh_tt = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)

        xx[:, :] = Sigh_tt + muh_tt
        # xx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * xval[:, np.newaxis, :], axis=0)

        xdx[:, :] = Sigh_ttp + muh_ttp - Sigh_tt - muh_tt
        # xdx[dim_x:, :dim_x] = np.mean(muh[:-1, dim_h:, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        # xdx[:dim_x, dim_x:] = np.mean(xval[:, :, np.newaxis] * dh[:, np.newaxis, :], axis=0)

        dxdx[:, :] = Sigh_tptp + muh_tptp - Sigh_ttp - Sigh_tpt - muh_ttp - muh_tpt + Sigh_tt + muh_tt
        # dxdx[dim_x:, :dim_x] = np.mean(dh[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)

        bkx[:, :] = np.mean(bk[:, :, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
        bkdx[:, :] = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)

        # xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
        # dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

        B = np.hstack((x, (bk + bk_plus) / 2))
        print(f"B = {B, B.shape}")
        BBT = np.mean(B[:, :, np.newaxis] * B[:, np.newaxis, :], axis=0)
        print(f"BBT = {BBT, BBT.shape}")

        ABT = np.mean(dx[:, :, np.newaxis] * B[:, np.newaxis, :], axis=0)
        print(f"ABT = {ABT, ABT.shape}")

        detd = np.linalg.det(Sigh[:-1, :, :])
        dets = np.linalg.det(Sigh[:-1, dim_h:, dim_h:])
        hSdouble = 0.5 * np.log(detd[detd > 0.0]).mean()
        hSsimple = 0.5 * np.log(dets[dets > 0.0]).mean()
        # TODO take care of initial value that is missing
        return {"BBT": BBT, "ABT": ABT, "dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"], "µ_0": muh[0, dim_h:], "Σ_0": Sigh[0, dim_h:, dim_h:], "hS": 0.5 * dim_h * (1 + np.log(2 * np.pi)) + hSdouble - hSsimple}
