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

    @staticmethod
    def _projection_matrix(dt, dim_x, dim_h):
        """
        Matrix that put the force at the right place
        """
        Pt = np.zeros((2 * dim_x + dim_h, dim_x))
        Pt[:dim_x, :] = 0.5 * dt**2
        Pt[dim_x : 2 * dim_x, :] = 0.5 * dt - 0.5 * dt**2
        Ptp = np.zeros((2 * dim_x + dim_h, dim_x))
        Ptp[dim_x : 2 * dim_x, :] = -0.5 * dt
        return Pt, Ptp

    @staticmethod
    def _friction_matrix(A, dt, dim_x, dim_h):
        """
        Matrix that put the friction at the right place
        """
        M = np.zeros((2 * dim_x + dim_h, 2 * dim_x + dim_h))

        M[dim_x:, dim_x:] = -A + 0.5 * np.matmul(A, A)
        M[:dim_x, dim_x : 2 * dim_x] = dt * (np.identity(dim_x) - 0.5 * A[:dim_x, dim_x])
        return M

    def preprocessingTraj(self, basis, X, idx_trajs=[]):  # TODO: A vérifier
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

    def compute_expectation_estep(self, traj, A_coeffs, force_coeffs, dim_h, dt, diffusion_coeffs):  # TODO: A vérifier
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

        mutilde[:, self.dim_x : 2 * self.dim_x] = dt / 2 * (np.matmul(force, Avv.T).T + force + force_plus)

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

        return q_plus, mutilde, R, sig_tetha

    def loglikelihood(self, suff_datas, A, SST, coeffs_force, dim_h, dt):  # TODO: A faire
        """
        Return the current value of the log-likelihood
        """
        Pt, Ptp = self._projection_matrix(dt, self.Dim_x, dim_h)
        cft = np.matmul(Pt, coeffs_force)
        bkbk = np.matmul(cft, np.matmul(suff_datas["bkbk"], cft.T))
        bkdx = np.matmul(cft, suff_datas["bkdx"])  # force * a.T
        bkx = np.matmul(cft, suff_datas["bkx"])  # force * v.T
        cftp = np.matmul(Pt, coeffs_force)
        bktpbk = np.matmul(cftp, np.matmul(suff_datas["bktpbk"], cft.T))
        bktpbktp = np.matmul(cftp, np.matmul(suff_datas["bktpbktp"], cftp.T))
        bktpdx = np.matmul(cftp, suff_datas["bktpdx"])  # force * a.T
        bktpx = np.matmul(cftp, suff_datas["bktpx"])  # force * v.T

        Mf = self._friction_matrix(A, dt, self.dim_x, dim_h)
        m1 = suff_datas["dxdx"] - np.matmul(Mf, suff_datas["xdx"]) - np.matmul(Mf, suff_datas["xdx"]).T - bkdx.T - bkdx
        m1 += np.matmul(Mf, np.matmul(suff_datas["xx"], Mf.T)) + np.matmul(Mf, bkx.T) + np.matmul(Mf, bkx.T).T + bkbk
        m1 += bktpbktp + np.matmul(Mf, bktpx.T) + np.matmul(Mf, bktpx.T).T + bktpbk + bktpbk.T - bktpdx - bktpdx.T

        S_mat = np.zeros((2 * self.dim_x + dim_h, 2 * self.dim_x + dim_h))
        sig = np.zeros((2 * self.dim_x + dim_h, 2 * self.dim_x + dim_h))

        sig[: self.dim_x, : self.dim_x] = SST[: self.dim_x, : self.dim_x]
        sig[self.dim_x : 2 * self.dim_x, : self.dim_x] = SST[: self.dim_x, : self.dim_x]
        sig[: self.dim_x, self.dim_x : 2 * self.dim_x] = SST[: self.dim_x, : self.dim_x]
        sig[self.dim_x :, self.dim_x :] = SST

        S_mat[: self.dim_x, : self.dim_x] = dt / (2 * np.sqrt(3)) * np.identity(self.dim_x)

        S_mat[self.dim_x : 2 * self.dim_x, : self.dim_x] = dt / 2 * np.identity(self.dim_x)
        S_mat[: self.dim_x, self.dim_x : 2 * self.dim_x] = -A[: self.dim_x, : self.dim_x] / (2 * np.sqrt(3))
        S_mat[self.dim_x :, self.dim_x :] = np.identity(self.dim_x + dim_h) - 0.5 * A

        sig_tetha = S_mat @ sig @ S_mat.T

        logdet = (self.dim_x + dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(sig_tetha))
        quad_part = -np.trace(np.matmul(np.linalg.inv(sig_tetha), 0.5 * m1))
        # print(SST, np.linalg.det(SST))
        # print(quad_part.shape)
        return quad_part - 0.5 * logdet

    def generator_one_step(self, x_t, p_t, h_t, dt, friction, force_coeffs, basis, gauss):
        force_t = dt * np.matmul(force_coeffs, basis.transform(np.reshape(x_t, (1, -1)))[0])
        x_tp = x_t + dt * (np.identity(self.dim_x) - 0.5 * friction[: self.dim_x, : self.dim_x]) @ p_t + 0.5 * dt**2 * force_t + gauss[: self.dim_x]

        force_tp = dt * np.matmul(force_coeffs, basis.transform(np.reshape(x_tp, (1, -1)))[0])

        force_vel = 0.5 * dt * (force_tp + force_t) - 0.5 * dt * np.matmul(friction[: self.dim_x, : self.dim_x], force_t)
        friction_appl = -friction + 0.5 * np.matmul(friction, friction)

        h_tp = h_t - np.matmul(friction_appl[self.dim_x :, : self.dim_x], p_t) - np.matmul(friction_appl[self.dim_x :, self.dim_x :], h_t) + gauss[self.dim_x :]
        p_tp = p_t - np.matmul(friction_appl[: self.dim_x, : self.dim_x], p_t) - np.matmul(friction_appl[: self.dim_x, self.dim_x :], h_t) + force_vel + gauss[: self.dim_x]
        return x_tp, p_tp, h_tp

    def sufficient_stats(self, traj, dim_x):  # TODO: A vérifier
        """
        Given a sample of trajectory, compute the averaged values of the sufficient statistics
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """

        # xval = traj[:-1, 2 * dim_x : 3 * dim_x]
        # dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]

        bk = traj[:-1, 2 * dim_x : 2 * dim_x + self.dim_basis]
        bktp = traj[:-1, 2 * dim_x + self.dim_basis :]

        q = traj[:, :dim_x]
        dx = traj[:-1, dim_x : 2 * dim_x] - traj[:-1, :dim_x]
        xx = np.mean(q[:, :, np.newaxis] * q[:, np.newaxis, :], axis=0)
        xdx = np.mean(q[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        dxdx = np.mean(dx[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkx = np.mean(bk[:, :, np.newaxis] * q[:, np.newaxis, :], axis=0)
        bkdx = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bkbk = np.mean(bk[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)

        bktpx = np.mean(bktp[:, :, np.newaxis] * q[:, np.newaxis, :], axis=0)
        bktpdx = np.mean(bktp[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
        bktpbk = np.mean(bktp[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)

        return {"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": bkbk, "bktpx": bktpx, "bktpdx": bktpdx, "bktpbk": bktpbk, "µ_0": 0, "Σ_0": 1, "hS": 0}

    def sufficient_stats_hidden(self, muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force, model="obabo"):  # TODO: A vérifier
        """
        Compute the sufficient statistics averaged over the hidden variable distribution
        Datas are stacked as (x, x_plus, bk, bk_plus, original_v)
        """
        # print("Suff_stats")
        xx = np.zeros((2 * dim_x + dim_h, 2 * dim_x + dim_h))
        xx[:dim_x, :dim_x] = old_stats["xx"]
        xdx = np.zeros_like(xx)
        xdx[:dim_x, :dim_x] = old_stats["xdx"]
        dxdx = np.zeros_like(xx)
        dxdx[:dim_x, :dim_x] = old_stats["dxdx"]
        bkx = np.zeros((dim_force, dim_x + dim_h))
        bkx[:, :dim_x] = old_stats["bkx"]
        bkdx = np.zeros_like(bkx)
        bkdx[:, :dim_x] = old_stats["bkdx"]

        bktpx = np.zeros((dim_force, dim_x + dim_h))
        bktpx[:, :dim_x] = old_stats["bktpx"]
        bktpdx = np.zeros_like(bkx)
        bktpdx[:, :dim_x] = old_stats["bktpdx"]

        xval = traj[:-1, dim_x : 2 * dim_x]
        dx = traj[:-1, dim_x : 2 * dim_x] - traj[:-1, :dim_x]
        bk = traj[:-1, 2 * dim_x : 2 * dim_x + self.dim_basis]
        bktp = traj[:-1, 2 * dim_x + self.dim_basis :]

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

        bktpx[:, dim_x:] = np.mean(bktp[:, :, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
        bktpdx[:, dim_x:] = np.mean(bktp[:, :, np.newaxis] * dh[:, np.newaxis, :], axis=0)

        xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
        dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

        detd = np.linalg.det(Sigh[:-1, :, :])
        dets = np.linalg.det(Sigh[:-1, dim_h:, dim_h:])
        hSdouble = 0.5 * np.log(detd[detd > 0.0]).mean()
        hSsimple = 0.5 * np.log(dets[dets > 0.0]).mean()
        # TODO take care of initial value that is missing
        return {"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"], "bktpx": bktpx, "bktpdx": bktpdx, "bktpbk": old_stats["bktpbk"], "µ_0": muh[0, dim_h:], "Σ_0": Sigh[0, dim_h:, dim_h:], "hS": 0.5 * dim_h * (1 + np.log(2 * np.pi)) + hSdouble - hSsimple}
