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
        friction = - np.ln(A) * 2 / dt
        diffusion = np.matmul(friction, C) + np.matmul(C, friction.T)
        return friction, diffusion

    def _convert_local_coefficients(self, friction_coeffs, diffusion_coeffs, dt):
        """
        Convert the estimator coefficients into the user one
        """
        if not np.isfinite(np.sum(friction_coeffs)) or not np.isfinite(np.sum(diffusion_coeffs)):  # Check for NaN value
            warnings.warn("NaN of infinite value in friction or diffusion coefficients.")
            return friction_coeffs, diffusion_coeffs

        A = np.exp( -friction_coeffs * dt / 2 )
        C = scipy.linalg.solve_continuous_lyapunov(friction_coeffs, diffusion_coeffs)

        return A, C

    def preprocessingTraj(self, basis, X, idx_trajs=[]):
        dt = X[1, 0] - X[0, 0]
        #v = (np.roll(X[:, 1 : 1 + self.dim_x], -1, axis=0) - X[:, 1 : 1 + self.dim_x]) / dt
        #v_plus = np.roll(v, -1, axis=0)
        x = X[:, 1 : 1 + self.dim_x]
        x_plus = np.roll(x, -1, axis=0)
        bk = basis.fit_transform(x)
        bk_plus = basis.fit_transform(x_plus)
        Xtraj = np.hstack((x, x_plus, bk, bk_plus))

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
        Basis_l = self.basis.nb_basis_elt_
        x_np1 =  traj[:, : 1 * self.dim_x] + dt**2 / 2 * np.matmul(force_coeffs,  traj[:, 2 * self.dim_x : (2 + Basis_l) * self.dim_x ].T) .T
        v_np1 =  dt/2 * np.matmul(A_coeffs[:, : self.dim_x], 
                                np.matmul(force_coeffs, traj[:, 2 * self.dim_x : (2 + Basis_l) * self.dim_x ].T) +  np.matmul(force_coeffs,  traj[:, (2 + Basis_l) * self.dim_x : (2 + 2 * Basis_l) * self.dim_x ].T) ).T
        
        mutilde = np.asarray([ x_np1 , v_np1 ])
        
        A2 = np.matmul(A_coeffs[:, : self.dim_x],A_coeffs[:, : self.dim_x])
        
        R = np.asarray([ dt * A_coeffs , A2])
        Xtplus = traj[:, self.dim_x : 2 * self.dim_x ]
        Id = np.identity(self.dim_x)
        SIG_TETHA = np.asarray([[ 0 * Id  , dt * diffusion_coeffs * Id ],
                                [ diffusion_coeffs * Id , np.matmul(A_coeffs[:, : self.dim_x], Id)]])
        
        return Xtplus, mutilde, R, SIG_TETHA


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

