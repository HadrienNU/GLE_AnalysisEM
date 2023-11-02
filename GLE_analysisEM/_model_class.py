"""
Definition of an abstract model class to be derived by other model
"""
import numpy as np
import scipy.optimize


class AbstractModel:
    def __init__(self, dim_x=1):
        self.dim_x = dim_x

    def expected_features(self):
        return 1 + 2 * self.dim_x

    def _convert_user_coefficients(self, A, C, dt):
        """
        Convert the user provided coefficients into the local one
        """
        raise NotImplementedError

    def _convert_local_coefficients(self, friction_coeffs, diffusion_coeffs, dt):
        """
        Convert the estimator coefficients into the user one
        """
        raise NotImplementedError

    def preprocessingTraj(self, basis, X, idx_trajs=[]):
        raise NotImplementedError

    def compute_expectation_estep(self, traj, A, force_coeffs, dim_h, dt):
        """
        Compute the value of mutilde and Xtplus
        Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
        """
        raise NotImplementedError

    def m_step(self, A_old, SST_old, coeffs_force_old, sufficient_stat, dim_h, dt, OptimizeDiffusion, OptimizeForce):
        """M step.
        Get results from likelihood minimization, should be reimplemented when analytic minimum is known
        """
        # Pour choisir ce qui est optimisé on passe des arguments différents
        init_params = np.hstack((A_old.ravel(), SST_old.ravel(), coeffs_force_old.ravel()))
        res = scipy.optimize.minimize(self.log_likelihood_negative, init_params, args=(sufficient_stat, dim_h, dt), methods="BFGS")

        A = res[: (self.dim_x + dim_h) ** 2].reshape((self.dim_x + dim_h, self.dim_x + dim_h))
        SST = res[(self.dim_x + dim_h) ** 2, 2 * (self.dim_x + dim_h) ** 2].reshape((self.dim_x + dim_h, self.dim_x + dim_h))
        force_coeffs = res[2 * (self.dim_x + dim_h) ** 2 :].reshape((self.dim_x, -1))
        return A, force_coeffs, SST

    def log_likelihood_negative(self, params, sufficient_stat, dim_h, dt):
        """
        Version of the likelihood to minimize for m_step
        """
        A = params[: (self.dim_x + dim_h) ** 2].reshape((self.dim_x + dim_h, self.dim_x + dim_h))
        SST = params[(self.dim_x + dim_h) ** 2, 2 * (self.dim_x + dim_h) ** 2].reshape((self.dim_x + dim_h, self.dim_x + dim_h))
        coeffs_force = params[2 * (self.dim_x + dim_h) ** 2 :].reshape((self.dim_x, -1))
        return -1 * self.log_likelihood(sufficient_stat, A, SST, coeffs_force, dim_h, dt)

    def loglikelihood(self, suff_datas, A, SST, coeffs_force, dim_h, dt):
        """
        Return the current value of the log-likelihood
        """
        raise NotImplementedError

    def generator(self, nsteps=50, dt=5e-3, dim_h=1, x0=None, v0=None, friction=None, SST=None, force_coeffs=None, muh0=0.0, sigh0=0.0, basis=None, rng=np.random.default_rng()):
        """
        Integrate the equation of nsteps steps
        """
        if x0 is None:
            x0 = np.zeros((self.dim_x,))
        if v0 is None:
            v0 = np.zeros((self.dim_x,))
        x_traj = np.empty((nsteps, self.dim_x))
        p_traj = np.empty((nsteps, self.dim_x))
        h_traj = np.zeros((nsteps, dim_h))
        t_traj = np.reshape(np.arange(0.0, nsteps) * dt, (-1, 1))
        x_traj[0, :] = x0
        p_traj[0, :] = v0
        if dim_h > 0:
            h_traj[0, :] = rng.multivariate_normal(muh0, sigh0)
        S = np.linalg.cholesky(SST)
        for n in range(1, nsteps):
            gauss = np.matmul(S, rng.standard_normal(size=S.shape[1]))
            x_traj[n, :], p_traj[n, :], h_traj[n, :] = self.generator_one_step(x_traj[n - 1, :], p_traj[n - 1, :], h_traj[n - 1, :], dt, friction, force_coeffs, basis, gauss)
        return np.hstack((t_traj, x_traj, p_traj)), h_traj

    def generator_one_step(self, x_t, p_t, h_t, dt, friction, force_coeffs, basis, gauss):
        raise NotImplementedError
