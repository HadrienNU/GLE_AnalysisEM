"""
This the main estimator module
"""
import numpy as np
import scipy.linalg

import warnings
from time import time

from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, check_array
from sklearn.exceptions import ConvergenceWarning

from .utils import generateRandomDefPosMat, filter_kalman, smoothing_rauch
from ._aboba_model import sufficient_stats_aboba, sufficient_stats_hidden_aboba, mle_derivative_expA_FDT, preprocessingTraj_aboba, compute_expectation_estep_aboba, loglikelihood_aboba, ABOBA_generator
from ._gle_basis_projection import GLE_BasisTransform


def convert_user_coefficients(dt, A, C):
    """
    Convert the user provided coefficients into the local one
    """
    expA = scipy.linalg.expm(-1 * dt * A)
    SST = C - np.matmul(expA, np.matmul(C, expA.T))
    return expA, SST


def convert_local_coefficients(dt, expA, SST):
    """
    Convert the estimator coefficients into the user one
    """
    A = -scipy.linalg.logm(expA) / dt
    C = scipy.linalg.solve_discrete_lyapunov(expA, SST)
    return A, C


def sufficient_stats(traj, dim_x, model="ABOBA"):
    if model == "ABOBA":
        return sufficient_stats_aboba(traj, dim_x)
    else:
        raise ValueError("Model {} not implemented".format(model))


def sufficient_stats_hidden(muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force, model="ABOBA"):
    if model == "ABOBA":
        return sufficient_stats_hidden_aboba(muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force)
    else:
        raise ValueError("Model {} not implemented".format(model))


def preprocessingTraj(X, idx_trajs, dim_x, model="ABOBA"):
    if model == "ABOBA":
        return preprocessingTraj_aboba(X, idx_trajs=idx_trajs, dim_x=dim_x)
    else:
        raise ValueError("Model {} not implemented".format(model))


def compute_expectation_estep(traj, expA, force_coeffs, dim_x, dim_h, dt, model="ABOBA"):
    if model == "ABOBA":
        return compute_expectation_estep_aboba(traj, expA, force_coeffs, dim_x, dim_h, dt)
    else:
        raise ValueError("Model {} not implemented".format(model))


class GLE_Estimator(DensityMixin, BaseEstimator):
    """ A GLE estimator based on Expectation-Maximation algorithm.
        We consider that the free energy have been estimated before and constant values of friction and diffusion coefficients are fitted

    Parameters
    ----------
    dt : float, default=5e-3
        The timestep of the trajectories

    dim_x : int, default=1
        The number of visible dimensions

    dim_h : int, default=1
        The number of hidden dimensions

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

    max_iter: int, default=100
        The maximum number of EM iterations


    OptimizeDiffusion: bool, default=True
        Optimize or not the diffusion coefficients

    EnforceFDT: bool, default =False
        Enforce the fluctuation-dissipation theorem

    init_params : {'markov', 'user','random'}, defaults to 'random'.

        The method used to initialize the fitting coefficients.
        Must be one of::
            'markov' : coefficients are initialized using markovian approximation.
            'user' : coefficients are initialized at values provided by the user
            'random' : coefficients are initialized randomly.

    force : callable, default to lambda x: -1.0*x
        Evaluation of the force field at x

    A_init, C_init, force_init, mu_init, sig_init: array, optional
        The user-provided initial coefficients, defaults to None.
        If it None, coefficients are initialized using the `init_params` method.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    no_stop: bool, default to False
        Does not stop the iterations if the algorithm have converged

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, optional (default=None)
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.
    """

    def __init__(
        self,
        dt=5e-3,
        dim_x=1,
        dim_h=1,
        tol=1e-3,
        max_iter=100,
        OptimizeDiffusion=True,
        EnforceFDT=False,
        init_params="random",
        model="ABOBA",
        A_init=None,
        C_init=None,
        force_init=None,
        mu_init=None,
        sig_init=None,
        n_init=1,
        random_state=None,
        warm_start=False,
        no_stop=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.dt = dt
        self.dim_x = dim_x
        self.dim_h = dim_h

        self.OptimizeDiffusion = OptimizeDiffusion
        self.EnforceFDT = EnforceFDT

        self.model = model

        self.A_init = A_init
        self.C_init = C_init
        self.force_init = force_init
        self.mu_init = mu_init
        self.sig_init = sig_init

        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.warm_start = warm_start

        self.no_stop = no_stop

        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def _more_tags(self):
        return {"X_types": "2darray"}

    def set_init_coeffs(self, coeffs):
        """ Set the initial values of the coefficients via a dict

        Parameters
        ----------
        coeffs : dict
        Contains the wanted values of the initial coefficients,
        if a key is absent the coefficients is set to None
        """
        keys_coeffs = ["A", "C", "force", "µ_0", "Σ_0"]
        init_coeffs = [None] * len(keys_coeffs)
        for n, key in enumerate(keys_coeffs):
            if key in coeffs:
                init_coeffs[n] = coeffs[key]
        self.A_init, self.C_init, self.force_init, self.mu_init, self.sig_init = init_coeffs

    def _check_initial_parameters(self):
        """Check values of the basic parameters.
        """
        if self.dt <= 0.0:
            raise ValueError("Invalid value for 'dt': %d " "Timestep should be positive" % self.dt)
        if self.dim_h < 1:
            raise ValueError("Invalid value for 'dim_h': %d " "Estimation requires at least one hidden dimension" % self.dim_h)
        if self.tol < 0.0:
            raise ValueError("Invalid value for 'tol': %.5f " "Tolerance used by the EM must be non-negative" % self.tol)
        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d " "Estimation requires at least one iteration" % self.max_iter)

        if self.init_params == "user":
            if self.n_init != 1:
                self.n_init = 1
                warnings.warn("The number of initialization have been put to 1 as the coefficients are user initialized.")

            if self.A_init is None:
                raise ValueError("No initial values for A is provided and init_params is set to user defined")
            if self.force_init is None:
                raise ValueError("No initial values for the force is provided and init_params is set to user defined")
            # if self.mu_init is None or self.sig_init is None:
            #     raise ValueError("No initial values for initial conditions are provided and init_params is set to user defined")

        if self.A_init is not None:
            if self.C_init is None:
                self.C_init = np.identity(self.dim_x + self.dim_h) / (self.dim_x + self.dim_h)
            if self.EnforceFDT:
                self.C_init = np.trace(self.C_init) * np.identity(self.dim_x + self.dim_h) / (self.dim_x + self.dim_h)

            expA, SST = convert_user_coefficients(self.dt, np.asarray(self.A_init), np.asarray(self.C_init))
            if not np.all(np.linalg.eigvals(SST) > 0):
                raise ValueError("Provided user values does not lead to definite positive diffusion matrix")

        if self.mu_init is not None:
            if np.asarray(self.mu_init).shape != (self.dim_h,):
                raise ValueError("Provided user values for initial mean of hidden variables have wrong shape, provided {}, wanted {}".format(np.asarray(self.mu_init).shape, (self.dim_h,)))

        if self.sig_init is not None:
            if np.asarray(self.sig_init).shape != (self.dim_h, self.dim_h):
                raise ValueError("Provided user values for initial variance of hidden variables have wrong shape, provided {}, wanted {}".format(np.asarray(self.sig_init).shape, (self.dim_h, self.dim_h)))
            if not np.all(np.linalg.eigvals(self.sig_init) >= 0):
                raise ValueError("Provided user values for initial variance of hidden variables is not a definite positive diffusion matrix")

    def _check_n_features(self, X):
        """Check if we have enough datas to to the fit.
        It is required that  1[time]+2*dim_x[position, velocity]+dim_coeffs_force[force basis] features are present.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        """
        _, n_features = X.shape
        if self.model in ["ABOBA"]:
            expected_features = 1 + 2 * self.dim_x  # Set the number of expected dimension in in input
        elif self.model == "overdamped":
            expected_features = 1 + self.dim_x  # Set the number of expected dimension in in input
        self.dim_coeffs_force = n_features - expected_features
        if self.dim_coeffs_force <= 0:
            raise ValueError(f"X has {n_features} features, but {self.__class__.__name__} " f"is expecting at least {expected_features+1} features as input. Did you forget to add basis features?")

    def _initialize_parameters(self, suff_stats_visibles, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        if self.init_params == "random":
            A = generateRandomDefPosMat(self.dim_h + self.dim_x, random_state)
            if self.EnforceFDT:
                C = np.identity(self.dim_x + self.dim_h)  # random_state.standard_exponential() *
            else:
                if self.C_init is None:
                    # temp_mat = generateRandomDefPosMat(self.dim_h + self.dim_x, random_state)
                    # C = temp_mat + temp_mat.T
                    C = np.identity(self.dim_x + self.dim_h)
                else:
                    C = np.asarray(self.C_init)
            (self.friction_coeffs, self.diffusion_coeffs) = convert_user_coefficients(self.dt, A, C)
        elif self.init_params == "user":
            (self.friction_coeffs, self.diffusion_coeffs) = convert_user_coefficients(self.dt, np.asarray(self.A_init), np.asarray(self.C_init))
            self.force_coeffs = np.asarray(self.force_init).reshape(self.dim_x, -1)
        elif self.init_params == "markov":
            self._m_step(suff_stats_visibles)
        else:
            raise ValueError("Unimplemented initialization method '%s'" % self.init_params)
        if not self.OptimizeDiffusion and self.A_init is not None and self.C_init is not None:
            _, self.diffusion_coeffs = convert_user_coefficients(self.dt, np.asarray(self.A_init), np.asarray(self.C_init))

        if self.force_init is not None:
            self.force_coeffs = np.asarray(self.force_init).reshape(self.dim_x, -1)
        else:
            self.force_coeffs = np.ones(self.dim_coeffs_force).reshape(self.dim_x, -1)
        # Initial conditions for hidden variables, either user provided or chosen from stationnary state probability fo the hidden variables
        if self.mu_init is not None:
            self.mu0 = np.asarray(self.mu_init)
        else:
            self.mu0 = np.zeros((self.dim_h))
        if self.sig_init is not None:
            self.sig0 = np.asarray(self.sig_init)
        else:
            self.sig0 = np.identity(self.dim_h)

    def fit(self, X, y=None, idx_trajs=[]):
        """Estimate model parameters with the EM algorithm.
        The method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        Upon consecutive calls, training starts where it left off.

        ..todo:: Change variable name of expA and SST into friction_coeffs and diffusion_coeffs

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of trajectories. Each row
            corresponds to a single trajectory.
        idx_trajs: array, default []
            Location of split if multiple trajectory are inputed

        Returns
        -------
        self
        """
        self._check_initial_parameters()
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        self._check_n_features(X)

        Xproc = preprocessingTraj(X, idx_trajs=idx_trajs, dim_x=self.dim_x)
        traj_list = np.split(Xproc, idx_trajs)
        # print(traj_list)
        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        self.logL = np.empty((n_init, self.max_iter))
        self.logL[:] = np.nan
        best_coeffs = {"A": np.identity(self.dim_x + self.dim_h), "C": np.identity(self.dim_x + self.dim_h), "µ_0": np.zeros((self.dim_h,)), "Σ_0": np.identity(self.dim_h)}
        best_n_iter = -1

        # Initial evalution of the sufficient statistics for observables
        datas_visible = 0.0
        for traj in traj_list:
            datas_visible += sufficient_stats(traj, self.dim_x)

        for init in range(n_init):

            if do_init:
                self._initialize_parameters(datas_visible, random_state)

            self._print_verbose_msg_init_beg(init)
            lower_bound = -np.infty if do_init else self.lower_bound_
            # Algorithm loop
            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                new_stat = 0.0
                # hidenS = 0.0
                self._enforce_degeneracy()
                for traj in traj_list:
                    muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
                    new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas_visible, self.dim_x, self.dim_h, self.dim_coeffs_force) / len(traj_list)
                    # hidenS += hidden_entropy(traj, global_param)
                self._m_step(new_stat)

                lower_bound = self.loglikelihood(new_stat)
                self.logL[init, n_iter - 1] = lower_bound
                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change, lower_bound)

                if abs(change) < self.tol and n_iter > 2:  # We require at least 2 iterations
                    self.converged_ = True
                    if not self.no_stop:
                        break
            self._print_verbose_msg_init_end(lower_bound)
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_coeffs = self.get_coefficients()
                best_n_iter = n_iter

            if not self.converged_:
                warnings.warn("Initialization %d did not converge. " "Try different init parameters, " "or increase max_iter, tol " "or check for degenerate data." % (init + 1), ConvergenceWarning)

        self.set_coefficients(best_coeffs)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        return self

    def _e_step(self, traj):
        """E step.
        Parameters
        ----------
        traj : array-like, shape (n_timstep, dim_x) One trajectory

        Returns
        -------
        muh : array-like, shape (n_timstep, 2*dim_h)
            Mean values of the pair of the hidden variables
        Sigh : array-like, shape (n_timstep, 2*dim_h,2*dim_h)
            Covariances of the pair of the hidden variables
        """
        # Initialize, we are going to use a numpy array for storing intermediate values and put the resulting µh and \Sigma_h into the xarray only at the end
        lenTraj = len(traj)
        muf = np.zeros((lenTraj, self.dim_h))
        Sigf = np.zeros((lenTraj, self.dim_h, self.dim_h))
        mus = np.zeros((lenTraj, self.dim_h))
        Sigs = np.zeros((lenTraj, self.dim_h, self.dim_h))
        # To store the pair probability distibution
        muh = np.zeros((lenTraj, 2 * self.dim_h))
        Sigh = np.zeros((lenTraj, 2 * self.dim_h, 2 * self.dim_h))

        Xtplus, mutilde = compute_expectation_estep(traj, self.friction_coeffs, self.force_coeffs, self.dim_x, self.dim_h, self.dt, self.model)

        if self.verbose >= 4:
            print("## Forward ##")
        # Forward Proba
        muf[0, :] = self.mu0
        Sigf[0, :, :] = self.sig0
        # Iterate and compute possible value for h at the same point
        for i in range(1, lenTraj):
            muf[i, :], Sigf[i, :, :], muh[i - 1, :], Sigh[i - 1, :, :] = filter_kalman(muf[i - 1, :], Sigf[i - 1, :, :], Xtplus[i - 1], mutilde[i - 1], self.friction_coeffs[:, self.dim_x :], self.diffusion_coeffs, self.dim_x, self.dim_h)

        # The last step comes only from the forward recursion
        Sigs[-1, :, :] = Sigf[-1, :, :]
        mus[-1, :] = muf[-1, :]

        # Backward proba
        if self.verbose >= 4:
            print("## Backward ##")
        for i in range(lenTraj - 2, -1, -1):  # From T-1 to 0
            mus[i, :], Sigs[i, :, :], muh[i, :], Sigh[i, :, :] = smoothing_rauch(muf[i, :], Sigf[i, :, :], mus[i + 1, :], Sigs[i + 1, :, :], Xtplus[i], mutilde[i], self.friction_coeffs[:, self.dim_x :], self.diffusion_coeffs, self.dim_x, self.dim_h)

        return muh, Sigh

    def _m_step(self, sufficient_stat):
        """M step.
        .. todo::   -Select dimension of fitted parameters from the sufficient stats (To deal with markovian initialization)
                -Allow to select statistical model (Euler/ ABOBA)
        """
        Pf = np.zeros((self.dim_x + self.dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = 0.5 * self.dt * np.identity(self.dim_x)

        bkbk = np.matmul(Pf, np.matmul(np.matmul(self.force_coeffs, np.matmul(sufficient_stat["bkbk"], self.force_coeffs.T)), Pf.T))
        bkdx = np.matmul(Pf, np.matmul(self.force_coeffs, sufficient_stat["bkdx"]))
        bkx = np.matmul(Pf, np.matmul(self.force_coeffs, sufficient_stat["bkx"]))
        Id = np.identity(self.dim_x + self.dim_h)
        if not self.EnforceFDT:

            YX = sufficient_stat["xdx"].T - 2 * bkx + bkdx.T - 2 * bkbk
            XX = sufficient_stat["xx"] + bkx + bkx.T + bkbk
            self.friction_coeffs = Id + np.matmul(YX, np.linalg.inv(XX))
            if self.OptimizeDiffusion:  # Optimize Diffusion based on the variance of the sufficients statistics
                residuals = sufficient_stat["dxdx"] - np.matmul(self.friction_coeffs - Id, sufficient_stat["xdx"]) - np.matmul(self.friction_coeffs - Id, sufficient_stat["xdx"]).T - np.matmul(self.friction_coeffs + Id, bkdx) - np.matmul(self.friction_coeffs + Id, bkdx).T
                residuals += (
                    np.matmul(self.friction_coeffs - Id, np.matmul(sufficient_stat["xx"], (self.friction_coeffs - Id).T)) + np.matmul(self.friction_coeffs + Id, np.matmul(bkx, (self.friction_coeffs - Id).T)) + np.matmul(self.friction_coeffs + Id, np.matmul(bkx, (self.friction_coeffs - Id).T)).T
                )
                residuals += np.matmul(self.friction_coeffs + Id, np.matmul(bkbk, (self.friction_coeffs + Id).T))
                print(residuals)
                self.diffusion_coeffs = residuals
        else:
            theta0 = self.friction_coeffs.ravel()  # Starting point of the scipy root algorithm
            # To find the better value of the parameters based on the means values
            sol = scipy.optimize.root(mle_derivative_expA_FDT, theta0, args=(sufficient_stat["dxdx"], sufficient_stat["xdx"], sufficient_stat["xx"], bkbk, bkdx, bkx, np.linalg.inv(self.diffusion_coeffs), self.dim_x + self.dim_h), method="hybr")
            if not sol.success:
                warnings.warn("M step did not converge" "{}".format(sol), ConvergenceWarning)
            self.friction_coeffs = sol.x.reshape((self.dim_x + self.dim_h, self.dim_x + self.dim_h))
            # Optimize Diffusion based on the variance of the sufficients statistics
            if self.OptimizeDiffusion:
                residuals = sufficient_stat["dxdx"] - np.matmul(self.friction_coeffs - Id, sufficient_stat["xdx"]) - np.matmul(self.friction_coeffs - Id, sufficient_stat["xdx"]).T - np.matmul(self.friction_coeffs + Id, bkdx) - np.matmul(self.friction_coeffs + Id, bkdx).T
                residuals += (
                    np.matmul(self.friction_coeffs - Id, np.matmul(sufficient_stat["xx"], (self.friction_coeffs - Id).T)) + np.matmul(self.friction_coeffs + Id, np.matmul(bkx, (self.friction_coeffs - Id).T)) + np.matmul(self.friction_coeffs + Id, np.matmul(bkx, (self.friction_coeffs - Id).T)).T
                )
                residuals += np.matmul(self.friction_coeffs + Id, np.matmul(bkbk, (self.friction_coeffs + Id).T))
                kbT = (self.dim_x + self.dim_h) / np.trace(np.matmul(np.linalg.inv(self.diffusion_coeffs), residuals))  # Update the temperature
                self.diffusion_coeffs = kbT * (Id - np.matmul(self.friction_coeffs, self.friction_coeffs.T))
        self.mu0 = sufficient_stat["µ_0"]
        self.sig0 = sufficient_stat["Σ_0"]

    def _minimize_wrapper_FDT(self, theta, suff_datas, dim_x, dim_h, dt):
        """Wrapper to use for the input of the minimize function of scipy
        """

        return loglikelihood_aboba(suff_datas, self.friction_coeffs, self.diffusion_coeffs, self.force_coeffs, dim_x, dim_h, dt, True)

    def _enforce_degeneracy(self):
        """Apply a basis change to the parameters (hence the hidden variables) to force a specific form of th coefficients
        """

    def loglikelihood(self, suff_datas):
        """
        Return the current value of the negative log-likelihood
        """
        if self.model == "ABOBA":
            return loglikelihood_aboba(suff_datas, self.friction_coeffs, self.diffusion_coeffs, self.force_coeffs, self.dim_x, self.dim_h, self.dt, self.OptimizeDiffusion)
        else:
            raise ValueError("Model {} not implemented".format(self.model))

    def score(self, X, y=None, idx_trajs=[]):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.
        """
        check_is_fitted(self, "converged_")
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        self._check_n_features(X)

        Xproc = preprocessingTraj(X, idx_trajs=idx_trajs, dim_x=self.dim_x)
        traj_list = np.split(Xproc, idx_trajs)
        # Initial evalution of the sufficient statistics for observables
        new_stat = 0.0
        # hidenS = 0.0
        for traj in traj_list:
            datas = sufficient_stats(traj, self.dim_x) / len(traj_list)
            muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
            new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas, self.dim_x, self.dim_h, self.dim_coeffs_force) / len(traj_list)
            # hidenS += hidden_entropy(traj, global_param)
        return self.loglikelihood(new_stat)  # +hidenS

    def predict(self, X, idx_trajs=[]):
        """Predict the hidden variables for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        check_is_fitted(self, "converged_")
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        self._check_n_features(X)
        Xproc = preprocessingTraj(X, idx_trajs=idx_trajs, dim_x=self.dim_x)
        traj_list = np.split(Xproc, idx_trajs)
        muh_out = None
        for traj in traj_list:
            muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
            if muh_out is None:
                muh_out = muh[:, self.dim_h :]
            else:
                muh_out = np.hstack((muh_out, muh[:, self.dim_h :]))
        return muh_out

    def sample(self, n_samples=50, x0=None, v0=None, basis=GLE_BasisTransform()):
        """Generate random samples from the fitted GLE model.

        Use the provided basis to compute the force term. The basis should be fitted first, if not will be fitted using a dummy trajectory.

        Parameters
        ----------
        n_samples : int, optional
            Number of timestep to generate. Defaults to 50.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated trajectory
        y : array, shape (nsamples,)
            Hidden variables values
        """
        random_state = check_random_state(self.random_state)

        self.dim_coeffs_force = basis.degree
        if not (self.warm_start and hasattr(self, "converged_")):
            self._initialize_parameters(None, random_state)
        if not hasattr(self, "fitted_"):  # Setup the basis if needed
            dummytraj = np.zeros((3, 1 + 2 * self.dim_x))
            basis.fit(dummytraj)

        if x0 is None:
            x0 = np.zeros((self.dim_x))
        if v0 is None:
            v0 = np.zeros((self.dim_x))

        if self.model == "ABOBA":
            return ABOBA_generator(nsteps=n_samples, dt=self.dt, dim_x=self.dim_x, dim_h=self.dim_h, x0=x0, v0=v0, expA=self.friction_coeffs, SST=self.diffusion_coeffs, force_coeffs=self.force_coeffs, muh0=self.mu0, sigh0=self.sig0, basis=basis, rng=random_state)
        else:
            raise ValueError("Model {} not implemented".format(self.model))

    def get_coefficients(self):
        """Return the actual values of the fitted coefficients.
        """
        A, C = convert_local_coefficients(self.dt, self.friction_coeffs, self.diffusion_coeffs)
        return {"A": A, "C": C, "F": self.force_coeffs, "µ_0": self.mu0, "Σ_0": self.sig0}

    def set_coefficients(self, coeffs):
        """Set the value of the coefficients

        Parameters
        ----------
        coeffs : dict
            Contains the value of the coefficients to set.
        """
        (self.friction_coeffs, self.diffusion_coeffs) = convert_user_coefficients(self.dt, np.asarray(coeffs["A"]), np.asarray(coeffs["C"]))
        (self.force_coeffs, self.mu0, self.sig0) = (coeffs["F"], coeffs["µ_0"], coeffs["Σ_0"])

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        if self.EnforceFDT:
            return (self.dim_x + self.dim_h) ** 2 + 1 + self.dim_h + self.dim_h ** 2
        else:
            return 2 * (self.dim_x + self.dim_h) ** 2 + self.dim_h + self.dim_h ** 2

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(X.shape[0])

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time
        if self.verbose >= 3:
            print("----------------Current parameters values------------------")
            print(self.get_coefficients())

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll, log_likelihood):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("***Iteration EM*** : {} / {} --- Current loglikelihood {}".format(n_iter, self.max_iter, log_likelihood))
            elif self.verbose >= 2:
                cur_time = time()
                print("***Iteration EM*** :%d / %d\t time lapse %.5fs\t Current loglikelihood %.5f loglikelihood change %.5f" % (n_iter, self.max_iter, cur_time - self._iter_prev_time, log_likelihood, diff_ll))
                self._iter_prev_time = cur_time
            if self.verbose >= 3:
                print("----------------Current parameters values------------------")
                print(self.get_coefficients())

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print("Initialization converged: %s\t time lapse %.5fs\t ll %.5f" % (self.converged_, time() - self._init_prev_time, ll))
            print("----------------Current parameters values------------------")
            print(self.get_coefficients())
