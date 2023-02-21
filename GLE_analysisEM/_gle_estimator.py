"""
This the main estimator module
"""
import numpy as np
import pandas as pd

import warnings
from time import time

from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, check_array
from sklearn.exceptions import ConvergenceWarning

from .random_matrix import generateRandomDefPosMat
from ._euler_model import EulerForceVisibleModel
from ._obabo_model import OBABO_Model

from ._gle_basis_projection import GLE_BasisTransform

# In case the fortran module is not available, there is the python fallback
try:
    from ._filter_smoother import filtersmoother
except ImportError as err:
    print(err)
    warnings.warn("Python fallback will been used for filtersmoother module. Consider compiling the fortran module")
    from ._kalman_python import filtersmoother

import multiprocessing

model_class = {"euler": EulerForceVisibleModel , "obabo": OBABO_Model }


def sufficient_stats(traj, dim_x):
    """
    Given a sample of trajectory, compute the averaged values of the sufficient statistics
    Datas are stacked as (xv_plus_proj, xv_proj, v, bk)
    """

    #xval = traj[:-1, 2 * dim_x : 3 * dim_x]
    #dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
    dim_bk = int(len(traj[0, 2 * dim_x :])/2)
    print(dim_bk , type(dim_bk))
    bk = traj[:-1, 2 * dim_x : 2 * dim_x + 2 * dim_x]
    #xx = np.mean(xval[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
    #xdx = np.mean(xval[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
    #dxdx = np.mean(dx[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
    #bkx = np.mean(bk[:, :, np.newaxis] * xval[:, np.newaxis, :], axis=0)
    #bkdx = np.mean(bk[:, :, np.newaxis] * dx[:, np.newaxis, :], axis=0)
    bkbk = np.mean(bk[:, :, np.newaxis] * bk[:, np.newaxis, :], axis=0)

    return pd.Series({"dxdx": np.zeros((dim_x, dim_x)), "xdx": np.zeros((dim_x, dim_x)), "xx": np.zeros((dim_x, dim_x)), "bkx": np.zeros((dim_bk, dim_x)), "bkdx": np.zeros((dim_bk, dim_x)), "bkbk": bkbk, "µ_0": 0, "Σ_0": 1, "hS": 0})


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

    #xval = traj[:-1, 2 * dim_x : 3 * dim_x]
    #dx = traj[:-1, :dim_x] - traj[:-1, dim_x : 2 * dim_x]
    bk = traj[:-1, 2 * dim_x : 2 * dim_x + dim_force]

    #dh = muh[:-1, :dim_h] - muh[:-1, dim_h:]

    #Sigh_tptp = np.mean(Sigh[:-1, :dim_h, :dim_h], axis=0)
    #Sigh_ttp = np.mean(Sigh[:-1, dim_h:, :dim_h], axis=0)
    #Sigh_tpt = np.mean(Sigh[:-1, :dim_h, dim_h:], axis=0)
    #Sigh_tt = np.mean(Sigh[:-1, dim_h:, dim_h:], axis=0)

    #muh_tptp = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
    #muh_ttp = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, :dim_h], axis=0)
    #muh_tpt = np.mean(muh[:-1, :dim_h, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)
    #muh_tt = np.mean(muh[:-1, dim_h:, np.newaxis] * muh[:-1, np.newaxis, dim_h:], axis=0)

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
    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"], "µ_0": muh[0, dim_h:], "Σ_0": Sigh[0, dim_h:, dim_h:], "hS": 0.5 * dim_h * (1 + np.log(2 * np.pi)) + hSdouble - hSsimple})


def e_step_worker_pool(est, traj, datas_visible, N):
    muh, Sigh = est._e_step(traj)  # Compute hidden variable distribution
    return sufficient_stats_hidden(muh, Sigh, traj, datas_visible, est.dim_x, est.dim_h, est.dim_coeffs_force) / N


class GLE_Estimator(DensityMixin, BaseEstimator):
    """A GLE estimator based on Expectation-Maximation algorithm.

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions

    dim_h : int, default=1
        The number of hidden dimensions

    tol : float, defaults to 1e-5.
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

    max_iter: int, default=100
        The maximum number of EM iterations

    OptimizeForce: bool, default=True
        Optimize or not the force coefficients, to be set to False if the force or the potential have been externally determined

    OptimizeDiffusion: bool, default=True
        Optimize or not the diffusion coefficients

    init_params : {'user','random','markov'}, defaults to 'random'.

        The method used to initialize the fitting coefficients.
        Must be one of::

            'user' : coefficients are initialized at values provided by the user
            'random' : coefficients are initialized randomly.
            'markov' : coefficients are initialized with Markovian estimation of the visible part

    model : {}, default to 'euler'.
        Choice of time discretized model to be fitted. For now only euler model is implemented

    basis: a scikit-learn Transformer class, default to linear basis.
        Transformer to get value of the basis function

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

    multiprocessing: int, default to 1
        Number of process to use for E step
    """

    def __init__(
        self,
        dim_x=1,
        dim_h=1,
        tol=1e-5,
        max_iter=100,
        OptimizeForce=True,
        OptimizeDiffusion=True,
        init_params="random",
        model="euler",
        basis=GLE_BasisTransform(basis_type="linear"),
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
        multiprocessing=1,
    ):
        self.dim_x = dim_x
        self.dim_h = dim_h

        self.OptimizeForce = OptimizeForce
        self.OptimizeDiffusion = OptimizeDiffusion

        self.model = model
        self.basis = basis

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
        self.multiprocessing = multiprocessing

    def _more_tags(self):
        return {"X_types": "2darray"}

    def set_init_coeffs(self, coeffs):
        """Set the initial values of the coefficients via a dict

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
        """Check values of the basic parameters."""
        if self.dt <= 0.0:
            raise ValueError("Invalid value for 'dt': %d " "Timestep should be positive" % self.dt)
        if self.dim_h < 0:
            raise ValueError("Invalid value for 'dim_h': %d " "Estimator requires non-negative hidden dimension" % self.dim_h)
        if self.tol < 0.0:
            raise ValueError("Invalid value for 'tol': %.5f " "Tolerance used by the EM must be non-negative" % self.tol)
        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d " "Estimation requires at least one iteration" % self.max_iter)

        self.model = self.model.casefold()

        if self.model not in model_class.keys():
            raise ValueError("Model {} not implemented".format(self.model))
        self.model_class = model_class[self.model](self.dim_x)

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
            if np.asarray(self.A_init).shape != (self.dim_h, self.dim_h):
                raise ValueError("Wrong dimensions for A_init")
            if self.C_init is None:
                self.C_init = np.identity(self.dim_h)
            expA, SST = self.model_class._convert_user_coefficients(np.asarray(self.A_init), np.asarray(self.C_init), self.dt)
            if not np.all(np.linalg.eigvals(SST) > 0):
                raise ValueError("Provided user values does not lead to definite positive diffusion matrix")

        if self.C_init is not None:
            if np.asarray(self.C_init).shape != (self.dim_h, self.dim_h):
                raise ValueError("Wrong dimensions for C_init")

        if self.mu_init is not None:
            if np.asarray(self.mu_init).shape != (self.dim_h,):
                raise ValueError("Provided user values for initial mean of hidden variables have wrong shape, provided {}, wanted {}".format(np.asarray(self.mu_init).shape, (self.dim_h,)))

        if self.sig_init is not None:
            if np.asarray(self.sig_init).shape != (self.dim_h, self.dim_h):
                raise ValueError("Provided user values for initial variance of hidden variables have wrong shape, provided {}, wanted {}".format(np.asarray(self.sig_init).shape, (self.dim_h, self.dim_h)))
            if not np.all(np.linalg.eigvals(self.sig_init) >= 0):
                raise ValueError("Provided user values for initial variance of hidden variables is not a definite positive diffusion matrix")

        # We initialize the coefficients value to dummy values to ensure existence of the variables
        if not hasattr(self, "friction_coeffs"):
            self.friction_coeffs = np.identity(self.dim_x + self.dim_h)
        if not hasattr(self, "diffusion_coeffs"):
            self.diffusion_coeffs = np.identity(self.dim_x + self.dim_h)

    def _initialize_parameters(self, random_state, traj_len=50):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        self.random_state = check_random_state(random_state)
        if self.init_params == "random" or self.init_params == "markov":
            A = generateRandomDefPosMat(dim_x=self.dim_x, dim_h=self.dim_h, rng=self.random_state, max_ev=(1.0 / 50) / self.dt, min_re_ev=(0.5 / traj_len) / self.dt)  # We ask the typical time scales to be correct with minimum and maximum timescale of the trajectory
            if self.C_init is None:
                # temp_mat = generateRandomDefPosMat(self.dim_h + self.dim_x, random_state)
                # C = temp_mat + temp_mat.T
                C = np.identity(self.dim_x + self.dim_h)
            else:
                C = np.asarray(self.C_init)
            (self.friction_coeffs, self.diffusion_coeffs) = self.model_class._convert_user_coefficients(A, C, self.dt)
        elif self.init_params == "user":
            (self.friction_coeffs, self.diffusion_coeffs) = self.model_class._convert_user_coefficients(np.asarray(self.A_init), np.asarray(self.C_init), self.dt)
            self.force_coeffs = np.asarray(self.force_init).reshape(self.dim_x, -1)
        else:
            raise ValueError("Unimplemented initialization method '%s'" % self.init_params)

        if not self.OptimizeDiffusion and self.A_init is not None and self.C_init is not None:
            _, self.diffusion_coeffs = self.model_class._convert_user_coefficients(np.asarray(self.A_init), np.asarray(self.C_init), self.dt)

        if not hasattr(self.basis, "fitted_"):  # Setup the basis if needed
            dummytraj = np.zeros((1, self.dim_x))
            self.basis.fit(dummytraj)
        self.dim_coeffs_force = self.basis.nb_basis_elt_

        if self.force_init is not None:
            self.force_coeffs = np.asarray(self.force_init).reshape(self.dim_x, -1)
        else:
            self.force_coeffs = -self.random_state.random(size=(self.dim_x, self.dim_coeffs_force))  # -np.ones((self.dim_x, self.dim_coeffs_force))

        # Initial conditions for hidden variables, either user provided or chosen from stationnary state probability fo the hidden variables
        if self.mu_init is not None:
            self.mu0 = np.asarray(self.mu_init)
        else:
            self.mu0 = np.zeros((self.dim_h))

        if self.sig_init is not None:
            self.sig0 = np.asarray(self.sig_init)
            print("hello1")
        elif self.C_init is not None:
            self.sig0 = self.C_init[self.dim_x :, self.dim_x :]
            print("hello2")
        else:
            self.sig0 = np.identity(self.dim_h)
            print("hello3")
        self.initialized_ = True

    def fit(self, X, y=None, idx_trajs=[]):
        """Estimate model parameters with the EM algorithm.
        The method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        Upon consecutive calls, training starts where it left off.


        Parameters
        ----------
        X : array-like, shape (n_samples, dim_x)
            List of positions data.
        idx_trajs: array, default []
            Location of split if multiple trajectory are inputed

        Returns
        -------
        self
        """
        X = check_array(X, ensure_min_samples=4, ensure_min_features=self.dim_x)
        self.dt = X[1, 0] - X[0, 0]
        self._check_initial_parameters()

        Xproc, idx_trajs = self.model_class.preprocessingTraj(self.basis, X, idx_trajs=idx_trajs)
        traj_list = np.split(Xproc, idx_trajs)
        _min_traj_len = np.min([trj.shape[0] for trj in traj_list])
        # print(traj_list)
        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        self.random_state = check_random_state(self.random_state)

        self.logL = np.empty((n_init, self.max_iter))
        self.logL[:] = np.nan

        # Initial evalution of the sufficient statistics for observables
        datas_visible = 0.0
        for traj in traj_list:
            datas_visible += sufficient_stats(traj, self.dim_x) / len(traj_list)

        self.coeffs_list_all = []

        # For referencement
        best_coeffs = None
        best_n_iter = -1
        best_n_init = -1

        for init in range(n_init):
            coeff_list_init = []
            if do_init:
                self._initialize_parameters(self.random_state, traj_len=_min_traj_len)
                if self.init_params == "markov":
                    self._m_step_markov(datas_visible)  # Initialize the visibles coefficients from markovian approx

            self._print_verbose_msg_init_beg(init)
            lower_bound = -np.infty if do_init else self.lower_bound_
            lower_bound_m_step = -np.infty
            # Algorithm loop
            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                new_stat = self._e_step_stats(traj_list, datas_visible)

                # new_stat_rs = self._rescale_hidden(new_stat)
                # new_stat_rs=new_stat

                lower_bound = self.loglikelihood(new_stat)
                if self.verbose >= 2:
                    if lower_bound - lower_bound_m_step < 0:
                        print("Delta ll after E step:", lower_bound - lower_bound_m_step)
                curr_coeffs = self.get_coefficients()
                curr_coeffs["ll"] = lower_bound
                coeff_list_init.append(curr_coeffs)

                self._m_step(new_stat)
                lower_bound_m_step = self.loglikelihood(new_stat)
                if self.verbose >= 2 and lower_bound_m_step - lower_bound < 0:
                    print("Delta ll after M step:", lower_bound_m_step - lower_bound)
                if np.isnan(lower_bound_m_step) or not self._check_finiteness():  # If we have nan value we simply restart the iteration
                    warnings.warn("Initialization %d has NaN values. Ends iteration" % (init), ConvergenceWarning)
                    if self.verbose >= 2:
                        print("Friction:\n{} \n Diffusion:\n{} \n Force :\n{} \n µ0 :\n{} \n Σ0:\n{} \n".format(self.friction_coeffs, self.diffusion_coeffs, self.force_coeffs, self.mu0, self.sig0))
                        print("ll: {}".format(lower_bound))
                    break

                self.logL[init, n_iter - 1] = lower_bound
                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change, lower_bound)

                if lower_bound > max_lower_bound:
                    max_lower_bound = lower_bound
                    best_coeffs = self.get_coefficients()
                    best_n_iter = n_iter
                    best_n_init = init

                if abs(change) < self.tol:
                    self.converged_ = True
                    if not self.no_stop:
                        break

            self._print_verbose_msg_init_end(lower_bound, n_iter)
            self.coeffs_list_all.append(coeff_list_init)
            if not self.converged_:
                warnings.warn("Initialization %d did not converge. " "Try different init parameters, " "or increase max_iter, tol " "or check for degenerate data." % (init + 1), ConvergenceWarning)
        if best_coeffs is not None:
            self.set_coefficients(best_coeffs, with_basis=False)  # Don't set basis coefficients
        self.n_iter_ = best_n_iter
        self.n_best_init_ = best_n_init
        self.lower_bound_ = max_lower_bound
        self._print_verbose_msg_fit_end(max_lower_bound, best_n_init, best_n_iter)

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
        Xtplus, mutilde, R , SIG_TETHA = self.model_class.compute_expectation_estep(traj, self.friction_coeffs, self.force_coeffs, self.dim_h, self.dt, self.diffusion_coeffs)
        print (f""" 
        Xtplus : {Xtplus, Xtplus.shape } \n 
        mutilde : {mutilde, mutilde.shape } \n 
        R :  {R, R.shape} \n 
        SIG_TETHA : {SIG_TETHA , SIG_TETHA.shape} 
        self.mu0 : {self.mu0 , self.mu0.shape}
        self.sig0 : {self.sig0 , self.sig0.shape}
        """)
        return filtersmoother(Xtplus, mutilde, R, SIG_TETHA, self.mu0, self.sig0)

    def _e_step_stats(self, traj_list, datas_visible):
        new_stat = 0.0
        if self.multiprocessing > 1:  # If we ask for more than one process
            with multiprocessing.Pool(processes=self.multiprocessing) as pool:
                proc = [pool.apply_async(e_step_worker_pool, args=(self, traj, datas_visible, len(traj_list))) for traj in traj_list]
                for p in proc:
                    ret = p.get()  # will block
                    new_stat += ret
        else:
            for traj in traj_list:
                muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
                new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas_visible, self.dim_x, self.dim_h, self.dim_coeffs_force) / len(traj_list)
        return new_stat

    def _m_step(self, sufficient_stat):
        """M step.
        .. todo::   -Select dimension of fitted parameters from the sufficient stats (To deal with markovian initialization)
        """
        friction, force, diffusion = self.model_class.m_step(self.friction_coeffs, self.diffusion_coeffs, self.force_coeffs, sufficient_stat, self.dim_h, self.dt, self.OptimizeDiffusion, self.OptimizeForce)

        self.friction_coeffs = friction
        if self.OptimizeForce:
            self.force_coeffs = force
        self.mu0 = sufficient_stat["µ_0"]
        # self.sig0 = sufficient_stat["Σ_0"]
        # A, C = self.model_class._convert_local_coefficients(self.friction_coeffs, self.diffusion_coeffs, self.dt)
        # self.sig0 = C[self.dim_x :, self.dim_x :]
        if self.OptimizeDiffusion:
            self.diffusion_coeffs = diffusion

    def _em_step(self, vect_coeff, traj_list, datas_visible):
        """
        Wrapper of the E and M for gradient descent algorithm
        """
        self.unvectorization_coefficient(vect_coeff)
        new_stat = self._e_step_stats(traj_list, datas_visible)
        return self.loglikelihood(new_stat)

    def _check_finiteness(self):
        """
        Check that all quantities are finite
        """
        return np.isfinite(np.sum(self.friction_coeffs)) and np.isfinite(np.sum(self.diffusion_coeffs)) and np.isfinite(np.sum(self.force_coeffs)) and np.isfinite(np.sum(self.mu0)) and np.isfinite(np.sum(self.sig0))

    def _m_step_markov(self, sufficient_stat_vis):
        """Compute coefficients estimate via Markovian approximation to provide initialization"""
        A_full, C_full = self.model_class._convert_local_coefficients(self.friction_coeffs, self.diffusion_coeffs, self.dt)
        friction, force, diffusion = self.model_class.m_step(self.friction_coeffs, self.diffusion_coeffs, self.force_coeffs, sufficient_stat_vis, 0, self.dt, self.OptimizeDiffusion, self.OptimizeForce)

        A, C = self.model_class._convert_local_coefficients(friction, diffusion, self.dt)
        A_full[: self.dim_x, : self.dim_x] = A

        if self.OptimizeForce:
            self.force_coeffs = force
        if self.OptimizeDiffusion:
            C_full[: self.dim_x, : self.dim_x] = C
        (self.friction_coeffs, self.diffusion_coeffs) = self.model_class._convert_user_coefficients(A_full, C_full, self.dt)

    def loglikelihood(self, suff_datas, dim_h=None):
        """
        Return the current value of the negative log-likelihood
        """
        if dim_h is None:
            dim_h = self.dim_h
        ll = self.model_class.loglikelihood(suff_datas, self.friction_coeffs, self.diffusion_coeffs, self.force_coeffs, dim_h, self.dt)
        if dim_h > 0 and not np.isnan(suff_datas["hS"]):
            return ll + suff_datas["hS"]
        else:
            if dim_h > 0:
                warnings.warn("NaN value in hidden entropy")
            return ll

    def score(self, X, y=None, idx_trajs=[], Xh=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, dim_x)
            List of positions data.
        idx_trajs: array, default []
            Location of split if multiple trajectory are inputed

        Returns
        -------
        log_likelihood : float
            Log likelihood of the generalized Langevin model given X.
        """
        check_is_fitted(self, "initialized_")
        X = check_array(X, ensure_min_samples=4, ensure_min_features=self.dim_x)
        self.dt = X[1, 0] - X[0, 0]

        Xproc, idx_trajs = self.model_class.preprocessingTraj(self.basis, X, idx_trajs=idx_trajs)
        traj_list = np.split(Xproc, idx_trajs)
        # Initial evalution of the sufficient statistics for observables
        new_stat = 0.0
        if Xh is None:
            datas_visible = 0.0
            for traj in traj_list:
                datas_visible += sufficient_stats(traj, self.dim_x) / len(traj_list)
            new_stat = self._e_step_stats(traj_list, datas_visible)
            # muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
            # new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas, self.dim_x, self.dim_h, self.dim_coeffs_force) / len(traj_list)
        else:
            traj_list_h = np.split(Xh, idx_trajs)
            for n, traj in enumerate(traj_list):
                datas_visible = sufficient_stats(traj, self.dim_x)
                zero_sig = np.zeros((len(traj), 2 * self.dim_h, 2 * self.dim_h))
                muh = np.hstack((np.roll(traj_list_h[n], -1, axis=0), traj_list_h[n]))
                new_stat += sufficient_stats_hidden(muh, zero_sig, traj, datas_visible, self.dim_x, self.dim_h, self.dim_coeffs_force) / len(traj_list)
        lower_bound = self.loglikelihood(new_stat)
        return lower_bound

    def predict(self, X, idx_trajs=[]):
        """Predict the hidden variables for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, dim_x)
            List of positions data.
        idx_trajs: array, default []
            Location of split if multiple trajectory are inputed

        Returns
        -------
        muh : array, shape (n_samples,dim_h)
            Average value of the hidden variables given current estimation of the parameters and provided trajectories.
        """
        check_is_fitted(self, "converged_")
        X = check_array(X, ensure_min_samples=4, ensure_min_features=self.dim_x)
        Xproc, idx_trajs = self.model_class.preprocessingTraj(self.basis, X, idx_trajs=idx_trajs)
        traj_list = np.split(Xproc, idx_trajs)
        muh_out = None
        for traj in traj_list:
            muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
            if muh_out is None:
                muh_out = muh[:, self.dim_h :]
            else:
                muh_out = np.hstack((muh_out, muh[:, self.dim_h :]))
        return muh_out

    def sample(self, n_samples=50, n_trajs=1, x0=None, v0=None, dt=5e-3, burnout=0, rng=None):
        """Generate random samples from the fitted GLE model.

        Use the provided basis to compute the force term. The basis should be fitted first, if not will be fitted using a dummy trajectory.

        Parameters
        ----------
        n_samples : int, default=50
            Number of timestep per trajectory to generate. Defaults to 50.
        n_trajs : int, default=1
            Number of trajectory to generate
        x0,v0 : array-like, optionnal
            Initial value of the trajectory
        dt : float, default = 5e-3
            Timestep to use for the integration
        burnout : int, default to 0
            Remove [burnout] step from the start of the trajectories
        rng: random number generator
            Provide if wanted a random number generator. See numpy documention about RNG

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated trajectory
        idx : array
            Index of the trajectories
        y : array, shape (nsamples,)
            Hidden variables values
        """
        if rng is None:
            self.random_state = check_random_state(self.random_state)
        else:
            self.random_state = check_random_state(rng)
        self.dt = dt
        self._check_initial_parameters()
        if not (self.warm_start or hasattr(self, "converged_")):
            self._initialize_parameters(self.random_state, traj_len=n_samples)

        if x0 is None:
            x0 = np.zeros((self.dim_x))
        if v0 is None:
            v0 = np.zeros((self.dim_x))

        X = None
        idx_trajs = []
        X_h = None

        if self.multiprocessing > 1:  # If we ask for more than one process
            child_seeds = np.random.SeedSequence(self.random_state.get_state()[1]).spawn(n_trajs)
            with multiprocessing.Pool(processes=self.multiprocessing) as pool:
                proc = [
                    pool.apply_async(
                        self.model_class.generator,
                        args=(),
                        kwds={
                            "nsteps": n_samples,
                            "rng": np.random.default_rng(child_seeds[n]),
                            "dt": self.dt,
                            "dim_h": self.dim_h,
                            "x0": x0,
                            "v0": v0,
                            "friction": self.friction_coeffs,
                            "SST": self.diffusion_coeffs,
                            "force_coeffs": self.force_coeffs,
                            "muh0": self.mu0,
                            "sigh0": self.sig0,
                            "basis": self.basis,
                        },
                    )
                    for n in range(n_trajs)
                ]
                for p in proc:
                    txv, h = p.get()  # will block
                    if X is None:
                        X = txv[burnout:, :]
                    else:
                        idx_trajs.append(len(X))
                        X = np.vstack((X, txv[burnout:, :]))

                    if X_h is None:
                        X_h = h[burnout:, :]
                    else:
                        X_h = np.vstack((X_h, h[burnout:, :]))
        else:
            for n in range(n_trajs):
                txv, h = self.model_class.generator(nsteps=n_samples, dt=self.dt, dim_h=self.dim_h, x0=x0, v0=v0, friction=self.friction_coeffs, SST=self.diffusion_coeffs, force_coeffs=self.force_coeffs, muh0=self.mu0, sigh0=self.sig0, basis=self.basis, rng=self.random_state)

                if X is None:
                    X = txv[burnout:, :]
                else:
                    idx_trajs.append(len(X))
                    X = np.vstack((X, txv[burnout:, :]))

                if X_h is None:
                    X_h = h[burnout:, :]
                else:
                    X_h = np.vstack((X_h, h[burnout:, :]))
        return X, idx_trajs, X_h

    def get_coefficients(self):
        """Return the actual values of the fitted coefficients."""
        A, C = self.model_class._convert_local_coefficients(self.friction_coeffs, self.diffusion_coeffs, self.dt)
        return {"A": A, "C": C, "force": self.force_coeffs, "µ_0": self.mu0, "Σ_0": self.sig0, "SST": self.diffusion_coeffs, "dt": self.dt, "basis": self.basis.get_coefficients()}

    def set_coefficients(self, coeffs, with_basis=True):
        """Set the value of the coefficients

        Parameters
        ----------
        coeffs : dict
            Contains the value of the coefficients to set.
        """
        (self.friction_coeffs, self.diffusion_coeffs) = self.model_class._convert_user_coefficients(np.asarray(coeffs["A"]), np.asarray(coeffs["C"]), self.dt)
        (self.force_coeffs, self.mu0, self.sig0) = (coeffs["force"], coeffs["µ_0"], coeffs["Σ_0"])
        if with_basis:
            self.basis.set_coefficients(coeffs["basis"])

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
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

    def _print_verbose_msg_init_end(self, ll, best_iter):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s at step %i \t ll %.5f" % (self.converged_, best_iter, ll))
        elif self.verbose >= 2:
            print("Initialization converged: %s at step %i \t time lapse %.5fs\t ll %.5f" % (self.converged_, best_iter, time() - self._init_prev_time, ll))
            print("----------------Current parameters values------------------")
            print(self.get_coefficients())

    def _print_verbose_msg_fit_end(self, ll, best_init, best_iter):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Fit converged: %s Init: %s at step %i \t ll %.5f" % (self.converged_, best_init, best_iter, ll))
        elif self.verbose >= 2:
            print("Fit converged: %s Init: %s at step %i \t time lapse %.5fs\t ll %.5f" % (self.converged_, best_init, best_iter, time() - self._init_prev_time, ll))
            print("----------------Fitted parameters values------------------")
            print(self.get_coefficients())
