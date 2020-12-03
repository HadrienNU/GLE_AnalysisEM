"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
import xarray as xr
import scipy.linalg

import warnings
from time import time

from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, check_array
from sklearn.exceptions import ConvergenceWarning


def preprocessingTraj(X, dt, dim_x, force):
    """
    From a flat array compute everythong that is needed for the follwoing computation
    """
    X = check_array(X, ensure_min_features=4, allow_nd=True)
    traj_list = []
    for trj in X:
        x = trj.reshape(-1, dim_x)
        tps = dt * np.arange(x.shape[0])
        v = (np.roll(x, -1, axis=0) - x) / dt
        xv_np = np.hstack((x, v))
        xhalf = xr.DataArray(x + 0.5 * dt * v, coords={"t": tps}, dims=["t", "space"])
        bk = xr.apply_ufunc(lambda x, fb: fb(x), xhalf, kwargs={"fb": force}, input_core_dims=[["space"]], output_core_dims=[["space"]], vectorize=True)

        projmat = np.zeros((dim_x, 2 * dim_x))
        projmat[:dim_x, :dim_x] = 0.5 * dt / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)
        projmat[:dim_x, dim_x : 2 * dim_x] = 1.0 / (1 + (0.5 * dt) ** 2) * np.identity(dim_x)

        P = projmat.copy()
        P[:dim_x, dim_x : 2 * dim_x] = (1 + ((0.5 * dt) ** 2 / (1 + (0.5 * dt) ** 2))) * np.identity(dim_x)
        xv_plus_proj = (np.matmul(projmat, np.roll(xv_np, -1, axis=0).T)).T
        xv_proj = np.matmul(P, xv_np.T).T

        xv = xr.Dataset({"xv_plus_proj": (["t", "dim_x"], xv_plus_proj), "xv_proj": (["t", "dim_x"], xv_proj), "v": (["t", "dim_x"], v), "bk": (["t", "dim_x"], bk)}, coords={"t": tps})
        xv.attrs["lenTraj"] = x.shape[0]
        traj_list.append(xv)
    return traj_list


def generateRandomDefPosMat(dim_tot=2, rng=np.random.default_rng()):
    """
    Generate a random value of the A matrix
    """
    A = rng.standard_normal(size=(dim_tot, dim_tot))
    if not np.all(np.linalg.eigvals(A + A.T) > 0):
        A += np.abs(0.75 * np.min(np.linalg.eigvals(A + A.T))) * np.identity(dim_tot)
    return A


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


def filter_kalman(mutm, Sigtm, Xt, mutilde_tm, expAh, SST, dim_x, dim_h):
    """
    Compute the foward step using Kalman filter, predict and update step
    Parameters
    ----------
    mutm, Sigtm: Values of the foward distribution at t-1
    Xt, mutilde_tm: Values of the trajectories at T and t-1
    expAh, SST: Coefficients parameters["expA"][:, dim_x:] (dim_x+dim_h, dim_h) and SS^T (dim_x+dim_h, dim_x+dim_h)
    dim_x,dim_h: Dimension of visibles and hidden variables
    """
    # Predict step marginalization Normal Gaussian
    mutemp = mutilde_tm + np.matmul(expAh, mutm)
    Sigtemp = SST + np.matmul(expAh, np.matmul(Sigtm, expAh.T))

    # Update step conditionnal Normal Gaussian
    invSYY = np.linalg.inv(Sigtemp[:dim_x, :dim_x])
    marg_mu = mutemp[dim_x:] + np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, Xt - mutemp[:dim_x]))
    marg_sig = Sigtemp[dim_x:, dim_x:] - np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, Sigtemp[dim_x:, :dim_x].T))

    R = expAh[dim_x:, :] - np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, expAh[:dim_x, :]))
    # Pair probability distibution Z_t,Z_{t-1}
    mu_pair = np.hstack((marg_mu, mutm))
    Sig_pair = np.zeros((2 * dim_h, 2 * dim_h))
    Sig_pair[:dim_h, :dim_h] = marg_sig
    Sig_pair[dim_h:, :dim_h] = np.matmul(R, Sigtm)
    Sig_pair[:dim_h, dim_h:] = Sig_pair[dim_h:, :dim_h].T
    Sig_pair[dim_h:, dim_h:] = Sigtm

    return marg_mu, marg_sig, mu_pair, Sig_pair


def smoothing_rauch(muft, Sigft, muStp, SigStp, Xtplus, mutilde_t, expAh, SST, dim_x, dim_h):
    """
    Compute the backward step using Kalman smoother
    """

    invTemp = np.linalg.inv(SST + np.matmul(expAh, np.matmul(Sigft, expAh.T)))
    R = np.matmul(np.matmul(Sigft, expAh.T), invTemp)

    mu_dym_rev = muft + np.matmul(R[:, :dim_x], Xtplus) - np.matmul(R, np.matmul(expAh, muft) + mutilde_t)
    Sig_dym_rev = Sigft - np.matmul(np.matmul(R, expAh), Sigft)

    marg_mu = mu_dym_rev + np.matmul(R[:, dim_x:], muStp)
    marg_sig = np.matmul(R[:, dim_x:], np.matmul(SigStp, R[:, dim_x:].T)) + Sig_dym_rev

    # Pair probability distibution Z_{t+1},Z_{t}
    mu_pair = np.hstack((muStp, marg_mu))
    Sig_pair = np.zeros((2 * dim_h, 2 * dim_h))
    Sig_pair[:dim_h, :dim_h] = SigStp
    Sig_pair[dim_h:, :dim_h] = np.matmul(R[:, dim_x:], SigStp)
    Sig_pair[:dim_h, dim_h:] = Sig_pair[dim_h:, :dim_h].T
    Sig_pair[dim_h:, dim_h:] = marg_sig

    return marg_mu, marg_sig, mu_pair, Sig_pair


def sufficient_stats(traj, dim_x, dim_force):
    """
    Given a sample of trajectory, compute the averaged values of the sufficient statistics
    """

    diffs = traj["xv_plus_proj"].values - traj["xv_proj"].values

    x_val_proj = traj["v"].values

    xx = np.zeros((dim_x, dim_x))
    xdx = np.zeros_like(xx)
    dxdx = np.zeros_like(xx)
    bkx = np.zeros((dim_force, dim_x))
    bkdx = np.zeros_like(bkx)
    bkbk = np.zeros((dim_force, dim_force))

    bk = traj["bk"].data
    lenTraj = traj.attrs["lenTraj"]
    for i in range(lenTraj - 1):  # The -1 comes from the last values removed
        xx += np.outer(x_val_proj[i], x_val_proj[i])
        xdx += np.outer(x_val_proj[i], diffs[i])
        dxdx += np.outer(diffs[i], diffs[i])
        bkx += np.outer(bk[i], x_val_proj[i])
        bkdx += np.outer(bk[i], diffs[i])
        bkbk += np.outer(bk[i], bk[i])

    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": bkbk}) / (lenTraj - 1)


def sufficient_stats_hidden(muh, Sigh, traj, old_stats, dim_x, dim_h, dim_force):
    """
    Compute the sufficient statistics averaged over the hidden variable distribution
    """

    dim_tot = dim_x + dim_h

    xx = np.zeros((dim_tot, dim_tot))
    xx[:dim_x, :dim_x] = old_stats["xx"]
    xdx = np.zeros_like(xx)
    xdx[:dim_x, :dim_x] = old_stats["xdx"]
    dxdx = np.zeros_like(xx)
    dxdx[:dim_x, :dim_x] = old_stats["dxdx"]
    bkx = np.zeros((dim_force, dim_tot))
    bkx[:, :dim_x] = old_stats["bkx"]
    bkdx = np.zeros_like(bkx)
    bkdx[:, :dim_x] = old_stats["bkdx"]

    bk = traj["bk"].data

    diffs_xv = traj["xv_plus_proj"].values - traj["xv_proj"].values

    x_val_proj = traj["v"].values

    lenTraj = traj.attrs["lenTraj"]

    for i in range(lenTraj - 1):  # The -1 comes from the last values removed
        # print(muh[i, dim_h:])
        xx[dim_x:, dim_x:] += Sigh[i, dim_h:, dim_h:] + np.outer(muh[i, dim_h:], muh[i, dim_h:])
        xx[dim_x:, :dim_x] += np.outer(muh[i, dim_h:], x_val_proj[i])

        xdx[dim_x:, dim_x:] += Sigh[i, dim_h:, :dim_h] + np.outer(muh[i, dim_h:], muh[i, :dim_h]) - Sigh[i, dim_h:, dim_h:] - np.outer(muh[i, dim_h:], muh[i, dim_h:])
        xdx[dim_x:, :dim_x] += np.outer(muh[i, dim_h:], diffs_xv[i])
        xdx[:dim_x, dim_x:] += np.outer(x_val_proj[i], muh[i, :dim_h] - muh[i, dim_h:])

        dxdx[dim_x:, dim_x:] += Sigh[i, :dim_h, :dim_h] + np.outer(muh[i, :dim_h], muh[i, :dim_h]) - 2 * Sigh[i, dim_h:, :dim_h] - np.outer(muh[i, dim_h:], muh[i, :dim_h]) - np.outer(muh[i, :dim_h], muh[i, dim_h:]) + Sigh[i, dim_h:, dim_h:] + np.outer(muh[i, dim_h:], muh[i, dim_h:])
        dxdx[dim_x:, :dim_x] += np.outer(muh[i, :dim_h] - muh[i, dim_h:], diffs_xv[i])

        bkx[:, dim_x:] += np.outer(bk[i], muh[i, dim_h:])
        bkdx[:, dim_x:] += np.outer(bk[i], muh[i, :dim_h] - muh[i, dim_h:])

    # Normalisation
    xx[dim_x:, :] /= lenTraj - 1

    xdx[dim_x:, :] /= lenTraj - 1
    xdx[:dim_x, dim_x:] /= lenTraj - 1

    dxdx[dim_x:, :] /= lenTraj - 1

    bkx[:, dim_x:] /= lenTraj - 1
    bkdx[:, dim_x:] /= lenTraj - 1

    xx[:dim_x, dim_x:] = xx[dim_x:, :dim_x].T
    dxdx[:dim_x, dim_x:] = dxdx[dim_x:, :dim_x].T

    return pd.Series({"dxdx": dxdx, "xdx": xdx, "xx": xx, "bkx": bkx, "bkdx": bkdx, "bkbk": old_stats["bkbk"]})


def mle_derivative_expA_FDT(theta, dxdx, xdx, xx, bkbk, bkdx, bkx, invSST, dim_tot):
    """
    Compute the value of the derivative with respect to expA only for the term related to the FDT (i.e. Sigma)
    TODO
    """
    expA = theta.reshape((dim_tot, dim_tot))
    deriv_expA = np.zeros_like(theta)
    # k is the chosen derivative
    YY = dxdx - xdx.T - xdx - 2 * (bkdx + bkdx.T) + xx + 2 * (bkx + bkx.T) + 4 * bkbk
    YX = xdx.T - 2 * bkx + bkdx.T - 2 * bkbk
    XX = xx + bkx + bkx.T + bkbk

    combYX = YY + np.matmul(expA - np.identity(dim_tot), np.matmul(XX, expA.T - np.identity(dim_tot))) - np.matmul(YX, expA.T - np.identity(dim_tot)) - np.matmul(YX, expA.T - np.identity(dim_tot)).T

    for k in range(dim_tot ** 2):
        DA_flat = np.zeros((dim_tot ** 2,))
        DA_flat[k] = 1.0
        DA = DA_flat.reshape((dim_tot, dim_tot))
        deriv_expA[k] = 2 * np.trace(np.matmul(invSST, np.matmul(np.matmul(expA, np.identity(dim_tot) - np.matmul(combYX, invSST)), DA.T)))
        deriv_expA[k] += np.trace(np.matmul(invSST, np.matmul(YX - np.matmul(expA - np.identity(dim_tot), XX), DA.T)))
    return deriv_expA


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

    force : callable, default to -0.5*x
        Evaluation of the force field at x

    A_init, C_init : array, optional
        The user-provided initial coefficients, defaults to None.
        If it None, coefficients are initialized using the `init_params` method.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

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

    def __init__(self, dt=5e-3, dim_x=1, dim_h=1, tol=1e-3, max_iter=100, OptimizeDiffusion=True, EnforceFDT=False, init_params="random", force=lambda x: -0.5 * x, A_init=None, C_init=None, n_init=1, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
        self.dt = dt
        self.dim_x = dim_x
        self.dim_h = dim_h

        self.OptimizeDiffusion = OptimizeDiffusion
        self.EnforceFDT = EnforceFDT

        self.force = force
        self.A_init = A_init
        self.C_init = C_init

        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.warm_start = warm_start

        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def _more_tags(self):
        return {"X_types": list}

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
                warnings.warn("The number of initialization have been puut to 1 as the coefficients are user initialized.")

            if self.A_init is None:
                raise ValueError("No initial values for A is provided and init_params is set to user defined")

        if self.A_init is not None:
            if self.C_init is None:
                self.C_init = np.identity(self.dim_x + self.dim_h) / (self.dim_x + self.dim_h)
            if self.EnforceFDT:
                self.C_init = np.trace(self.C_init) * np.identity(self.dim_x + self.dim_h) / (self.dim_x + self.dim_h)

            expA, SST = convert_user_coefficients(self.dt, self.A_init, self.C_init)
            if not np.all(np.linalg.eigvals(SST) > 0):
                raise ValueError("Provided user values does not lead to definite positive diffusion matrix")

        self.dim_coeffs_force = self.dim_x
        self.coeffs_force = np.identity(self.dim_x)

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
                C = random_state.standard_exponential() * np.identity(self.dim_x + self.dim_h) / (self.dim_x + self.dim_h)
            else:
                temp_mat = generateRandomDefPosMat(self.dim_h + self.dim_x, random_state)
                C = temp_mat + temp_mat.T
            (self.expA, self.SST) = convert_user_coefficients(self.dt, A, C)
        elif self.init_params == "user":
            (self.expA, self.SST) = convert_user_coefficients(self.dt, self.A_init, self.C_init)
        elif self.init_params == "markov":
            self._m_step(suff_stats_visibles)
        else:
            raise ValueError("Unimplemented initialization method '%s'" % self.init_params)

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.
        The method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        Upon consecutive calls, training starts where it left off.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestep, dim_x)
            List of trajectories. Each row
            corresponds to a single trajectory.
        Returns
        -------
        self
        """
        self._check_initial_parameters()
        traj_list = preprocessingTraj(np.real(X), self.dt, self.dim_x, self.force)
        # print(traj_list)
        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        self.nlogL = np.zeros((n_init, self.max_iter))
        best_params = {"A": np.identity(self.dim_x + self.dim_h), "C": np.identity(self.dim_x + self.dim_h)}
        best_n_iter = 1

        # Initial evalution of the sufficient statistics for observables
        datas_visible = 0.0
        for traj in traj_list:
            datas_visible += sufficient_stats(traj, self.dim_x, self.dim_coeffs_force) / len(traj_list)

        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(datas_visible, random_state)

            lower_bound = np.infty if do_init else self.lower_bound_
            # Algorithm loop
            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                new_stat = 0.0
                # hidenS = 0.0
                for traj in traj_list:
                    muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
                    new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas_visible, self.dim_x, self.dim_h, self.dim_coeffs_force) / len(traj_list)
                    # hidenS += hidden_entropy(traj, global_param)
                self._m_step(new_stat)

                lower_bound = self.loglikelihood(new_stat)
                self.nlogL[init, n_iter - 1] = lower_bound
                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change, lower_bound)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break
            self._print_verbose_msg_init_end(lower_bound)
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

            if not self.converged_:
                warnings.warn("Initialization %d did not converge. " "Try different init parameters, " "or increase max_iter, tol " "or check for degenerate data." % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        return self

    def _compute_expectation_estep(self, traj):
        """
        Compute the value of mutilde and Xtplus
        """
        Xt = traj["xv_proj"].values
        Xtplus = traj["xv_plus_proj"].values
        Vt = traj["v"].values
        bkt = traj["bk"].values
        Pf = np.zeros((self.dim_x + self.dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = 0.5 * self.dt * np.identity(self.dim_x)

        mutilde = (np.matmul(np.identity(self.dim_x + self.dim_h)[:, : self.dim_x], Xt.T - Vt.T) + np.matmul(self.expA[:, : self.dim_x], Vt.T) + np.matmul(self.expA + np.identity(self.dim_x + self.dim_h), np.matmul(Pf, bkt.T))).T

        return Xtplus, mutilde

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
        lenTraj = traj.attrs["lenTraj"]
        muf = np.zeros((lenTraj, self.dim_h))
        Sigf = np.zeros((lenTraj, self.dim_h, self.dim_h))
        mus = np.zeros((lenTraj, self.dim_h))
        Sigs = np.zeros((lenTraj, self.dim_h, self.dim_h))
        # To store the pair probability distibution
        muh = np.zeros((lenTraj, 2 * self.dim_h))
        Sigh = np.zeros((lenTraj, 2 * self.dim_h, 2 * self.dim_h))

        Xtplus, mutilde = self._compute_expectation_estep(traj)

        if self.verbose >= 2:
            print("## Forward ##")
        # Forward Proba
        muf[0, :] = np.zeros((self.dim_h,))  # Initialize with Z_0 = 0.
        Sigf[0, :, :] = np.zeros((self.dim_h, self.dim_h))
        # Iterate and compute possible value for h at the same point
        for i in range(1, lenTraj):
            muf[i, :], Sigf[i, :, :], muh[i - 1, :], Sigh[i - 1, :, :] = filter_kalman(muf[i - 1, :], Sigf[i - 1, :, :], Xtplus[i - 1], mutilde[i - 1], self.expA[:, self.dim_x :], self.SST, self.dim_x, self.dim_h)

        # The last step comes only from the forward recursion
        Sigs[-1, :, :] = Sigf[-1, :, :]
        mus[-1, :] = muf[-1, :]

        # Backward proba
        if self.verbose >= 2:
            print("## Backward ##")
        for i in range(lenTraj - 2, -1, -1):  # From T-1 to 0
            mus[i, :], Sigs[i, :, :], muh[i, :], Sigh[i, :, :] = smoothing_rauch(muf[i, :], Sigf[i, :, :], mus[i + 1, :], Sigs[i + 1, :, :], Xtplus[i], mutilde[i], self.expA[:, self.dim_x :], self.SST, self.dim_x, self.dim_h)

        return muh, Sigh

    def _m_step(self, sufficient_stat):
        """M step."""
        Pf = np.zeros((self.dim_x + self.dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = 0.5 * self.dt * np.identity(self.dim_x)

        bkbk = np.matmul(Pf, np.matmul(np.matmul(self.coeffs_force, np.matmul(sufficient_stat["bkbk"], self.coeffs_force.T)), Pf.T))
        bkdx = np.matmul(Pf, np.matmul(self.coeffs_force, sufficient_stat["bkdx"]))
        bkx = np.matmul(Pf, np.matmul(self.coeffs_force, sufficient_stat["bkx"]))
        Id = np.identity(self.dim_x + self.dim_h)
        if not self.EnforceFDT:

            YX = sufficient_stat["xdx"].T - 2 * bkx + bkdx.T - 2 * bkbk
            XX = sufficient_stat["xx"] + bkx + bkx.T + bkbk
            self.expA = Id + np.matmul(YX, np.linalg.inv(XX))
        else:
            theta0 = self.expA.ravel()  # Starting point of the scipy root algorithm
            # To find the better value of the parameters based on the means values
            sol = scipy.optimize.root(mle_derivative_expA_FDT, theta0, args=(sufficient_stat["dxdx"], sufficient_stat["xdx"], sufficient_stat["xx"], bkbk, bkdx, bkx, np.linalg.inv(self.SST), self.dim_x + self.dim_h), method="hybr")
            if not sol.success:
                print(sol)
                raise ValueError("M step did not converge")
            self.expA = sol.x.reshape((self.dim_x + self.dim_h, self.dim_x + self.dim_h))

        # Optimize based on  the variance of the sufficients statistics
        if self.OptimizeDiffusion:
            residuals = sufficient_stat["dxdx"] - np.matmul(self.expA - Id, sufficient_stat["xdx"]) - np.matmul(self.expA - Id, sufficient_stat["xdx"]).T - np.matmul(self.expA + Id, bkdx) - np.matmul(self.expA + Id, bkdx).T
            residuals += np.matmul(self.expA - Id, np.matmul(sufficient_stat["xx"], (self.expA - Id).T)) + np.matmul(self.expA + Id, np.matmul(bkx, (self.expA - Id).T)) + np.matmul(self.expA + Id, np.matmul(bkx, (self.expA - Id).T)).T
            residuals += np.matmul(self.expA + Id, np.matmul(bkbk, (self.expA + Id).T))
            if self.EnforceFDT:  # In which case we only optimize the temperature
                kbT = (self.dim_x + self.dim_h) / np.trace(np.matmul(np.linalg.inv(self.SST), residuals))  # Update the temperature
                self.SST = kbT * (Id - np.matmul(self.expA, self.expA.T))
            else:  # In which case we optimize the full diffusion matrix
                self.SST = residuals

    def loglikelihood(self, suff_datas):
        """
        Return the current value of the negative log-likelihood
        """
        Pf = np.zeros((self.dim_x + self.dim_h, self.dim_x))
        Pf[: self.dim_x, : self.dim_x] = 0.5 * self.dt * np.identity(self.dim_x)

        bkbk = np.matmul(Pf, np.matmul(np.matmul(self.coeffs_force, np.matmul(suff_datas["bkbk"], self.coeffs_force.T)), Pf.T))
        bkdx = np.matmul(Pf, np.matmul(self.coeffs_force, suff_datas["bkdx"]))
        bkx = np.matmul(Pf, np.matmul(self.coeffs_force, suff_datas["bkx"]))

        Id = np.identity(self.dim_x + self.dim_h)
        m1 = suff_datas["dxdx"] - np.matmul(self.expA - Id, suff_datas["xdx"]) - np.matmul(self.expA - Id, suff_datas["xdx"]).T - np.matmul(self.expA + Id, bkdx).T - np.matmul(self.expA + Id, bkdx)
        m1 += np.matmul(self.expA - Id, np.matmul(suff_datas["xx"], (self.expA - Id).T)) + np.matmul(self.expA - Id, np.matmul(bkx.T, (self.expA + Id).T)) + np.matmul(self.expA - Id, np.matmul(bkx.T, (self.expA + Id).T)).T + np.matmul(self.expA + Id, np.matmul(bkbk, (self.expA + Id).T))
        logdet = (self.dim_x + self.dim_h) * np.log(2 * np.pi) + np.log(np.linalg.det(self.SST))
        return -np.trace(np.matmul(np.linalg.inv(self.SST), 0.5 * m1)) - 0.5 * logdet

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.
        """
        check_is_fitted(self)
        # print(X.shape)
        traj_list = preprocessingTraj(X, self.dt, self.dim_x, self.force)
        # Initial evalution of the sufficient statistics for observables
        new_stat = 0.0
        # hidenS = 0.0
        for traj in traj_list:
            # computeForce(traj, init_param, global_param) # To be done before by transform
            datas = sufficient_stats(traj, self.dim_x, self.dim_coeffs_force) / len(traj_list)
            muh, Sigh = self._e_step(traj)  # Compute hidden variable distribution
            new_stat += sufficient_stats_hidden(muh, Sigh, traj, datas, self.dim_x, self.dim_h, self.dim_coeffs_force) / len(traj_list)
            # hidenS += hidden_entropy(traj, global_param)
        return self.loglikelihood(new_stat)  # +hidenS

    def sample(self, n_samples=50):
        """Generate random samples from the fitted GLE model.
        Parameters
        ----------
        n_samples : int, optional
            Number of timestep to generate. Defaults to 50.
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample
        y : array, shape (nsamples,)
            Component labels
        """
        check_is_fitted(self)
        raise NotImplementedError

    def _get_parameters(self):
        A, C = convert_local_coefficients(self.dt, self.expA, self.SST)
        return {"A": A, "C": C}

    def _set_parameters(self, params):
        (self.expA, self.SST) = convert_user_coefficients(self.dt, params["A"], params["C"])

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        if self.EnforceFDT:
            return (self.dim_x + self.dim_h) ** 2 + 1
        else:
            return 2 * (self.dim_x + self.dim_h) ** 2

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.
        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
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
        X : array of shape (n_samples, n_dimensions)
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

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll, log_likelihood):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("***Iteration EM*** : {} / {} --- Current loglikelihood {}".format(n_iter, self.max_iter, log_likelihood))
            elif self.verbose >= 2:
                cur_time = time()
                print("  Iteration %d\t time lapse %.5fs\t ll change %.5f" % (n_iter, cur_time - self._iter_prev_time, diff_ll))
                self._iter_prev_time = cur_time
                print("----------------Current parameters values and diff------------------")
                print("ExpA", self.expA)
                if self.OptimizeDiffusion:
                    print("SST", self.SST)

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print("Initialization converged: %s\t time lapse %.5fs\t ll %.5f" % (self.converged_, time() - self._init_prev_time, ll))
