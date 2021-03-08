"""
This the main estimator module
"""
import numpy as np
from sklearn.utils import check_random_state, check_array

from ._gle_estimator import GLE_Estimator, sufficient_stats


class Markov_Estimator(GLE_Estimator):
    """A Langevin Equation estimator based on Maximum Likelihood algorithm.
        We consider that the free energy have been estimated before and constant values of friction and diffusion coefficients are fitted

    Parameters
    ----------
    dim_x : int, default=1
        The number of visible dimensions

    tol : float, defaults to 5e-4.
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

    OptimizeForce: bool, default=True
        Optimize or not the force coefficients, to be set to False if the force or the potential have been externally determined

    OptimizeDiffusion: bool, default=True
        Optimize or not the diffusion coefficients

    EnforceFDT: bool, default =False
        Enforce the fluctuation-dissipation theorem

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
    """

    def __init__(self, dim_x=1, tol=5e-4, OptimizeForce=True, OptimizeDiffusion=True, EnforceFDT=False, model="euler", A_init=None, C_init=None, force_init=None, random_state=None, warm_start=False, verbose=0, **kwargs):
        super(Markov_Estimator, self).__init__(
            dim_x=dim_x,
            dim_h=0,
            tol=tol,
            max_iter=1,
            OptimizeForce=OptimizeForce,
            OptimizeDiffusion=OptimizeDiffusion,
            EnforceFDT=EnforceFDT,
            init_params="random",
            model=model,
            A_init=A_init,
            C_init=C_init,
            force_init=force_init,
            mu_init=None,
            sig_init=None,
            n_init=1,
            random_state=random_state,
            warm_start=warm_start,
            no_stop=False,
            verbose=verbose,
            verbose_interval=1,
        )

    def fit(self, X, y=None, idx_trajs=[]):
        """Estimate model parameters with the MLE algorithm.

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
        X = check_array(X, ensure_min_samples=4, allow_nd=True)
        self.dt = X[1, 0] - X[0, 0]
        self._check_initial_parameters()

        self._check_n_features(X)

        Xproc, idx_trajs = self.model_class.preprocessingTraj(X, idx_trajs=idx_trajs)
        traj_list = np.split(Xproc, idx_trajs)
        _min_traj_len = np.min([trj.shape[0] for trj in traj_list])
        # print(traj_list)
        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        self.logL = np.empty((n_init, self.max_iter))
        self.logL[:] = np.nan

        # Initial evalution of the sufficient statistics for observables
        datas_visible = 0.0
        for traj in traj_list:
            datas_visible += sufficient_stats(traj, self.dim_x) / len(traj_list)

        if do_init:
            self._initialize_parameters(random_state, traj_len=_min_traj_len)
        self._print_verbose_msg_init_beg(0)
        self._m_step_markov(datas_visible)
        n_init = 0  # To avoid running the loop
        self.converged_ = True
        max_lower_bound = self.loglikelihood(datas_visible, dim_h=0)
        self.n_iter_ = 0
        self.n_best_init_ = 0
        self.lower_bound_ = max_lower_bound
        self._print_verbose_msg_fit_end(max_lower_bound, 0, 0)

        return self
