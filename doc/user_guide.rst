.. title:: User guide : contents

.. _user_guide:

==================================================================================
User guide: EM estimator for Generalized Langevin Equations
==================================================================================

EM Estimator
------------

The central piece of the package is the :class:`GLE_analysisEM.GLE_Estimator`. All estimators in scikit-learn are derived
from this class. In more details, this base class enables to set and get
parameters of the estimator. It can be imported as::

    >>> from GLE_analysisEM import GLE_Estimator

Once imported, lauching the estimation is as simple as ::

    >>> estimator = GLE_Estimator(dim_x=dim_x, dim_h=dim_h, basis=basis)
    >>> estimator.fit(X, idx_trajs=idx)

Several parameters are available. At least ``dim_x`` that give the dimension of the system under study and ``dim_h`` that give the number of hidden dimension to fit should be provided. 
A functional basis for fitting of the mean force is also required and is explained below.

The trajectories data are provided to the ``GLE_analysisEM.fit`` function as ``X`` and ``idx`` under a format that is explained below.

Once fitted (that can be quite long), the estimated parameters can be obtained as a dictionary ::

    >>> estimator.get_coefficients()

Functional basis
-----------------

In GLE_analysisEM, the mean force term is fitted on a functional basis that should be provided to :class:`GLE_analysisEM.GLE_Estimator`. 
Functional basis are implemented in :class:`GLE_analysisEM.GLE_BasisTransform` that could be imported  and initialized as ::

    >>> from GLE_analysisEM import GLE_BasisTransform
    >>> basis = GLE_BasisTransform(basis_type="linear")

Several options are available for the type of basis, please refer to the documentation of  :class:`GLE_analysisEM.GLE_BasisTransform`.  Some type required the basis to be fitted from the data a priori using ::
    
    >>> basis = GLE_BasisTransform(basis_type="free_energy").fit(X)
    
Trajectory format
-----------------

The trajectories should be pass as a single array of the shape (Ndatas x dim). 
If multiple trajectories shoud be provided, all trajectories should be stacked together and another array that contain the indices to split the array should be provided (passed to numpy.split).
Helpers function that load trajectories from files (one file per trajectories) are provided in :class:`GLE_analysisEM.data_loaders`.


Generation of new trajectories
------------------------------
Once the estimator is converged, it can be used to generate new trajectories with ``GLE_analysisEM.sample``.


Predict value of the hidden variables
-------------------------------------

An estimation of the value of the hidden variable can be obtained with ``GLE_analysisEM.predict``


