.. title:: User guide : contents

.. _user_guide:

==================================================================================
User guide: EM estimator for Generalized Langevin Equations
==================================================================================

EM Estimator
---------

The central piece of the package is the :class:`GLE_analysisEM.GLE_Estimator`. All estimators in scikit-learn are derived
from this class. In more details, this base class enables to set and get
parameters of the estimator. It can be imported as::

    >>> from sklearn.base import BaseEstimator

Once imported, you can create a class which inherate from this base class::

    >>> class MyOwnEstimator(BaseEstimator):
    ...     pass

Transformer
-----------

Transformers are scikit-learn estimators which implement a ``transform`` method.
The use case is the following:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``transform``, `X` will be transformed, using the parameters learned
  during ``fit``.

.. _mixin: https://en.wikipedia.org/wiki/Mixin

In addition, scikit-learn provides a
mixin_, i.e. :class:`sklearn.base.TransformerMixin`, which
implement the combination of ``fit`` and ``transform`` called ``fit_transform``


Therefore, when creating a transformer, you need to create a class which
inherits from both :class:`sklearn.base.BaseEstimator` and
:class:`sklearn.base.TransformerMixin`. The scikit-learn API imposed ``fit`` to
**return ``self``**. The reason is that it allows to pipeline ``fit`` and
``transform`` imposed by the :class:`sklearn.base.TransformerMixin`. The
``fit`` method is expected to have ``X`` and ``y`` as inputs. Note that
``transform`` takes only ``X`` as input and is expected to return the
transformed version of ``X``::

    >>> class MyOwnTransformer(BaseEstimator, TransformerMixin):
    ...     def fit(self, X, y=None):
    ...         return self
    ...     def transform(self, X):
    ...         return X

We build a basic example to show that our :class:`MyOwnTransformer` is working
within a scikit-learn ``pipeline``::

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> pipe = make_pipeline(MyOwnTransformer(),
    ...                      LogisticRegression(random_state=10,
    ...                                         solver='lbfgs'))
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])




TODO List
---------

.. todolist::
