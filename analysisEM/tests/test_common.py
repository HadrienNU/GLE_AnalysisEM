import pytest

from sklearn.utils.estimator_checks import check_estimator

from analysisEM import GLE_Estimator
from analysisEM import GLE_Transformer


@pytest.mark.parametrize("Estimator", [GLE_Estimator, GLE_Transformer])
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
