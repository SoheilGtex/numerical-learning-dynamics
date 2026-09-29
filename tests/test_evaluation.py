import numpy as np
import pytest

from numerical_learning.evaluation import (
    coefficient_relative_error,
    mae,
    r2_score,
    rmse,
)


def test_hand_checkable_metrics() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 3.0, 1.0])
    assert mae(actual, predicted) == pytest.approx(1.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(5 / 3))
    assert r2_score(actual, predicted) == pytest.approx(-1.5)
    assert coefficient_relative_error(np.array([3.0, 4.0]), np.array([0.0, 4.0])) == pytest.approx(0.75)


def test_constant_target_r2_is_explicitly_undefined() -> None:
    with pytest.raises(ValueError, match="constant target"):
        r2_score(np.ones(3), np.ones(3))
