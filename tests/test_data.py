import numpy as np
import pytest

from numerical_learning.data import FEATURE_NAMES, generate_synthetic_data


def test_fixed_seed_is_deterministic() -> None:
    first = generate_synthetic_data(n_samples=12, seed=7)
    second = generate_synthetic_data(n_samples=12, seed=7)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)


def test_different_seed_changes_observations() -> None:
    first = generate_synthetic_data(n_samples=12, seed=7)
    second = generate_synthetic_data(n_samples=12, seed=8)
    assert not np.array_equal(first.X, second.X)
    assert not np.array_equal(first.y, second.y)


def test_shapes_ranges_and_ground_truth() -> None:
    dataset = generate_synthetic_data(n_samples=25, seed=3)
    assert dataset.X.shape == (25, 5)
    assert dataset.y.shape == (25,)
    assert dataset.true_coefficients.shape == (6,)
    assert dataset.feature_names == FEATURE_NAMES
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.y).all()
    assert np.all((dataset.X[:, 0] >= 8) & (dataset.X[:, 0] <= 20))
    assert np.all((dataset.X[:, 2] >= 0.6) & (dataset.X[:, 2] <= 1.0))


def test_invalid_generator_arguments() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        generate_synthetic_data(n_samples=1)
    with pytest.raises(ValueError, match="noise_std"):
        generate_synthetic_data(noise_std=-1)
