import numpy as np

from numerical_learning.temporal_data import (
    CHECKPOINTS,
    generate_temporal_data,
    make_temporal_split,
)


def test_temporal_generation_is_deterministic_and_finite() -> None:
    first = generate_temporal_data(seed=17)
    second = generate_temporal_data(seed=17)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert np.isfinite(first.X).all() and np.isfinite(first.y).all()


def test_checkpoint_features_are_nested_without_future_leakage() -> None:
    dataset = generate_temporal_data(seed=19)
    previous = set()
    for checkpoint in CHECKPOINTS:
        current = set(dataset.checkpoint_features[checkpoint])
        assert previous <= current
        previous = current
    assert "midterm" not in dataset.checkpoint_features["early_semester"]
    assert "updated_attendance" not in dataset.checkpoint_features["mid_semester"]


def test_split_is_shared_and_deterministic() -> None:
    first = make_temporal_split(100, seed=21)
    second = make_temporal_split(100, seed=21)
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
