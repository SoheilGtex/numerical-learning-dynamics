import numpy as np

from numerical_learning.forecasting import contaminate_training, prepare_checkpoint
from numerical_learning.temporal_data import generate_temporal_data, make_temporal_split


def test_zero_contamination_preserves_training_data() -> None:
    dataset = generate_temporal_data(seed=31)
    split = make_temporal_split(len(dataset.y), seed=32)
    data = prepare_checkpoint(dataset, split, "later_semester")
    corrupted = contaminate_training(data, 0.0, seed=33, mode="response")
    np.testing.assert_array_equal(corrupted.train_X, data.train_X)
    np.testing.assert_array_equal(corrupted.train_y, data.train_y)
    np.testing.assert_array_equal(corrupted.raw_validation_X, data.raw_validation_X)
    np.testing.assert_array_equal(corrupted.raw_test_X, data.raw_test_X)


def test_contamination_is_training_only_and_deterministic() -> None:
    dataset = generate_temporal_data(seed=34)
    split = make_temporal_split(len(dataset.y), seed=35)
    data = prepare_checkpoint(dataset, split, "later_semester")
    first = contaminate_training(data, 0.2, seed=36, mode="response")
    second = contaminate_training(data, 0.2, seed=36, mode="response")
    np.testing.assert_array_equal(first.train_y, second.train_y)
    np.testing.assert_array_equal(first.raw_validation_X, data.raw_validation_X)
    np.testing.assert_array_equal(first.raw_test_X, data.raw_test_X)
    assert not np.array_equal(first.train_y, data.train_y)


def test_feature_contamination_is_separate_from_response_contamination() -> None:
    dataset = generate_temporal_data(seed=37)
    split = make_temporal_split(len(dataset.y), seed=38)
    data = prepare_checkpoint(dataset, split, "later_semester")
    response = contaminate_training(data, 0.1, seed=39, mode="response")
    feature = contaminate_training(data, 0.1, seed=39, mode="feature")
    assert np.array_equal(response.train_X, data.train_X)
    assert np.array_equal(feature.train_y, data.train_y)
    assert not np.array_equal(feature.raw_train_X, data.raw_train_X)
    assert np.array_equal(feature.raw_validation_X, data.raw_validation_X)
    assert np.array_equal(feature.raw_test_X, data.raw_test_X)
    np.testing.assert_allclose(feature.train_X.mean(axis=0), 0.0, atol=1e-12)
