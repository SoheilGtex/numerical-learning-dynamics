"""Controlled synthetic data for early-semester forecasting checkpoints."""

from dataclasses import dataclass

import numpy as np

CHECKPOINTS = ("pre_semester", "early_semester", "mid_semester", "later_semester")
FEATURE_NAMES = (
    "prior_gpa",
    "prerequisite_grade",
    "early_attendance",
    "quiz_1",
    "assignment_performance",
    "quiz_2",
    "midterm",
    "updated_attendance",
)
CHECKPOINT_FEATURES = {
    "pre_semester": FEATURE_NAMES[:2],
    "early_semester": FEATURE_NAMES[:4],
    "mid_semester": FEATURE_NAMES[:6],
    "later_semester": FEATURE_NAMES[:8],
}


@dataclass(frozen=True)
class TemporalDataset:
    """One synthetic student sample with explicitly nested information sets."""

    X: np.ndarray
    y: np.ndarray
    latent_ability: np.ndarray
    feature_names: tuple[str, ...]
    checkpoint_features: dict[str, tuple[str, ...]]

    def feature_indices(self, checkpoint: str) -> tuple[int, ...]:
        """Return feature indices available at a named checkpoint."""
        if checkpoint not in self.checkpoint_features:
            raise KeyError(f"unknown checkpoint: {checkpoint}")
        return tuple(self.feature_names.index(name) for name in self.checkpoint_features[checkpoint])

    def checkpoint_matrix(self, checkpoint: str) -> np.ndarray:
        """Return only features available by the requested checkpoint."""
        return self.X[:, self.feature_indices(checkpoint)]


@dataclass(frozen=True)
class TemporalSplit:
    """A deterministic student partition reused at every checkpoint."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def generate_temporal_data(n_samples: int = 240, seed: int = 5200) -> TemporalDataset:
    """Generate illustrative, non-empirical information accruing through a semester.

    Each observable is an independent noisy measurement of a persistent latent
    ability. The target has an independent outcome-noise term, so later
    checkpoints become more informative without becoming deterministic.
    """
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, n_samples)
    loadings = np.array([0.75, 0.85, 0.70, 0.80, 0.78, 0.82, 0.88, 0.72])
    noise_scales = np.array([0.70, 0.65, 0.75, 0.70, 0.68, 0.65, 0.60, 0.70])
    measurements = latent[:, None] * loadings + rng.normal(
        0.0, noise_scales, size=(n_samples, len(FEATURE_NAMES))
    )
    X = np.empty_like(measurements)
    X[:, 0] = 13.0 + 2.0 * measurements[:, 0]
    X[:, 1] = 13.0 + 2.0 * measurements[:, 1]
    X[:, 2] = np.clip(0.82 + 0.08 * measurements[:, 2], 0.0, 1.0)
    X[:, 3] = 10.0 + 2.0 * measurements[:, 3]
    X[:, 4] = 10.0 + 2.0 * measurements[:, 4]
    X[:, 5] = 10.0 + 2.0 * measurements[:, 5]
    X[:, 6] = 10.0 + 2.0 * measurements[:, 6]
    X[:, 7] = np.clip(0.82 + 0.08 * measurements[:, 7], 0.0, 1.0)
    y = 12.0 + 3.0 * latent + rng.normal(0.0, 0.9, n_samples)
    return TemporalDataset(
        X=X,
        y=y,
        latent_ability=latent,
        feature_names=FEATURE_NAMES,
        checkpoint_features={key: tuple(value) for key, value in CHECKPOINT_FEATURES.items()},
    )


def make_temporal_split(n_samples: int, seed: int = 8800) -> TemporalSplit:
    """Create a deterministic 60/20/20 split shared across all checkpoints."""
    if n_samples < 5:
        raise ValueError("n_samples must be at least 5")
    indices = np.random.default_rng(seed).permutation(n_samples)
    train_end = int(0.6 * n_samples)
    validation_end = int(0.8 * n_samples)
    return TemporalSplit(indices[:train_end], indices[train_end:validation_end], indices[validation_end:])
