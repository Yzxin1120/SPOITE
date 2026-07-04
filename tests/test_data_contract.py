"""Unit tests for the shared DatasetBundle contract."""

from __future__ import annotations

import numpy as np
import pytest

from spoite.data import DatasetBundle


def _bundle(**updates) -> DatasetBundle:
    values = {
        "X": np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),
        "A": np.array([0, 1, 0]),
        "Y": np.array([1.0, 2.0, 3.0]),
        "tau_true": np.array([0.1, 0.2, 0.3]),
        "e_true": np.array([0.2, 0.7, 0.4]),
        "dataset_id": "unit-test",
        "metadata": {"source": "synthetic"},
    }
    values.update(updates)
    return DatasetBundle(**values)


def test_contract_normalizes_shapes_and_exposes_observed_only() -> None:
    bundle = _bundle()
    assert bundle.n == 3
    assert bundle.p == 2
    assert bundle.treatment_rate == pytest.approx(1.0 / 3.0)
    assert bundle.missing_x_fraction == pytest.approx(1.0 / 6.0)
    X, A, Y = bundle.observed()
    assert X is bundle.X
    assert A is bundle.A
    assert Y is bundle.Y


def test_contract_allows_nan_covariates_but_rejects_infinity() -> None:
    assert np.isnan(_bundle().X).any()
    with pytest.raises(ValueError, match="infinity"):
        _bundle(X=np.array([[1.0], [np.inf], [3.0]]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("A", np.array([0, 2, 0]), "binary"),
        ("Y", np.array([1.0, np.nan, 3.0]), "Y must be finite"),
        (
            "tau_true",
            np.array([0.1, np.inf, 0.3]),
            "tau_true must be finite",
        ),
        ("e_true", np.array([0.2, 1.1, 0.4]), r"e_true must lie"),
    ],
)
def test_contract_rejects_invalid_causal_arrays(
    field: str, value: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _bundle(**{field: value})


def test_contract_rejects_row_mismatch() -> None:
    with pytest.raises(ValueError, match="tau_true must have shape"):
        _bundle(tau_true=np.array([0.1, 0.2]))


def test_metadata_is_immutable() -> None:
    bundle = _bundle()
    with pytest.raises(TypeError):
        bundle.metadata["source"] = "changed"
