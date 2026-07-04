from pathlib import Path

import numpy as np
import pytest

from spoite.data import (
    acic_outcome_files, load_acic_split, load_ihdp_split, load_twins_split,
)

DATA = Path(__file__).resolve().parents[2] / "experiment/8.3/semi/data"


@pytest.mark.skipif(not DATA.exists(), reason="reference benchmark data unavailable")
def test_ihdp_canonical_and_validation_splits_are_disjoint_in_role():
    train, val, test = load_ihdp_split(DATA, 0)
    assert (train.n, val.n, test.n) == (537, 135, 75)
    assert train.p == val.p == test.p == 25
    assert train.e_true is None and test.e_true is None
    assert all(np.isfinite(d.tau_true).all() for d in (train, val, test))


@pytest.mark.skipif(not DATA.exists(), reason="reference benchmark data unavailable")
def test_acic_uses_all_files_and_train_fitted_encoding():
    files = acic_outcome_files(DATA)
    assert len(files) == 20
    train, val, test = load_acic_split(DATA, files[0])
    assert (train.n, val.n, test.n) == (2881, 960, 961)
    assert train.p == val.p == test.p
    assert all(np.isfinite(d.X).all() for d in (train, val, test))
    assert all(d.e_true is not None for d in (train, val, test))
    assert train.metadata["preprocessor_fit_split"] == "train"


@pytest.mark.skipif(not DATA.exists(), reason="reference benchmark data unavailable")
def test_twins_assignment_and_splits_are_reproducible():
    first = load_twins_split(DATA, 1.0, 0, subsample=500)
    second = load_twins_split(DATA, 1.0, 0, subsample=500)
    assert tuple(d.n for d in first) == (300, 100, 100)
    for a, b in zip(first, second):
        assert np.array_equal(a.A, b.A)
        assert np.array_equal(a.Y, b.Y)
        assert np.array_equal(a.e_true, b.e_true)
        assert a.X.shape[1] == 52
        assert a.metadata["raw_columns"] == 54
