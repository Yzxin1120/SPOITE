import numpy as np

from spoite.causal import NuisanceConfig, crossfit_dr, validation_dr
from spoite.data import SyntheticConfig, generate_synthetic


def test_crossfit_is_complete_and_finite():
    d = generate_synthetic(200, SyntheticConfig(), 10)
    gamma, nuis = crossfit_dr(
        d.X, d.A, d.Y, NuisanceConfig(folds=5, seed=10)
    )
    assert np.isfinite(gamma).all()
    assert set(nuis["fold_id"]) == set(range(5))
    assert np.all((nuis["e"] > 0) & (nuis["e"] < 1))


def test_validation_nuisances_do_not_need_truth():
    tr = generate_synthetic(200, SyntheticConfig(), 1)
    va = generate_synthetic(80, SyntheticConfig(), 2)
    gamma, nuis = validation_dr(
        tr.X, tr.A, tr.Y, va.X, va.A, va.Y, NuisanceConfig(seed=1)
    )
    assert gamma.shape == (80,)
    assert np.isfinite(gamma).all()
    assert nuis["e"].shape == (80,)
