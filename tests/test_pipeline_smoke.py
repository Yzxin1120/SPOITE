import numpy as np

from spoite.causal import NuisanceConfig, crossfit_dr
from spoite.data import SyntheticConfig, generate_synthetic
from spoite.evaluation import decision_metrics, pehe
from spoite.methods import QuadraticFeaturizer, fit_cpo, fit_mae, fit_mse, make_instances
from spoite.optimization import AllocationOracle, enumerate_vertices, make_problem


def test_all_shared_learners_complete_smoke_run():
    d = generate_synthetic(160, SyntheticConfig(), 3)
    gamma, _ = crossfit_dr(d.X, d.A, d.Y, NuisanceConfig(seed=3))
    feat = QuadraticFeaturizer()
    phi = feat.fit_transform(d.X)
    cost = np.random.default_rng(3).uniform(0.3, 1.2, len(d.X))
    inst = make_instances(phi, gamma, cost, d.X, 16, 3)
    problem = make_problem(16, 0.30)
    oracle = AllocationOracle(problem)
    vertices = enumerate_vertices(problem)
    fits = {
        "mse": fit_mse(phi, gamma, 1e-3),
        "mae": fit_mae(phi, gamma, 1e-3, 0.05, 2, 3),
        "cpo": fit_cpo(inst, oracle, 1e-3, 0.05, 2, 3),
        "dc": fit_cpo(
            inst, oracle, 1e-3, 0.05, 2, 3,
            boundary_penalty=0.1, outer_rounds=2, vertices=vertices,
        ),
    }
    for theta in fits.values():
        pred = phi @ theta
        assert np.isfinite(pred).all()
        assert np.isfinite(pehe(pred, d.tau_true))
        out = decision_metrics(pred, d.tau_true, cost, inst.batches, oracle)
        assert all(np.isfinite(v) for v in out.values())
