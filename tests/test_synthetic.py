import numpy as np

from spoite.data.synthetic import SyntheticConfig, cate, generate_synthetic
from spoite.methods import QuadraticFeaturizer, fit_mse


def test_cate_definitions():
    X = np.zeros((2, 10))
    X[1, :4] = [1, 2, 3, 4]
    near = cate(X, "near_linear")
    assert np.allclose(near, [1.0, 1 + 0.5 - 1 + 0.3 * 3 * 4])
    nonlinear = cate(X, "nonlinear")
    expected0 = 1 - 0.6
    expected1 = 1 + 0.5 - 1 + 0.9 - 0.6 * np.cos(4) + 0.45 * np.sin(6)
    assert np.allclose(nonlinear, [expected0, expected1])


def test_generator_reproducible_and_true_propensity_unclipped():
    cfg = SyntheticConfig(gamma_e=4, tail="rare_mixture")
    a = generate_synthetic(1000, cfg, 5)
    b = generate_synthetic(1000, cfg, 5)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.A, b.A)
    assert np.array_equal(a.Y, b.Y)
    assert np.array_equal(a.e_true, b.e_true)
    assert a.metadata["true_propensity_clipped"] is False
    assert a.e_true.min() < 0.01 or a.e_true.max() > 0.99
    assert a.metadata["rare_count"] > 0


def test_shared_feature_class_represents_only_near_linear_cate():
    X = np.random.default_rng(44).normal(size=(500, 10))
    phi = QuadraticFeaturizer().fit_transform(X)
    near = cate(X, "near_linear")
    nonlinear = cate(X, "nonlinear")
    assert np.sqrt(np.mean((phi @ fit_mse(phi, near, 0.0) - near) ** 2)) < 1e-10
    assert np.sqrt(np.mean((phi @ fit_mse(phi, nonlinear, 0.0) - nonlinear) ** 2)) > 0.05
