import itertools

import numpy as np

from spoite.optimization import AllocationOracle, enumerate_vertices, make_problem


def test_oracle_matches_complete_vertex_enumeration():
    p = make_problem(16, 0.30)
    vertices = enumerate_vertices(p)
    oracle = AllocationOracle(p)
    rng = np.random.default_rng(4)
    for _ in range(10):
        b = rng.normal(size=16)
        obj, w = oracle(b)
        exhaustive = vertices @ b
        assert np.isclose(obj, exhaustive.max())
        assert np.isclose(obj, b @ w)
        assert any(np.array_equal(w, v) for v in vertices)


def test_budget_rounding_and_group_cap():
    for fraction, budget in ((0.15, 2), (0.30, 5), (0.45, 7)):
        p = make_problem(16, fraction)
        assert p.budget == budget
        assert p.upper == (int(np.ceil(0.75 * budget)),) * 2
