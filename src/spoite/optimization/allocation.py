"""Exact oracle for the approved two-group binary allocation polytope."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class AllocationProblem:
    groups: np.ndarray
    budget: int
    lower: tuple[int, int] = (0, 0)
    upper: tuple[int, int] = (4, 4)

    def __post_init__(self):
        groups = np.asarray(self.groups, dtype=int)
        if groups.ndim != 1 or set(np.unique(groups)) != {0, 1}:
            raise ValueError("exactly two nonempty groups are required")
        if sum(self.lower) > self.budget or self.budget > sum(self.upper):
            raise ValueError("infeasible allocation bounds")
        object.__setattr__(self, "groups", groups)

    @property
    def m(self) -> int:
        return len(self.groups)


def make_problem(m: int, budget_fraction: float) -> AllocationProblem:
    if m % 2:
        raise ValueError("m must be even")
    budget = int(np.floor(budget_fraction * m + 0.5))
    upper = min(m // 2, int(np.ceil(0.75 * budget)))
    return AllocationProblem(
        groups=np.repeat((0, 1), m // 2),
        budget=budget,
        lower=(0, 0),
        upper=(upper, upper),
    )


class AllocationOracle:
    def __init__(self, problem: AllocationProblem):
        self.problem = problem
        self._group_idx = tuple(np.flatnonzero(problem.groups == g) for g in (0, 1))
        self._count_pairs = tuple(
            (k0, k1)
            for k0 in range(problem.lower[0], problem.upper[0] + 1)
            for k1 in range(problem.lower[1], problem.upper[1] + 1)
            if k0 + k1 <= problem.budget
        )

    def __call__(self, benefit: np.ndarray) -> tuple[float, np.ndarray]:
        b = np.asarray(benefit, dtype=float)
        if b.shape != (self.problem.m,) or not np.isfinite(b).all():
            raise ValueError("benefit has invalid shape or non-finite entries")
        orders = tuple(
            idx[np.lexsort((idx, -b[idx]))] for idx in self._group_idx
        )
        prefix = tuple(np.r_[0.0, np.cumsum(b[order])] for order in orders)
        objectives = np.fromiter(
            (prefix[0][k0] + prefix[1][k1] for k0, k1 in self._count_pairs),
            dtype=float,
        )
        # np.argmax supplies deterministic count-pair tie-breaking.
        k0, k1 = self._count_pairs[int(np.argmax(objectives))]
        chosen = np.r_[orders[0][:k0], orders[1][:k1]]
        best = np.zeros(self.problem.m)
        best[chosen] = 1.0
        best_obj = float(b @ best)
        return best_obj, best


def enumerate_vertices(problem: AllocationProblem) -> np.ndarray:
    group_idx = [np.flatnonzero(problem.groups == g) for g in (0, 1)]
    rows: list[np.ndarray] = []
    for k0 in range(problem.lower[0], problem.upper[0] + 1):
        for k1 in range(problem.lower[1], problem.upper[1] + 1):
            if k0 + k1 > problem.budget:
                continue
            for c0 in combinations(group_idx[0], k0):
                for c1 in combinations(group_idx[1], k1):
                    w = np.zeros(problem.m)
                    w[list(c0) + list(c1)] = 1.0
                    rows.append(w)
    return np.stack(rows)
