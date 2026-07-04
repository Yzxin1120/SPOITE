"""Shared, immutable data contract for every SPOITE dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DatasetBundle:
    """Observed causal data plus evaluation-only ground truth.

    Learners may consume only ``X``, ``A``, and ``Y``. ``tau_true`` and
    ``e_true`` are held for evaluation and overlap diagnostics.

    Raw covariates may contain NaN values (the Twins benchmark does). Dataset
    adapters must preserve them; imputation belongs in a fitted preprocessing
    stage and must never inspect evaluation outcomes.
    """

    X: FloatArray
    A: IntArray
    Y: FloatArray
    tau_true: FloatArray
    e_true: FloatArray | None
    dataset_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        A = np.asarray(self.A, dtype=np.int64)
        Y = np.asarray(self.Y, dtype=float)
        tau = np.asarray(self.tau_true, dtype=float)
        e = None if self.e_true is None else np.asarray(self.e_true, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be two-dimensional; got shape {X.shape}")
        n = X.shape[0]
        for name, value in (("A", A), ("Y", Y), ("tau_true", tau)):
            if value.shape != (n,):
                raise ValueError(
                    f"{name} must have shape ({n},); got {value.shape}"
                )
        if e is not None and e.shape != (n,):
            raise ValueError(f"e_true must have shape ({n},); got {e.shape}")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if not np.isin(A, (0, 1)).all():
            raise ValueError("A must contain only binary values 0 and 1")
        if np.isinf(X).any():
            raise ValueError("X may contain NaN but must not contain infinity")
        if not np.isfinite(Y).all():
            raise ValueError("Y must be finite")
        if not np.isfinite(tau).all():
            raise ValueError("tau_true must be finite")
        if e is not None:
            if not np.isfinite(e).all():
                raise ValueError("e_true must be finite when provided")
            if ((e < 0.0) | (e > 1.0)).any():
                raise ValueError("e_true must lie in [0, 1]")

        # Normalize representations once so downstream code sees one contract.
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "tau_true", tau)
        object.__setattr__(self, "e_true", e)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def treatment_rate(self) -> float:
        return float(np.mean(self.A))

    @property
    def missing_x_fraction(self) -> float:
        return float(np.mean(np.isnan(self.X)))

    def observed(self) -> tuple[FloatArray, IntArray, FloatArray]:
        """Return only the quantities that a learner is allowed to consume."""

        return self.X, self.A, self.Y
