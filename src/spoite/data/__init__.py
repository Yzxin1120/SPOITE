"""Dataset contracts and synthetic/semi-synthetic adapters."""

from .base import DatasetBundle
from .benchmarks import (
    acic_outcome_files, load_acic_split, load_ihdp_split, load_twins_split,
)
from .synthetic import SyntheticConfig, cate, generate_synthetic

__all__ = [
    "DatasetBundle", "SyntheticConfig", "cate", "generate_synthetic",
    "acic_outcome_files", "load_acic_split", "load_ihdp_split",
    "load_twins_split",
]
