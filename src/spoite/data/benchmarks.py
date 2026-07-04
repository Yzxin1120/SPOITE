"""Leakage-safe IHDP and ACIC benchmark adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import DatasetBundle


def _bundle(X, A, Y, tau, e, dataset_id, metadata):
    return DatasetBundle(
        X=np.asarray(X, float), A=np.asarray(A, int), Y=np.asarray(Y, float),
        tau_true=np.asarray(tau, float),
        e_true=None if e is None else np.asarray(e, float),
        dataset_id=dataset_id, metadata=metadata,
    )


def load_ihdp_split(
    data_root: str | Path, realization: int, seed: int = 20260704
) -> tuple[DatasetBundle, DatasetBundle, DatasetBundle]:
    root = Path(data_root) / "IHDP"
    train_raw = np.load(root / "ihdp_npci_1-100.train.npz")
    test_raw = np.load(root / "ihdp_npci_1-100.test.npz")

    def arrays(raw):
        X = raw["x"][:, :, realization].astype(float)
        A = raw["t"][:, realization].astype(int)
        Y = raw["yf"][:, realization].astype(float)
        tau = (raw["mu1"][:, realization] - raw["mu0"][:, realization]).astype(float)
        return X, A, Y, tau

    X, A, Y, tau = arrays(train_raw)
    fit_idx, val_idx = train_test_split(
        np.arange(len(A)), test_size=0.20, random_state=seed,
        stratify=A,
    )
    Xt, At, Yt, taut = arrays(test_raw)
    common = {"realization": realization, "true_propensity_available": False}
    return (
        _bundle(X[fit_idx], A[fit_idx], Y[fit_idx], tau[fit_idx], None,
                f"ihdp-r{realization}-train", {**common, "split": "train"}),
        _bundle(X[val_idx], A[val_idx], Y[val_idx], tau[val_idx], None,
                f"ihdp-r{realization}-validation", {**common, "split": "validation"}),
        _bundle(Xt, At, Yt, taut, None,
                f"ihdp-r{realization}-test", {**common, "split": "canonical_test"}),
    )


def acic_outcome_files(data_root: str | Path) -> list[Path]:
    return sorted(
        p for p in (Path(data_root) / "ACIC/generated_outcomes").glob("*.csv")
        if p.name != "MANIFEST.csv"
    )


def load_acic_split(
    data_root: str | Path, outcome_file: str | Path, seed: int = 20260704
) -> tuple[DatasetBundle, DatasetBundle, DatasetBundle]:
    root = Path(data_root) / "ACIC"
    cov = pd.read_csv(root / "acic2016_covariates.csv")
    outcome_file = Path(outcome_file)
    if not outcome_file.is_absolute():
        outcome_file = root / "generated_outcomes" / outcome_file
    out = pd.read_csv(outcome_file)
    A = out["z"].to_numpy(int)
    idx_train, idx_rest = train_test_split(
        np.arange(len(A)), train_size=0.60, random_state=seed, stratify=A
    )
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.50, random_state=seed + 1, stratify=A[idx_rest]
    )
    categorical = cov.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric = [c for c in cov.columns if c not in categorical]
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ), categorical),
    ])
    prep.fit(cov.iloc[idx_train])
    tau = (out["mu1"] - out["mu0"]).to_numpy(float)
    e = out["e"].to_numpy(float)
    Y = out["y"].to_numpy(float)
    stem = outcome_file.stem

    def make(idx, split):
        X = prep.transform(cov.iloc[idx])
        return _bundle(
            X, A[idx], Y[idx], tau[idx], e[idx], f"{stem}-{split}",
            {
                "outcome_file": outcome_file.name, "split": split,
                "preprocessor_fit_split": "train",
                "raw_categorical_columns": categorical,
            },
        )
    return make(idx_train, "train"), make(idx_val, "validation"), make(idx_test, "test")


def load_twins_split(
    data_root: str | Path,
    temperature: float,
    assignment_seed: int,
    subsample: int = 5000,
) -> tuple[DatasetBundle, DatasetBundle, DatasetBundle]:
    root = Path(data_root) / "Twins"
    Xdf = pd.read_csv(root / "twin_pairs_X_3years_samesex.csv")
    Xdf = Xdf.drop(columns=[c for c in Xdf if c.startswith("Unnamed")])
    X = Xdf.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    outcomes = pd.read_csv(root / "twin_pairs_Y_3years_samesex.csv")
    y0 = outcomes["mort_0"].to_numpy(float)
    y1 = outcomes["mort_1"].to_numpy(float)
    valid = np.flatnonzero(np.isfinite(y0) & np.isfinite(y1))
    rng = np.random.default_rng(assignment_seed)
    chosen = np.sort(rng.choice(valid, size=min(subsample, len(valid)), replace=False))
    X, y0, y1 = X[chosen], y0[chosen], y1[chosen]

    # A construction-only subset fixes the assignment mechanism before the
    # learner split. It uses X only and is recorded separately for provenance.
    construction, _ = train_test_split(
        np.arange(len(X)), train_size=0.60, random_state=assignment_seed
    )
    score_pipe = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=1)
    )
    score_pipe.fit(X[construction])
    score = score_pipe.transform(X).ravel()
    construction_imputed = score_pipe.named_steps["simpleimputer"].transform(
        X[construction]
    )
    nonconstant = np.flatnonzero(np.std(construction_imputed, axis=0) > 1e-12)
    if not len(nonconstant):
        raise ValueError("Twins construction covariates are all constant")
    orientation_feature = construction_imputed[:, nonconstant[0]]
    if np.corrcoef(score[construction], orientation_feature)[0, 1] < 0:
        score = -score
    score = (score - score[construction].mean()) / (
        score[construction].std() + 1e-12
    )
    e = 1.0 / (1.0 + np.exp(-temperature * score))
    A = rng.binomial(1, e).astype(int)
    Y = np.where(A == 1, y1, y0)
    tau = y1 - y0
    idx_train, idx_rest = train_test_split(
        np.arange(len(X)), train_size=0.60, random_state=assignment_seed + 1,
        stratify=A,
    )
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.50, random_state=assignment_seed + 2,
        stratify=A[idx_rest],
    )
    common = {
        "temperature": temperature,
        "assignment_seed": assignment_seed,
        "subsample": len(X),
        "assignment_direction": "oriented_PC1",
        "orientation_feature_index": int(nonconstant[0]),
        "assignment_fit_rows": construction.tolist(),
        "raw_columns": 54,
        "analytic_columns_after_dropping_index": int(X.shape[1]),
    }

    def make(idx, split):
        return _bundle(
            X[idx], A[idx], Y[idx], tau[idx], e[idx],
            f"twins-t{temperature:g}-s{assignment_seed}-{split}",
            {**common, "split": split},
        )
    return make(idx_train, "train"), make(idx_val, "validation"), make(idx_test, "test")
