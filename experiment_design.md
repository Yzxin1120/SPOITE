# SPOITE Experiment Design

Status: **FROZEN BASELINE v1.0 — execution parameters approved**

Date frozen: 2026-07-04

Open execution parameters were approved and finalized on 2026-07-04. Their exact values and provenance are recorded in
`SPOITE_experiment/configs/final.json` and
`SPOITE_experiment/DESIGN_CHANGELOG.md`. Those records resolve every
`APPROVAL REQUIRED` marker below without changing the frozen scientific
structure.

This document is the governing specification for the new
`SPOITE_experiment/` implementation. The implementation must answer the
empirical questions stated by SPOITE itself. The existing `dccpo/` and
`experiment/` directories are reference material only; neither is the source of
truth for the new experiment.

## 1. Change-control rule

No material change to this design may be made without first reporting it to the
user and receiving explicit approval.

A material change includes:

- adding, removing, or redefining a dataset, method, metric, experiment factor,
  comparison, or headline claim;
- changing the data split, randomization, nuisance estimation, tuning,
  bootstrap, decision problem, or inference protocol;
- moving an appendix experiment into the main experiment;
- substituting a proxy for a specified exact quantity;
- filling a parameter marked `APPROVAL REQUIRED`;
- changing code in a way that alters the scientific estimand or comparison.

Every proposed change must be presented before implementation using:

1. proposed change;
2. reason;
3. affected experiments and outputs;
4. expected scientific consequence;
5. alternatives considered.

Implementation bugs may be diagnosed freely. A bug fix that could change
scientific results must still be reported before it is applied.

Approved changes must be recorded in
`SPOITE_experiment/DESIGN_CHANGELOG.md` and reflected in this document.

## 2. Scientific scope

The main experiments use the SPOITE framework:

```text
data
  -> train/validation/test split
  -> cross-fitted nuisance estimation
  -> standard doubly robust pseudo-outcome
  -> shared feature/model class
  -> MSE-DR / MAE-DR / CPO+ / DC-CPO
  -> shared constrained allocation oracle
  -> shared evaluation and inference
```

Optimized weight clipping, OCDR, and Heaviside/PIP optimization are not part of
the SPOITE main framework and must not enter the main experiments.

Fixed symmetric propensity clipping may be studied only in the pre-declared
appendix ablation in Section 10. It must not be called OCDR and must not be
presented as an implementation of Liu et al. (2026).

Active sampling is outside the scope of this paper and this implementation.

## 3. Shared data contract

Every synthetic or semi-synthetic dataset adapter must return the same logical
object:

```python
DatasetBundle(
    X=...,
    A=...,
    Y=...,
    tau_true=...,   # evaluation only; never available to a learner
    e_true=...,     # optional; evaluation/diagnostics only
    dataset_id=...,
    metadata=...,
)
```

The data layer generates or loads the quantities required for evaluation. It
does not calculate post-training metrics.

The learner may access only the training `X`, `A`, and `Y`, plus known
operational costs and constraints. `tau_true`, potential outcomes, and `e_true`
must never enter training or hyperparameter selection except where a synthetic
validation-regret protocol is explicitly approved.

## 4. Shared methods

The primary method set is:

1. `Oracle` - evaluation reference only;
2. `MSE-DR`;
3. `MAE-DR`;
4. `CPO+`;
5. `DC-CPO(band)` - primary DC-CPO implementation using the band choice allowed
   by SPOITE Equation 19.

`DC-CPO(inv_margin)` is a numerical-sensitivity appendix method. It is not a
headline method because uncapped inverse-margin weights can diverge.

All learned methods must use the same standard cross-fitted DR pseudo-outcome
within a paired run. No method may receive a more favorable nuisance estimate,
data split, cost vector, decision batch, or bootstrap sample.

## 5. Shared downstream decision problem

For every decision batch, define:

```text
benefit_i = tau_hat(X_i) - cost_i
```

and solve:

```text
maximize    sum_i benefit_i * w_i
subject to  sum_i w_i <= B
            L_g <= sum_{i:G_i=g} w_i <= U_g
            0 <= w_i <= 1
```

Primary decision size: `m = 16`. This permits exact enumeration of feasible
vertices for the primary DFI calculation.

Primary budget fraction: `B / m = 0.30`, except when budget tightness is the
declared experimental factor.

The group construction, cost distribution, and integer rounding rule are
`APPROVAL REQUIRED` before the first result-producing run. Once approved, they
must be fixed across methods and recorded in configuration files.

## 6. Shared metrics

Primary metrics:

- PEHE;
- regret;
- normalized regret;
- exact decision agreement;
- Decision Fragility Index (DFI);
- boundary-crossing frequency.

### 6.1 DFI protocol

The primary `m = 16` experiment must enumerate all feasible decision vertices
and implement Definition 5.1 over every competitor with positive margin. It
must not use the convention "no competitor within rho means DFI = 0."

Estimator covariance must be obtained with a full-pipeline bootstrap:

```text
resample training observations
  -> repeat nuisance cross-fitting
  -> rebuild the DR pseudo-outcome
  -> refit the same learner with the same configuration
  -> predict on the fixed evaluation set
```

All methods within a paired run must share the bootstrap resample indices.

For secondary experiments where exact enumeration is infeasible, a top-K
approximation may be used only if:

- it is labeled `top-K DFI proxy`;
- K is recorded;
- no empty active-set value is silently converted to zero;
- an active/competitor coverage diagnostic is reported.

Final DFI bootstrap count: at least 100 replicates. Smoke tests may use fewer,
but smoke-test outputs must never be cited as results.

### 6.2 Statistical reporting

Primary method comparisons use paired seeds and report:

- paired mean difference;
- 95% confidence interval;
- median and interquartile range for heavy-tailed metrics such as DFI;
- per-seed win rate;
- divergence/failure rate.

The final confidence-interval construction is `APPROVAL REQUIRED` before the
first result-producing run. The default proposal is a nonparametric bootstrap
over paired seeds.

## 7. Experiment 1 - core synthetic study for Q1-Q3

Purpose:

- Q1: Does CPO+ outperform MSE-DR and MAE-DR under model
  misspecification?
- Q2: Does DC-CPO improve regret and decision fragility beyond CPO+?
- Q3: Can lower PEHE fail to imply lower regret?

Fixed factors:

| Factor | Values |
|---|---|
| CATE regime | near-linear, nonlinear |
| Covariate distribution | correlated Gaussian |
| Overlap | medium, `gamma_e = 2` |
| Budget fraction | `B / m = 0.30` |
| Decision size | `m = 16` |
| Estimator | standard cross-fitted DR |
| Methods | Oracle, MSE-DR, MAE-DR, CPO+, DC-CPO(band) |
| Final seeds | `0..49` |

The shared feature class must represent the near-linear regime and remain
misspecified for the nonlinear regime.

Training, validation, and test sample sizes and the final tuning grids are
`APPROVAL REQUIRED`. They must be fixed before the first result-producing run.

Primary outputs:

- one table for PEHE, regret, normalized regret, DFI, and crossing;
- paired CPO+ versus MSE/MAE regret contrasts;
- paired DC-CPO versus CPO+ regret/DFI/crossing contrasts;
- PEHE-regret scatter plot;
- PEHE/regret ranking-reversal rate.

The primary Q3 statistic is the paired reversal frequency:

```text
P(PEHE_A < PEHE_B and regret_A > regret_B)
```

A pooled correlation may be shown as a secondary descriptive statistic, but
correlated method/scenario observations must not be treated as independent
samples for a p-value.

## 8. Experiment 2 - Theorem 5.3 geometry

Use the Appendix A.4 construction:

```text
S = conv{w0, v}
w0 = (0, 0)
v  = (-1, 0)
b0 = (1, 0)

Sigma_1 = diag(1/a, a)
Sigma_2 = diag(a, 1/a)
a in {4, 16, 64}
```

Both determinant and trace are equal across orientations. Report:

- confidence ellipsoids;
- the decision boundary;
- exact DFI;
- Monte Carlo crossing frequency;
- Monte Carlo regret.

This experiment is independent of the causal data pipeline. Weight clipping is
irrelevant to it.

## 9. Experiment 3 - Q4 hard-regime study

Purpose: determine whether the benefit of DC-CPO over CPO+ grows under weak
overlap, long-tail covariates, or tight resource constraints.

Fix the nonlinear misspecified CATE regime and standard DR.

Primary factorial matrix:

| Factor | Values |
|---|---|
| Overlap `gamma_e` | `0.5`, `2`, `4` |
| Covariate tail | Gaussian, rare mixture |
| Budget fraction | `0.15`, `0.30`, `0.45` |
| Methods | CPO+, DC-CPO(band) |
| Final seeds | `0..49` |

The rare-mixture probability and shift are `APPROVAL REQUIRED`.

`gamma_e = 4` is a near-positivity-violation stress test. Results from that
regime must not be claimed to fall under a theorem requiring uniformly bounded
overlap.

Primary effect:

```text
DC_gain(metric) = metric(CPO+) - metric(DC-CPO)
```

Positive gain means DC-CPO is better for a lower-is-better metric.

PEHE and regret are computed for every factorial cell and seed.

Full-pipeline DFI/bootstrap is restricted to a pre-declared extreme-cell subset
to control computation:

| Factor | Extreme values |
|---|---|
| Overlap | `0.5`, `4` |
| Covariate tail | Gaussian, rare mixture |
| Budget fraction | `0.15`, `0.45` |

The final number of seeds for the fragility subset must be at least 20 and is
`APPROVAL REQUIRED` before execution.

## 10. Appendix experiment - fixed propensity clipping

This experiment may begin only after Experiments 1-4 are implemented and their
main outputs are frozen.

Use symmetric fixed clipping:

```text
e_clipped = min(max(e_hat, epsilon), 1 - epsilon)
epsilon in {0.01, 0.025, 0.05}
```

Run only on pre-declared hard-regime cells and compare:

```text
standard DR vs fixed-clipped DR
    x
CPO+ vs DC-CPO(band)
```

This is a finite-sample sensitivity ablation. It is not OCDR, is not
policy-dependent, and must not inherit Liu et al.'s optimal-threshold claims.

## 11. Experiment 4 - semi-synthetic benchmark validation

Use the same shared pipeline and standard DR on:

| Dataset | Primary role |
|---|---|
| IHDP, 100 realizations | Q1-Q3 validation |
| ACIC, full/one-term settings and available replicates | Q1-Q4, especially overlap |
| Twins | Q1-Q3; assignment-temperature sensitivity may support Q4 |

Each dataset is reported separately. Observations from different datasets must
not be pooled as if they were independent draws from one population.

These are semi-synthetic benchmarks, not real-data external-validity evidence.

Dataset-specific treatment construction, realization selection, train/
validation/test rules, and replicate counts are `APPROVAL REQUIRED` before the
first result-producing run. The existing data and loaders may be audited and
reused only after they satisfy the shared data contract and leakage tests.

## 12. Pairing and tuning requirements

Within every paired comparison, methods must share:

- raw generated/loaded data;
- train/validation/test indices;
- nuisance folds and pseudo-outcomes;
- cost vectors;
- decision batches and group labels;
- oracle and tie-breaking;
- bootstrap indices.

Hyperparameters must be selected on validation data only. The test set is
evaluated once after selection.

CPO+ and DC-CPO must both receive tuning for shared optimization parameters
such as learning rate, L2 penalty, and training iterations. DC-CPO may
additionally tune its boundary penalty and active-band parameter.

No method may be selected or reconfigured after inspecting its test result.

## 13. Execution stages

Implementation and execution proceed in this order:

1. audit and test data adapters;
2. implement and unit-test the shared core;
3. run tiny smoke tests that produce no citable result;
4. obtain approval for all `APPROVAL REQUIRED` items;
5. freeze configuration files;
6. run Experiment 1;
7. verify Experiment 2;
8. run Experiment 3;
9. run Experiment 4;
10. optionally request approval to run the clipping appendix.

Passing one stage does not authorize changing a later stage's design.

## 14. Reproducibility requirements

Every result row must record:

- git commit;
- experiment/config identifier;
- dataset and regime;
- seed;
- data split identifier;
- nuisance configuration;
- method configuration;
- decision-problem configuration;
- metric/DFI configuration;
- runtime and failure status.

Smoke, provisional, stale, and final outputs must be stored separately.

Large generated outputs are not source code and should not be committed unless
explicitly designated as final tables or figures.
